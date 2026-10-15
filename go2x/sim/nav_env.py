"""Velocity-level navigation environment for a planar Go2 model.

Simulation only. The dynamics default to LagDelayDynamics, whose parameters are unfitted
assumptions (see config/sim.yaml). Nothing here describes measured Go2 X behavior.

Observation (all scaled to about [-1, 1]):
    goal distance / room diagonal bound, cos and sin of the goal bearing in the body frame,
    body velocity / limits, previous action, LiDAR ranges / max range.
Action: [-1, 1]^3, scaled to (vx, vy, wz) limits.

The one place a safety filter can act is `action_filter`: a callable
f(cmd, env) -> cmd applied in step() between the scaled policy output and the dynamics.
It defaults to None (no filter).
"""
import copy

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from go2x.config import load_config
from go2x.sim.dynamics import LagDelayDynamics
from go2x.sim.scene import sample_scene


class Go2NavEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, config=None, dynamics=None, action_filter=None, render_mode=None):
        cfg = copy.deepcopy(config) if config is not None else load_config("sim")
        self.cfg = cfg
        self.dt = cfg["dt"]
        self.max_steps = int(round(cfg["max_episode_s"] / self.dt))
        self.limits = np.array([cfg["limits"]["vx"], cfg["limits"]["vy"], cfg["limits"]["wz"]])
        self.radius = cfg["robot_radius"]
        self.n_beams = cfg["lidar"]["n_beams"]
        self.max_range = cfg["lidar"]["max_range"]
        self.beam_angles = np.arange(self.n_beams) * 2 * np.pi / self.n_beams
        self.dist_scale = np.hypot(*([cfg["scene"]["room_size_range"][1]] * 2))
        dc = cfg["dynamics"]
        self.dynamics = dynamics or LagDelayDynamics(dc["tau_range"], dc["delay_range"], dc["substeps"], self.dt)
        self.action_filter = action_filter
        self.render_mode = render_mode

        n_obs = 1 + 2 + 3 + 3 + self.n_beams
        self.observation_space = spaces.Box(-1.0, 1.0, (n_obs,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (3,), np.float32)
        self.scene = None
        self._fig = None

    # -- helpers ---------------------------------------------------------------------
    def goal_polar(self):
        """Distance and body-frame bearing to the goal."""
        x, y, yaw = self.dynamics.pose
        dx, dy = self.scene.goal - (x, y)
        bearing = (np.arctan2(dy, dx) - yaw + np.pi) % (2 * np.pi) - np.pi
        return np.hypot(dx, dy), bearing

    def lidar(self):
        x, y, yaw = self.dynamics.pose
        return self.scene.raycast((x, y), yaw + self.beam_angles, self.max_range)

    def _obs(self):
        dist, bearing = self.goal_polar()
        obs = np.concatenate([
            [min(dist / self.dist_scale, 1.0), np.cos(bearing), np.sin(bearing)],
            np.clip(self.dynamics.vel / self.limits, -1, 1),
            self.prev_action,
            self.lidar() / self.max_range,
        ])
        return obs.astype(np.float32)

    def _info(self, **extra):
        dist, bearing = self.goal_polar()
        return {"goal_distance": dist, "goal_bearing": bearing, "pose": self.dynamics.pose.copy(),
                "vel": self.dynamics.vel.copy(), **extra}

    # -- gymnasium API ---------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.scene = sample_scene(self.np_random, self.cfg["scene"], self.radius)
        self.dynamics.reset(self.np_random, self.scene.start)
        self.prev_action = np.zeros(3)
        self.steps = 0
        self.prev_dist = self.goal_polar()[0]
        return self._obs(), self._info()

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        cmd_policy = action * self.limits
        # The single insertion point for a safety filter (later phases).
        cmd = self.action_filter(cmd_policy, self) if self.action_filter else cmd_policy
        self.dynamics.step(cmd, self.dt)
        self.prev_action = action
        self.steps += 1

        dist = self.goal_polar()[0]
        collided = bool(self.scene.collides(self.dynamics.pose[:2], self.radius)[0])
        success = dist <= self.cfg["goal_tolerance"] and not collided
        rc = self.cfg["reward"]
        reward = rc["progress"] * (self.prev_dist - dist) + rc["time"]
        if success:
            reward += rc["success"]
        if collided:
            reward += rc["collision"]
        self.prev_dist = dist
        terminated = success or collided
        truncated = not terminated and self.steps >= self.max_steps
        info = self._info(is_success=success, collision=collided, cmd_policy=cmd_policy, cmd_applied=cmd,
                          time_s=self.steps * self.dt)
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Polygon, Rectangle

        if self._fig is None:
            self._fig, self._ax = plt.subplots(figsize=(5, 5), dpi=100)
        ax = self._ax
        ax.clear()
        sc = self.scene
        ax.add_patch(Rectangle((0, 0), sc.width, sc.depth, fill=False, lw=2, ec="#333333"))
        for cx, cy, r in sc.cylinders:
            ax.add_patch(Circle((cx, cy), r, fc="#8c8c8c", ec="none"))
        for cx, cy, yaw, hx, hy in sc.boxes:
            c, s = np.cos(yaw), np.sin(yaw)
            corners = [(cx + c * a - s * b, cy + s * a + c * b) for a, b in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))]
            ax.add_patch(Polygon(corners, fc="#8c8c8c", ec="none"))
        x, y, yaw = self.dynamics.pose
        ranges = self.lidar()
        for a, r in zip(yaw + self.beam_angles, ranges):
            ax.plot([x, x + r * np.cos(a)], [y, y + r * np.sin(a)], color="#e8a33d", lw=0.5, alpha=0.6)
        ax.add_patch(Circle(tuple(sc.goal), self.cfg["goal_tolerance"], fc="#3a9e5c", alpha=0.5, ec="none"))
        ax.add_patch(Circle((x, y), self.radius, fc="#2f6db5", alpha=0.8, ec="none"))
        ax.plot([x, x + self.radius * np.cos(yaw)], [y, y + self.radius * np.sin(yaw)], color="white", lw=2)
        ax.set_xlim(-0.2, sc.width + 0.2)
        ax.set_ylim(-0.2, sc.depth + 0.2)
        ax.set_aspect("equal")
        ax.set_title(f"t = {self.steps * self.dt:4.1f} s   goal distance {self.goal_polar()[0]:.2f} m", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        self._fig.canvas.draw()
        return np.asarray(self._fig.canvas.buffer_rgba())[..., :3].copy()

    def close(self):
        if self._fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
            self._fig = None

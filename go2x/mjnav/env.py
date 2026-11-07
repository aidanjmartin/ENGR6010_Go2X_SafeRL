"""Task 4 navigation in CPU MuJoCo: the navigation policy (10 Hz) over a frozen walking policy (50 Hz) on
Unitree's full go2.xml, in the RSL-RL 3.1.2 VecEnv interface (scripts/mjnav/train_nav.py, eval_nav.py).

Everything above the physics is the Isaac navigation task's own code, unchanged: the atrium layouts and moving
people (go2x/atrium/state.py AtriumBatch), the drawn top-down image with camera realism (go2x/atrium/topdown.py
CameraFeed, robot or world view), the reward terms (go2x/atrium/rewards.py), the action scaling, the
observation layout (stacked current and previous frame, then the previous normalized action) and the critic's
extra state (go2x/isaac/nav_env_cfg.py obs_critic_state). The physics is MuJoCo 3.16 on go2.xml with every
collision geom; PD at every 5 ms step (position actuators, go2x/mjx/go2_model.py); walking-policy observation
noise as in Isaac (config/walk.yaml obs_noise). The B robots step together in MuJoCo's rollout module.

Differences from the Isaac task:
- Obstacles are drawn and checked by footprint only; MuJoCo has no obstacle bodies here, so there is no contact
  force check (Isaac counted either). The robot runner uses the footprint check too.
- A fall is tilt above 0.8 rad or base below 0.15 m (scripts/mujoco_check.py), not a base contact force.
- Episodes start with the feet on the floor at the default pose.
"""
import math
import mujoco
import numpy as np
import torch
from mujoco import rollout
from tensordict import TensorDict

from go2x.atrium.rewards import TERMS, front_point, reward_terms, scale_action
from go2x.atrium.state import AtriumBatch
from go2x.atrium.topdown import CameraFeed, view_config
from go2x.config import load_config
from go2x.mjx import go2_model as G
from go2x.sim2sim import DEFAULT_POS, OBS_SCALES

try:
    from rsl_rl.env import VecEnv
except ImportError:          # the evaluation can run without RSL-RL
    VecEnv = object


class MujocoNavEnv(VecEnv):
    def __init__(self, num_envs, model_dir, walk_policy, seed=0, scenarios=None, view=None, device="cpu", threads=12,
                 walk_noise=True):
        self.NAV, self.ATRIUM, self.TD = load_config("nav"), load_config("atrium"), load_config("topdown")
        self.WALK, self.MJ = load_config("walk"), load_config("mujoco")
        self.num_envs, self.B, self.device = num_envs, num_envs, device
        self.num_actions = 3
        self.cfg = {"name": "mujoco_nav", "seed": seed}
        t = self.NAV["timing"]
        self.walk_dt = self.MJ["timestep"] * self.MJ["pd_substeps"]
        self.k_walk = t["walk_steps_per_nav"]
        self.step_dt = self.walk_dt * self.k_walk
        self.max_episode_length = int(round(t["episode_s"] / self.step_dt))
        dev = self.adev = torch.device(device)     # atrium, drawing and observations; physics and walker on the CPU
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=dev)
        # physics
        self.model = G.eval_model(model_dir, self.MJ["timestep"])
        ix = G.indices(self.model)
        self.qadr, self.vadr, self.act = ix["qadr"], ix["vadr"], ix["act"]
        G.use_position_actuators(self.model, self.WALK["actuator"]["kp"], self.WALK["actuator"]["kd"], self.act)
        self.z0 = G.standing_height(self.model)
        # All robot states live in one array (mjSTATE_FULLPHYSICS rows) that MuJoCo's rollout module steps in
        # C++ threads: 4 substeps for 1024 robots in about 8 ms on the Dell (20 threads).
        m = self.model
        self.spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
        self.state = np.zeros((num_envs, mujoco.mj_stateSize(m, self.spec)))
        self.sens = np.zeros((num_envs, m.nsensordata))
        self.o_q, self.o_v = 1, 1 + m.nq                       # time, qpos, qvel, act, ...
        self.ctrl = np.zeros((num_envs, self.MJ["pd_substeps"], m.nu))
        self.scratch = mujoco.MjData(m)
        self.threads = threads
        self.roll = rollout.Rollout(nthread=threads)
        self.rdatas = [mujoco.MjData(m) for _ in range(threads)]
        self.gyro_adr = m.sensor("imu_gyro").adr[0]
        self.quat_adr = m.sensor("imu_quat").adr[0]
        self.walk = torch.jit.load(walk_policy, map_location="cpu").eval()
        self.walk_noise = walk_noise
        self.ascale = self.WALK["actuator"]["action_scale"]
        self.rng = np.random.default_rng(seed + 11)
        # atrium (CPU torch)
        self.batch = AtriumBatch(num_envs, dev, seed=seed, dt=self.walk_dt, scenarios=scenarios)
        g = torch.Generator(device=dev)
        g.manual_seed(seed + 1)
        self.view = view or self.NAV["policy"]["image_view"]
        self.feed = CameraFeed(num_envs, view_config(self.TD, self.view), tuple(self.ATRIUM["robot"]["footprint"]), dev, g)
        S = self.TD["size"]
        self.cur = torch.zeros(num_envs, 4, S, S, device=dev)
        self.prev = torch.zeros_like(self.cur)
        self.first = torch.ones(num_envs, dtype=torch.bool, device=dev)
        self.action_n = torch.zeros(num_envs, 3, device=dev)
        self.prev_action_n = torch.zeros(num_envs, 3, device=dev)
        self.prev_dist = torch.zeros(num_envs, device=dev)
        self.walk_last = np.zeros((num_envs, 12), np.float32)
        self.ep_min_person_gap = torch.full((num_envs,), 1e3, device=dev)
        self.ep_fp_steps = torch.zeros(num_envs, device=dev)
        self.last_metrics, self.last_terms = None, None
        self.reset_ids(torch.arange(num_envs, device=dev))
        self._obs = self._observe()

    # ------------------------------------------------------------------------------------------- physics
    def _pose_speed(self):
        """Square-frame pose (B, 3), planar speed in the body frame (B,), tilt (B,), base height (B,)."""
        q = self.state[:, self.o_q:self.o_q + 7]
        v = self.state[:, self.o_v:self.o_v + 3]
        w, x, y, z = q[:, 3], q[:, 4], q[:, 5], q[:, 6]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        gz = -(1 - 2 * (x * x + y * y))                      # body-frame z of gravity (0, 0, -1)
        tilt = np.arccos(np.clip(-gz, -1, 1))
        # body-frame planar velocity: rotate world velocity by the full orientation
        R = np.stack([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]).transpose(2, 0, 1)
        vb = np.einsum("bji,bj->bi", R, v)
        pose = torch.tensor(np.stack([q[:, 0], q[:, 1], yaw], -1), dtype=torch.float32, device=self.adev)
        speed = torch.tensor(np.linalg.norm(vb[:, :2], axis=-1), dtype=torch.float32, device=self.adev)
        return pose, speed, tilt, q[:, 2], vb

    def _walk_obs(self, cmd):
        B = self.B
        gyro = self.sens[:, self.gyro_adr:self.gyro_adr + 3]
        quat = self.sens[:, self.quat_adr:self.quat_adr + 4]
        jq = self.state[:, self.o_q + self.qadr]
        jv = self.state[:, self.o_v + self.vadr]
        w, x, y, z = quat.T
        g = np.stack([-(2 * (x * z - w * y)), -(2 * (y * z + w * x)), -(1 - 2 * (x * x + y * y))], -1)
        if self.walk_noise:
            n = self.WALK["randomization"]["obs_noise"]
            u = lambda s, k: self.rng.uniform(-s, s, (B, k))  # noqa: E731
            gyro, g, jq, jv = gyro + u(n["base_ang_vel"], 3), g + u(n["projected_gravity"], 3), jq + u(n["joint_pos"], 12), jv + u(n["joint_vel"], 12)
        obs = np.concatenate([gyro * OBS_SCALES["ang_vel"], g, cmd, jq - DEFAULT_POS, jv * OBS_SCALES["joint_vel"],
                              self.walk_last], -1).astype(np.float32)
        return torch.from_numpy(obs)

    def _physics(self, targets):
        self.ctrl[:, :, self.act] = targets[:, None, :]
        st, sens = self.roll.rollout(self.model, self.rdatas, self.state, self.ctrl, nstep=self.MJ["pd_substeps"])
        self.state = st[:, -1].copy()
        self.sens = sens[:, -1].copy()

    # ------------------------------------------------------------------------------------------- episode
    def reset_ids(self, ids):
        if len(ids) == 0:
            return
        start = self.batch.reset(ids).cpu()
        d = self.scratch
        for k, e in enumerate(ids.tolist()):
            mujoco.mj_resetData(self.model, d)
            x, y, yaw = start[k].tolist()
            d.qpos[0:3] = [x, y, self.z0]
            d.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
            d.qpos[self.qadr] = DEFAULT_POS
            d.ctrl[self.act] = DEFAULT_POS
            mujoco.mj_forward(self.model, d)
            mujoco.mj_getState(self.model, d, self.state[e], self.spec)
            self.sens[e] = d.sensordata
        self.walk_last[ids.cpu().numpy()] = 0.0
        self.feed.reset(ids)
        self.first[ids] = True
        self.action_n[ids] = 0.0
        self.prev_action_n[ids] = 0.0
        self.episode_length_buf[ids] = 0
        pose, *_ = self._pose_speed()
        self.prev_dist[ids] = self.batch.target_distance(pose)[ids]
        self.ep_min_person_gap[ids] = 1e3
        self.ep_fp_steps[ids] = 0.0

    def _critic_state(self, pose, vb):
        b = self.batch
        center = torch.where(b.target_is_circle[:, None], b.target_circle[:, :2], b.target_rect[:, :2])
        front = front_point(pose, self.ATRIUM["robot"]["front_offset"])
        d = center - front
        c, s = torch.cos(pose[:, 2]), torch.sin(pose[:, 2])
        rel = torch.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], -1)
        wz = torch.tensor(self.state[:, self.o_v + 5], dtype=torch.float32, device=self.adev)[:, None]
        vxy = torch.tensor(vb[:, :2], dtype=torch.float32, device=self.adev)
        return torch.cat([rel, b.target_distance(pose)[:, None], vxy, wz], -1)

    def _observe(self):
        pose, _, _, _, vb = self._pose_speed()
        t = self.episode_length_buf.float() * self.step_dt
        frame = self.feed.observe(t, self.batch.perception_state(pose))
        self.prev = torch.where(self.first[:, None, None, None], frame, self.cur)
        self.cur = frame
        self.first[:] = False
        pol = torch.cat([torch.cat([self.cur, self.prev], 1).flatten(1), self.action_n], -1)
        crit = self._critic_state(pose, vb)
        return TensorDict({"policy": pol, "critic": crit}, batch_size=[self.B], device=self.device)

    def get_observations(self):
        return self._obs

    def image(self):
        """Current delivered frame (B, 4, S, S)."""
        return self.cur

    # ------------------------------------------------------------------------------------------- step
    def step(self, actions):
        a = actions.detach().to(self.adev, torch.float32).clamp(-1.0, 1.0)
        self.prev_action_n = self.action_n.clone()
        self.action_n = a.clone()
        cmd = scale_action(a, self.NAV["action"]).cpu().numpy()
        for k in range(self.k_walk):
            idx = self.episode_length_buf * self.k_walk + k
            self.batch.move_people(idx)
            pose, *_ = self._pose_speed()
            self.feed.observe(idx.float() * self.walk_dt, self.batch.perception_state(pose))
            with torch.no_grad():
                act = self.walk(self._walk_obs(cmd)).numpy().astype(np.float32)
            self.walk_last = act
            self._physics(DEFAULT_POS + self.ascale * act)
        self.episode_length_buf += 1
        pose, speed, tilt, height, vb = self._pose_speed()
        m = self.batch.metrics(pose, speed)
        fl = self.NAV["termination"]
        m["fall"] = torch.tensor((tilt > fl["fall_tilt_rad"]) | (height < self.MJ["fall"]["base_height_m"]), device=self.adev)
        m["collision"] = m["collision_footprint"]
        m["success"] = m["at_target"] & ~m["collision"]
        terms = reward_terms(self.prev_dist, m["target_distance"], m["success"], m["collision"], m["person_gap"],
                             m["fall"], m["out_of_square"], self.action_n, self.prev_action_n, self.NAV["reward"],
                             self.ATRIUM["collision"]["person_near_distance"])
        self.prev_dist = m["target_distance"].clone()
        self.ep_min_person_gap = torch.minimum(self.ep_min_person_gap, m["person_gap"])
        self.ep_fp_steps += m["collision_footprint"].float()
        reward = sum(terms[k] for k in TERMS)
        m["time_out"] = self.episode_length_buf >= self.max_episode_length
        terminal = m["success"] | m["collision"] | m["fall"] | m["out_of_square"]
        dones = terminal | m["time_out"]
        # per-episode info for the evaluation, before the reset clears it
        self.last_metrics = {k: v.clone() for k, v in m.items()}
        self.last_metrics["episode_steps"] = self.episode_length_buf.clone()
        self.last_metrics["min_person_gap"] = self.ep_min_person_gap.clone()
        self.last_metrics["scenario_id"] = self.batch.scenario_id.clone()
        self.last_terms = terms
        log = {f"/reward/{k}": v.mean() for k, v in terms.items()}
        if dones.any():
            for k in ("success", "collision", "fall", "out_of_square", "time_out"):
                log[f"/episode/{k}"] = m[k][dones].float().mean()
        extras = {"time_outs": m["time_out"] & ~terminal, "log": log}
        self.reset_ids(torch.nonzero(dones).flatten())
        self._obs = self._observe()
        return self._obs, reward, dones, extras

    def close(self):
        self.roll.close()

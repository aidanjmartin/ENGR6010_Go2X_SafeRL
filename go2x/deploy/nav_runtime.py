"""The navigation policy outside Isaac Lab: the exported actor, the scene it navigates, and odometry mapping.

- `NavPolicy`: loads scripts/export_nav_actor.py output and turns a drawn frame into a velocity target, with the
  same input layout as training (current and previous frame stacked, then the previous normalized action) and the
  same action scaling (go2x.atrium.rewards.scale_action).
- `SquareScene`: one scenario (config/nav_eval_scenarios.json format) in the square frame: obstacle footprints,
  people (moved along their scripts by time), target, the drawn image in the policy's view, and the checks the
  Isaac environment ends episodes on (success, footprint collision, leaving the square).
- `OdomToSquare`: the robot's odometry frame is wherever the robot booted. The robot is placed at the scenario's
  start pose; the first odometry reading is mapped onto that pose and later readings follow by the same rigid
  transform.
"""
import math

import numpy as np
import torch

from go2x.atrium.rewards import scale_action
from go2x.atrium.state import AtriumBatch
from go2x.atrium.topdown import draw, view_config
from go2x.config import load_config
from go2x.nav_net import ImageMLP


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class NavPolicy:
    def __init__(self, path):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        self.meta = m = blob["meta"]
        self.net = ImageMLP(m["image_shape"], m["extra_dim"], 3, m["cnn_channels"], m["cnn_out"], m["mlp"])
        self.net.load_state_dict(blob["state_dict"])
        self.net.eval()
        self.view = m["image_view"]
        self.draw_cfg = view_config(m["topdown"], self.view)
        self.limits = m["action"]
        self.reset()

    def reset(self):
        self.prev_frame = None
        self.prev_action = torch.zeros(3)

    @torch.no_grad()
    def act(self, frame):
        """frame (4, S, S) drawn in the policy's view -> (normalized action (3,), velocity target vx, vy, wz)."""
        prev = frame if self.prev_frame is None else self.prev_frame
        x = torch.cat([torch.cat([frame, prev], 0).flatten(), self.prev_action])[None]
        a = self.net(x)[0].clamp(-1.0, 1.0)
        self.prev_frame, self.prev_action = frame, a.clone()
        return a.numpy(), scale_action(a[None], self.limits)[0].numpy()


class SquareScene:
    def __init__(self, scenario, atrium_cfg=None, nav_cfg=None):
        self.cfg = atrium_cfg or load_config("atrium")
        self.scenario = scenario
        self.dt = 0.02          # people scripts are sampled at the walking-policy step, as in training
        self.batch = AtriumBatch(1, "cpu", seed=0, atrium_cfg=self.cfg, nav_cfg=nav_cfg, dt=self.dt,
                                 scenarios=[scenario])
        self.start_pose = self.batch.reset([0])[0].numpy().astype(float)
        self.footprint = tuple(self.cfg["robot"]["footprint"])
        self.walking_people = any(p.get("walking") for p in scenario.get("people", []))

    def set_time(self, t):
        """People to their scripted positions at t seconds after the policy started."""
        self.batch.move_people(torch.tensor([int(round(t / self.dt))]))

    def draw(self, pose, draw_cfg):
        p = torch.tensor(np.asarray([pose], np.float32))
        st = self.batch.perception_state(p)
        return draw(p, self.footprint, st["rects"], st["rect_mask"], st["circles"], st["circle_mask"],
                    st["target_rect"], st["target_circle"], st["target_is_circle"], draw_cfg)[0]

    def checks(self, pose, planar_speed):
        m = self.batch.metrics(torch.tensor(np.asarray([pose], np.float32)), torch.tensor([float(planar_speed)]))
        return {k: (bool(v[0]) if v.dtype == torch.bool else float(v[0])) for k, v in m.items()}

    def people_xy(self):
        return self.batch.circles[0, :, :2][self.batch.circle_mask[0]].numpy().tolist()

    def part_poses(self):
        """(n_parts, 7) poses of every simulator part (go2x.atrium.state.object_parts order)."""
        return self.batch.part_poses()[0].numpy()


class OdomToSquare:
    """Maps odometry (x, y, yaw) to the square frame, anchored at the first reading = the start pose."""

    def __init__(self, start_pose_square, first_odom):
        self.s = np.asarray(start_pose_square, float)
        self.o = np.asarray(first_odom, float)

    def __call__(self, odom):
        dx, dy = odom[0] - self.o[0], odom[1] - self.o[1]
        c, s = math.cos(-self.o[2]), math.sin(-self.o[2])
        lx, ly = c * dx - s * dy, s * dx + c * dy          # displacement in the start's body frame
        c2, s2 = math.cos(self.s[2]), math.sin(self.s[2])
        return np.array([self.s[0] + c2 * lx - s2 * ly, self.s[1] + s2 * lx + c2 * ly,
                         wrap(self.s[2] + odom[2] - self.o[2])])

    def to_dict(self):
        return {"start_pose_square": self.s.tolist(), "first_odom": self.o.tolist()}


def odom_pose(state):
    """(x, y, yaw) from a sport-mode state: odometry position and IMU yaw."""
    return np.array([state.position[0], state.position[1], state.rpy[2]], float)


def load_scenario(scenario_id=None, layout_path=None):
    import json

    from go2x.config import CONFIG_DIR
    if layout_path:
        with open(layout_path) as f:
            return json.load(f), f"layout {layout_path}"
    with open(CONFIG_DIR / "nav_eval_scenarios.json") as f:
        return json.load(f)["scenarios"][scenario_id], f"evaluation scenario {scenario_id}"

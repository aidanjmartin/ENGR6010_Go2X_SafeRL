"""Batched atrium state for the navigation environment (torch, no Isaac imports, so it is tested on the CPU).

Obstacle slots per environment: `chairs_per_color` chairs for each chair color, then one table (rectangles),
then the people (circles). A layout from the sampler fills some slots; the rest are parked behind the camera,
outside the square. Each slot is drawn in the simulator as one or more kinematic parts (`object_parts`).

All positions here are in the square frame (origin at the square center = the environment origin).
"""
import math

import numpy as np
import torch

from go2x.atrium import batch as Bt
from go2x.atrium.scene import layout_from_dict, sample_layout
from go2x.config import load_config


def object_parts(cfg):
    """Simulator parts: list of dicts (name, slot, shape, size, offset (x, y, z) in the obstacle frame, color)."""
    ob = cfg["obstacles"]
    parts, slot = [], 0
    ch = ob["chair"]
    w, d = ch["footprint"]
    t = ch["back_thickness"]
    for color, rgb in ch["colors"].items():
        for k in range(cfg["slots"]["chairs_per_color"]):
            name = f"chair_{color}_{k}"
            parts.append(dict(name=f"{name}_seat", slot=slot, shape="box", size=(w, d, ch["seat_height"]),
                              offset=(0.0, 0.0, ch["seat_height"] / 2), color=tuple(rgb), obstacle=name))
            back_h = ch["back_height"] - ch["seat_height"]
            parts.append(dict(name=f"{name}_back", slot=slot, shape="box", size=(t, d, back_h),
                              offset=(-(w / 2 - t / 2), 0.0, ch["seat_height"] + back_h / 2), color=tuple(rgb),
                              obstacle=name))
            slot += 1
    tb = ob["table"]
    tw, td = tb["footprint"]
    th, ls = tb["top_thickness"], tb["leg_size"]
    wood = tuple(cfg["slots"]["table_color"])
    parts.append(dict(name="table_top", slot=slot, shape="box", size=(tw, td, th), offset=(0.0, 0.0, tb["height"] - th / 2),
                      color=wood, obstacle="table"))
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        parts.append(dict(name=f"table_leg{i}", slot=slot, shape="box", size=(ls, ls, tb["height"] - th),
                          offset=(sx * (tw / 2 - ls / 2), sy * (td / 2 - ls / 2), (tb["height"] - th) / 2), color=wood,
                          obstacle="table"))
    slot += 1
    pe = ob["person"]
    for k in range(ob["person"]["count"][1]):
        parts.append(dict(name=f"person_{k}", slot=slot, shape="cylinder", size=(pe["radius"], pe["height"]),
                          offset=(0.0, 0.0, pe["height"] / 2), color=tuple(cfg["slots"]["person_color"]),
                          obstacle=f"person_{k}"))
        slot += 1
    return parts


class AtriumBatch:
    def __init__(self, num_envs, device="cpu", seed=0, atrium_cfg=None, nav_cfg=None, dt=0.02, scenarios=None):
        self.cfg = atrium_cfg or load_config("atrium")
        self.nav = nav_cfg or load_config("nav")
        self.B, self.device, self.dt = num_envs, device, dt
        ob = self.cfg["obstacles"]
        self.colors = list(ob["chair"]["colors"])
        self.cpc = self.cfg["slots"]["chairs_per_color"]
        self.n_chair_slots = len(self.colors) * self.cpc
        self.table_slot = self.n_chair_slots
        self.n_rects = self.n_chair_slots + 1
        self.n_people = ob["person"]["count"][1]
        self.n_slots = self.n_rects + self.n_people
        self.half = self.cfg["square"]["side"] / 2
        self.T = int(math.ceil(self.nav["timing"]["episode_s"] / dt)) + 2
        self.rng = np.random.default_rng(seed)
        self.scenarios = list(scenarios) if scenarios is not None else None
        self.next_scenario = 0
        z = lambda *s, **k: torch.zeros(*s, device=device, **k)  # noqa: E731
        self.rects, self.rect_mask = z(num_envs, self.n_rects, 5), z(num_envs, self.n_rects, dtype=torch.bool)
        self.circles = z(num_envs, self.n_people, 3)
        self.circle_mask = z(num_envs, self.n_people, dtype=torch.bool)
        self.traj = z(num_envs, self.n_people, self.T, 2)
        self.target_rect, self.target_circle = z(num_envs, 5), z(num_envs, 3)
        self.target_is_circle = z(num_envs, dtype=torch.bool)
        self.target_slot = z(num_envs, dtype=torch.long)
        self.scenario_id = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self.start_pose = z(num_envs, 3)
        self.layouts = [None] * num_envs
        self.parts = object_parts(self.cfg)
        pk = self.cfg["slots"]["park"]
        self.park_xy = torch.tensor([[pk["x0"] + pk["spacing"] * (s // pk["columns"]),
                                      pk["y0"] + pk["spacing"] * (s % pk["columns"])] for s in range(self.n_slots)],
                                    device=device, dtype=torch.float32)
        self.slot_pose = z(num_envs, self.n_slots, 3)    # x, y, yaw of each slot's obstacle frame
        self.slot_pose[:, :, :2] = self.park_xy

    # ---------------------------------------------------------------------------------------------- reset
    def reset(self, env_ids):
        """Draw (or take from the scenario list) a layout for each env id. Returns the robot start poses."""
        rr, w, d = self.cfg["obstacles"]["person"]["radius"], *self.cfg["obstacles"]["chair"]["footprint"]
        tw, td = self.cfg["obstacles"]["table"]["footprint"]
        for e in env_ids.tolist() if torch.is_tensor(env_ids) else env_ids:
            if self.scenarios is not None and self.next_scenario < len(self.scenarios):
                sid = self.next_scenario
                L = layout_from_dict(self.scenarios[sid], self.cfg)
                self.next_scenario += 1
            else:
                sid = -1
                L, _ = sample_layout(self.rng, self.cfg)
            self.layouts[e] = L
            self.scenario_id[e] = sid
            self.rect_mask[e] = False
            self.circle_mask[e] = False
            self.slot_pose[e, :, :2] = self.park_xy
            self.slot_pose[e, :, 2] = 0.0
            self.rects[e] = 0.0
            used = {c: 0 for c in range(len(self.colors))}
            chair_slot = []
            for x, y, yaw, c in L.chairs:
                s = int(c) * self.cpc + used[int(c)]
                used[int(c)] += 1
                chair_slot.append(s)
                self.rects[e, s] = torch.tensor([x, y, yaw, w / 2, d / 2])
                self.rect_mask[e, s] = True
                self.slot_pose[e, s] = torch.tensor([x, y, yaw])
            for x, y, yaw in L.table:
                self.rects[e, self.table_slot] = torch.tensor([x, y, yaw, tw / 2, td / 2])
                self.rect_mask[e, self.table_slot] = True
                self.slot_pose[e, self.table_slot] = torch.tensor([x, y, yaw])
            self.circles[e] = 0.0
            for k, p in enumerate(L.people):
                tr = torch.as_tensor(p.trajectory(self.dt, self.T), dtype=torch.float32)
                self.traj[e, k] = tr
                self.circles[e, k] = torch.tensor([tr[0, 0], tr[0, 1], rr])
                self.circle_mask[e, k] = True
                self.slot_pose[e, self.n_rects + k, :2] = tr[0]
            for k in range(len(L.people), self.n_people):
                self.traj[e, k] = self.park_xy[self.n_rects + k]
            kind, idx = L.target
            if kind == "person":
                self.target_is_circle[e] = True
                self.target_slot[e] = self.n_rects + idx
                self.target_circle[e] = self.circles[e, idx]
            else:
                self.target_is_circle[e] = False
                s = chair_slot[idx] if kind == "chair" else self.table_slot
                self.target_slot[e] = s
                self.target_rect[e] = self.rects[e, s]
            self.start_pose[e] = torch.as_tensor(L.robot_start, dtype=torch.float32)
        return self.start_pose[env_ids]

    # ---------------------------------------------------------------------------------------------- per step
    def move_people(self, step_idx):
        """Set people to their trajectory point at walking-policy step index (B,). Returns positions (B, P, 2)."""
        i = step_idx.clamp(0, self.T - 1).long()
        pos = self.traj[torch.arange(self.B, device=self.device)[:, None], torch.arange(self.n_people, device=self.device)[None], i[:, None]]
        self.circles[..., :2] = torch.where(self.circle_mask[..., None], pos, self.circles[..., :2])
        self.slot_pose[:, self.n_rects:, :2] = torch.where(self.circle_mask[..., None], pos, self.park_xy[self.n_rects:])
        moving = self.target_is_circle
        if moving.any():
            k = (self.target_slot - self.n_rects).clamp(min=0)
            self.target_circle[moving] = self.circles[torch.arange(self.B, device=self.device), k][moving]
        return pos

    def perception_state(self, robot_pose):
        return {"robot_pose": robot_pose, "rects": self.rects, "rect_mask": self.rect_mask, "circles": self.circles,
                "circle_mask": self.circle_mask, "target_rect": self.target_rect, "target_circle": self.target_circle,
                "target_is_circle": self.target_is_circle}

    def target_distance(self, robot_pose):
        """Front of the robot to the target footprint (B,)."""
        f = self.cfg["robot"]["front_offset"]
        front = robot_pose[:, :2] + f * torch.stack([torch.cos(robot_pose[:, 2]), torch.sin(robot_pose[:, 2])], -1)
        d_rect = Bt.point_rect_distance(front, self.target_rect)
        d_circ = (torch.linalg.norm(front - self.target_circle[:, :2], dim=-1) - self.target_circle[:, 2]).clamp(min=0)
        return torch.where(self.target_is_circle, d_circ, d_rect)

    def robot_rect(self, robot_pose):
        L, W = self.cfg["robot"]["footprint"]
        x, y, yaw = robot_pose.unbind(-1)
        return torch.stack([x, y, yaw, torch.full_like(x, L / 2), torch.full_like(x, W / 2)], -1)

    def metrics(self, robot_pose, planar_speed):
        rr = self.robot_rect(robot_pose)
        hit, _, _ = Bt.robot_obstacle_overlap(rr, self.rects, self.rect_mask, self.circles, self.circle_mask)
        gap = Bt.robot_circle_gap(rr, self.circles, self.circle_mask)
        dist = self.target_distance(robot_pose)
        sc = self.nav["success"]
        out = (robot_pose[:, :2].abs() > self.half).any(-1)
        return {"collision_footprint": hit, "person_gap": gap, "target_distance": dist,
                "at_target": (dist <= sc["distance"]) & (planar_speed < sc["speed"]), "out_of_square": out}

    # ---------------------------------------------------------------------------------------------- sim poses
    def part_poses(self):
        """World-frame offsets of every simulator part relative to the env origin: (B, n_parts, 7) as
        (x, y, z, qw, qx, qy, qz), yaw-only rotations."""
        out = torch.zeros(self.B, len(self.parts), 7, device=self.device)
        for j, p in enumerate(self.parts):
            sp = self.slot_pose[:, p["slot"]]
            c, s = torch.cos(sp[:, 2]), torch.sin(sp[:, 2])
            ox, oy, oz = p["offset"]
            out[:, j, 0] = sp[:, 0] + c * ox - s * oy
            out[:, j, 1] = sp[:, 1] + s * ox + c * oy
            out[:, j, 2] = oz
            out[:, j, 3] = torch.cos(sp[:, 2] / 2)
            out[:, j, 6] = torch.sin(sp[:, 2] / 2)
        return out

    def person_part_ids(self):
        return [j for j, p in enumerate(self.parts) if p["shape"] == "cylinder"]

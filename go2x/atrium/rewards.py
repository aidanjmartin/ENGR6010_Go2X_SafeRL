"""Navigation reward terms (torch, batched). Weights from config/nav.yaml `reward`.

Each term is returned separately so the environment can log it; the reward is their sum.
- progress: weight x (previous - current) front-to-target distance
- success: weight on the step the success condition first holds
- collision: weight on a collision step (footprint overlap or contact force)
- person_near: weight on each step the footprint is within `person_near_distance` of a person
- fall: weight on a fall step
- time: weight every step
- smoothness: weight x squared change of the normalized action
"""
import torch

TERMS = ("progress", "success", "collision", "person_near", "fall", "time", "smoothness")


def reward_terms(prev_dist, dist, success, collision, person_gap, fall, action, prev_action, w, person_near_distance):
    f = lambda b: b.float()  # noqa: E731
    return {
        "progress": w["progress"] * (prev_dist - dist),
        "success": w["success"] * f(success),
        "collision": w["collision"] * f(collision),
        "person_near": w["person_near"] * f(person_gap < person_near_distance),
        "fall": w["fall"] * f(fall),
        "time": torch.full_like(dist, w["time"]),
        "smoothness": w["smoothness"] * ((action - prev_action) ** 2).sum(-1),
    }


def total(terms):
    return sum(terms[k] for k in TERMS)


def normalize_action(cmd, limits):
    """Velocity target -> [-1, 1] per axis, using the configured ranges."""
    lo = torch.tensor([limits[k][0] for k in ("vx", "vy", "wz")], device=cmd.device)
    hi = torch.tensor([limits[k][1] for k in ("vx", "vy", "wz")], device=cmd.device)
    return 2 * (cmd - lo) / (hi - lo) - 1


def scale_action(a, limits):
    """[-1, 1] policy output -> velocity target within the ranges (clipped)."""
    lo = torch.tensor([limits[k][0] for k in ("vx", "vy", "wz")], device=a.device)
    hi = torch.tensor([limits[k][1] for k in ("vx", "vy", "wz")], device=a.device)
    return lo + (a.clamp(-1, 1) + 1) * 0.5 * (hi - lo)


def front_point(pose, front_offset):
    return pose[:, :2] + front_offset * torch.stack([torch.cos(pose[:, 2]), torch.sin(pose[:, 2])], -1)

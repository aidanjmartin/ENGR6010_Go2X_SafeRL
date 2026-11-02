"""The drawn top-down image: navigation input drawn from simulator state, plus camera realism.

Image: `size` x `size` pixels over the square plus margin (config/topdown.yaml). Pixel (row, col) has its
center at x = -E + (col + 0.5) * res, y = E - (row + 0.5) * res in the square frame (origin at the square
center, x right, y up in the image), where E is `half_extent`. Channels:
  0 static: chairs and the table (building boxes never fall inside the drawn area; see LOG.md)
  1 people
  2 robot: footprint at `robot_value`, front `heading_fraction` of it at `heading_value`
  3 target: the target object's footprint
"""
import math

import torch

from go2x.atrium import batch as Bt
from go2x.config import load_config


def pixel_centers(cfg, device="cpu"):
    S, E = cfg["size"], cfg["half_extent"]
    res = 2 * E / S
    c = -E + (torch.arange(S, device=device, dtype=torch.float32) + 0.5) * res
    xs = c[None, :].expand(S, S)
    ys = c.flip(0)[:, None].expand(S, S)
    return torch.stack([xs, ys], -1)  # (S, S, 2)


def world_to_pixel(xy, cfg):
    """Square-frame (x, y) -> continuous (row, col)."""
    S, E = cfg["size"], cfg["half_extent"]
    res = 2 * E / S
    return (E - xy[..., 1]) / res, (xy[..., 0] + E) / res


def _rect_mask(pix, rects, mask):
    """pix (S, S, 2); rects (B, N, 5); mask (B, N) -> (B, S, S) float."""
    if rects.shape[1] == 0:
        return torch.zeros(rects.shape[0], *pix.shape[:2], device=pix.device)
    d = Bt.point_rect_distance(pix[None, None], rects[:, :, None, None, :])
    return ((d <= 0) & mask[:, :, None, None]).any(1).float()


def _circle_mask(pix, circles, mask):
    if circles.shape[1] == 0:
        return torch.zeros(circles.shape[0], *pix.shape[:2], device=pix.device)
    d = torch.hypot(pix[None, None, ..., 0] - circles[:, :, None, None, 0], pix[None, None, ..., 1] - circles[:, :, None, None, 1])
    return ((d <= circles[:, :, None, None, 2]) & mask[:, :, None, None]).any(1).float()


def draw(robot_pose, footprint, rects, rect_mask, circles, circle_mask, target_rect, target_circle, target_is_circle,
         cfg=None, pix=None):
    """Draw (B, 4, S, S).

    robot_pose (B, 3); footprint (length, width); rects (B, N, 5) static obstacles; circles (B, M, 3) people;
    target_rect (B, 5) and target_circle (B, 3), one of which is used per env according to target_is_circle (B,).
    """
    cfg = cfg or load_config("topdown")
    pix = pixel_centers(cfg, robot_pose.device) if pix is None else pix
    B = robot_pose.shape[0]
    L, W = footprint
    static = _rect_mask(pix, rects, rect_mask)
    people = _circle_mask(pix, circles, circle_mask)
    x, y, yaw = robot_pose.unbind(-1)
    half_l = torch.full_like(x, L / 2)
    half_w = torch.full_like(x, W / 2)
    body = torch.stack([x, y, yaw, half_l, half_w], -1)[:, None]
    f = cfg["heading_fraction"]
    off = L / 2 - L * f / 2
    head = torch.stack([x + off * torch.cos(yaw), y + off * torch.sin(yaw), yaw, half_l * f, half_w], -1)[:, None]
    ones = torch.ones(B, 1, dtype=torch.bool, device=robot_pose.device)
    robot = torch.maximum(_rect_mask(pix, body, ones) * cfg["robot_value"], _rect_mask(pix, head, ones) * cfg["heading_value"])
    t_rect = _rect_mask(pix, target_rect[:, None], (~target_is_circle)[:, None])
    t_circ = _circle_mask(pix, target_circle[:, None], target_is_circle[:, None])
    target = torch.maximum(t_rect, t_circ)
    return torch.stack([static, people, robot, target], 1)


class CameraFeed:
    """Camera realism for the drawn image: frames captured at `camera_rate_hz` with pose noise, obstacle
    position noise and missed detections, delivered after a per-episode delay.

    Call `reset(env_ids, t0)` at episode start and `observe(t, state)` every control step with each env's
    episode time t (B,) and the true state. `observe` returns the image the policy sees at time t."""

    def __init__(self, num_envs, cfg=None, footprint=(0.70, 0.31), device="cpu", generator=None):
        self.cfg = cfg or load_config("topdown")
        r = self.cfg["realism"]
        self.rate = r["camera_rate_hz"]
        self.K = int(math.ceil(max(r["delay_s"]) * self.rate)) + 2
        S = self.cfg["size"]
        self.B, self.device, self.footprint = num_envs, device, footprint
        self.g = generator
        self.pix = pixel_centers(self.cfg, device)
        self.frames = torch.zeros(num_envs, self.K, 4, S, S, device=device)
        self.times = torch.full((num_envs, self.K), -1e9, device=device)
        self.last_idx = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self.delay = torch.zeros(num_envs, device=device)

    def _rand(self, *shape):
        return torch.rand(*shape, device=self.device, generator=self.g)

    def _randn(self, *shape):
        return torch.randn(*shape, device=self.device, generator=self.g)

    def reset(self, env_ids):
        lo, hi = self.cfg["realism"]["delay_s"]
        self.delay[env_ids] = lo + (hi - lo) * self._rand(len(env_ids))
        self.times[env_ids] = -1e9
        self.last_idx[env_ids] = -1
        self.frames[env_ids] = 0

    def capture(self, state):
        """A noisy frame of the given state (dict of tensors for all envs)."""
        r = self.cfg["realism"]
        pose = state["robot_pose"].clone()
        pose[:, :2] += r["robot_pos_noise_m"] * self._randn(self.B, 2)
        pose[:, 2] += r["robot_yaw_noise_rad"] * self._randn(self.B)
        rects, circles = state["rects"].clone(), state["circles"].clone()
        rects[..., :2] += r["obstacle_pos_noise_m"] * self._randn(*rects.shape[:2], 2)
        circles[..., :2] += r["obstacle_pos_noise_m"] * self._randn(*circles.shape[:2], 2)
        rmask = state["rect_mask"] & (self._rand(*rects.shape[:2]) >= r["miss_probability"])
        cmask = state["circle_mask"] & (self._rand(*circles.shape[:2]) >= r["miss_probability"])
        t_rect, t_circ = state["target_rect"].clone(), state["target_circle"].clone()
        shift = r["obstacle_pos_noise_m"] * self._randn(self.B, 2)
        t_rect[:, :2] += shift
        t_circ[:, :2] += shift
        return draw(pose, self.footprint, rects, rmask, circles, cmask, t_rect, t_circ, state["target_is_circle"],
                    self.cfg, self.pix)

    def observe(self, t, state):
        idx = torch.floor(t * self.rate + 1e-6).long()
        new = idx > self.last_idx
        if new.any():
            frame = self.capture(state)
            slot = idx % self.K
            e = torch.nonzero(new).squeeze(-1)
            self.frames[e, slot[e]] = frame[e]
            self.times[e, slot[e]] = idx[e].float() / self.rate
            self.last_idx[new] = idx[new]
        valid = self.times > -1e8
        ready = valid & (self.times <= (t - self.delay)[:, None])
        # latest ready frame; before the first one is ready, the oldest captured frame
        score = torch.where(ready, self.times, torch.full_like(self.times, -1e10))
        pick = torch.where(ready.any(1), score.argmax(1),
                           torch.where(valid, self.times, torch.full_like(self.times, 1e10)).argmin(1))
        return self.frames[torch.arange(self.B, device=self.device), pick]

    def delivered_capture_time(self, t):
        """Capture time of the frame `observe(t, ...)` returned (for tests and logs)."""
        ready = (self.times > -1e8) & (self.times <= (t - self.delay)[:, None])
        score = torch.where(ready, self.times, torch.full_like(self.times, -1e10))
        return score.max(1).values

"""Batched torch footprint tests for the per-step checks in the Isaac Lab environment.

Shapes: B environments, N rectangles, M circles. Rectangles (..., 5) = (cx, cy, yaw, hx, hy); circles
(..., 3) = (cx, cy, r). Masks mark which slots hold a real obstacle (parked slots are False).
"""
import torch


def rect_corners(rect):
    """(..., 5) -> (..., 4, 2)."""
    cx, cy, yaw, hx, hy = rect.unbind(-1)
    c, s = torch.cos(yaw), torch.sin(yaw)
    sx = torch.stack([-hx, hx, hx, -hx], -1)
    sy = torch.stack([-hy, -hy, hy, hy], -1)
    x = cx[..., None] + c[..., None] * sx - s[..., None] * sy
    y = cy[..., None] + s[..., None] * sx + c[..., None] * sy
    return torch.stack([x, y], -1)


def point_rect_distance(points, rect):
    """points (..., 2), rect (..., 5) broadcast -> distance (...), 0 inside."""
    cx, cy, yaw, hx, hy = rect.unbind(-1)
    px, py = points[..., 0] - cx, points[..., 1] - cy
    c, s = torch.cos(yaw), torch.sin(yaw)
    qx, qy = c * px + s * py, -s * px + c * py
    return torch.hypot((qx.abs() - hx).clamp(min=0), (qy.abs() - hy).clamp(min=0))


def rects_overlap(a, b):
    """Separating axis test. a (..., 5), b (..., 5) broadcast -> bool (...)."""
    A, B = rect_corners(a), rect_corners(b)
    A, B = torch.broadcast_tensors(A, B)
    sep = torch.zeros(A.shape[:-2], dtype=torch.bool, device=A.device)
    for poly in (A, B):
        for i in range(2):
            e = poly[..., i + 1, :] - poly[..., i, :]
            n = torch.stack([-e[..., 1], e[..., 0]], -1)
            pa = (A * n[..., None, :]).sum(-1)
            pb = (B * n[..., None, :]).sum(-1)
            sep |= (pa.max(-1).values < pb.min(-1).values) | (pb.max(-1).values < pa.min(-1).values)
    return ~sep


def rect_rect_distance(a, b):
    """Gap between two rectangles, 0 when they overlap. The smallest corner-to-rectangle distance, both ways,
    is exact: the closest points of two disjoint convex polygons include a vertex of one of them."""
    A, B = rect_corners(a), rect_corners(b)
    da = point_rect_distance(A, b[..., None, :]).min(-1).values
    db = point_rect_distance(B, a[..., None, :]).min(-1).values
    d = torch.minimum(da, db)
    return torch.where(rects_overlap(a, b), torch.zeros_like(d), d)


def robot_obstacle_overlap(robot, rects, rect_mask, circles, circle_mask):
    """Footprint collision. robot (B, 5); rects (B, N, 5); circles (B, M, 3). Returns (B,) bool and the
    per-obstacle hits."""
    hit_r = rects_overlap(robot[:, None, :], rects) & rect_mask
    d_c = point_rect_distance(circles[..., :2], robot[:, None, :])
    hit_c = (d_c < circles[..., 2]) & circle_mask
    return hit_r.any(-1) | hit_c.any(-1), hit_r, hit_c


def robot_circle_gap(robot, circles, circle_mask, far=1e3):
    """Distance from the robot footprint to the nearest circle edge (B,); `far` when there is none."""
    d = point_rect_distance(circles[..., :2], robot[:, None, :]) - circles[..., 2]
    return torch.where(circle_mask, d.clamp(min=0), torch.full_like(d, far)).min(-1).values

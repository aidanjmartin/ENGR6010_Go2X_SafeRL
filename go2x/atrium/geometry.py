"""Planar footprints in numpy: oriented rectangles (cx, cy, yaw, hx, hy) with half sizes, and circles (cx, cy, r).

Used by the scene sampler and the planner. go2x/atrium/batch.py has the same tests in batched torch for the
per-step checks; tests/test_atrium_geometry.py checks that the two agree.
"""
import numpy as np


def rect_corners(rect):
    cx, cy, yaw, hx, hy = rect
    c, s = np.cos(yaw), np.sin(yaw)
    local = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
    return local @ np.array([[c, s], [-s, c]]) + [cx, cy]


def point_rect_distance(points, rect):
    """Distance from each point (N, 2) to the rectangle (0 inside)."""
    cx, cy, yaw, hx, hy = rect
    p = np.atleast_2d(points) - [cx, cy]
    c, s = np.cos(yaw), np.sin(yaw)
    qx, qy = c * p[:, 0] + s * p[:, 1], -s * p[:, 0] + c * p[:, 1]
    return np.hypot(np.maximum(np.abs(qx) - hx, 0), np.maximum(np.abs(qy) - hy, 0))


def rects_overlap(a, b, gap=0.0):
    """Separating axis test for two rectangles, each grown by gap / 2 on every side."""
    A, B = rect_corners(grow(a, gap / 2)), rect_corners(grow(b, gap / 2))
    for poly in (A, B):
        for i in range(2):
            e = poly[i + 1] - poly[i]
            n = np.array([-e[1], e[0]])
            pa, pb = A @ n, B @ n
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def grow(rect, d):
    cx, cy, yaw, hx, hy = rect
    return (cx, cy, yaw, hx + d, hy + d)


def rect_circle_overlap(rect, circle, gap=0.0):
    cx, cy, r = circle
    return bool(point_rect_distance([[cx, cy]], rect)[0] < r + gap)


def rect_inside_square(rect, half, margin=0.0):
    return bool(np.all(np.abs(rect_corners(rect)) <= half - margin + 1e-12))


def robot_rect(pose, footprint):
    """Robot footprint centered on the base origin, aligned with its heading."""
    x, y, yaw = pose
    return (x, y, yaw, footprint[0] / 2, footprint[1] / 2)

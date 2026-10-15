"""Rectangular room with box and cylinder obstacles: sampling, collision and planar LiDAR.

The room spans [0, width] x [0, depth]. Cylinders are (cx, cy, r). Boxes are oriented
rectangles (cx, cy, yaw, hx, hy) with half sizes hx, hy. The robot is a circle.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Scene:
    width: float
    depth: float
    cylinders: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 5)))
    start: np.ndarray = field(default_factory=lambda: np.zeros(3))  # x, y, yaw
    goal: np.ndarray = field(default_factory=lambda: np.zeros(2))

    def collides(self, points, radius):
        """True for each (x, y) point whose circle of the given radius touches anything."""
        p = np.atleast_2d(points)
        hit = ((p[:, 0] < radius) | (p[:, 0] > self.width - radius)
               | (p[:, 1] < radius) | (p[:, 1] > self.depth - radius))
        for cx, cy, r in self.cylinders:
            hit |= np.hypot(p[:, 0] - cx, p[:, 1] - cy) < r + radius
        for cx, cy, yaw, hx, hy in self.boxes:
            c, s = np.cos(yaw), np.sin(yaw)
            dx, dy = p[:, 0] - cx, p[:, 1] - cy
            qx, qy = c * dx + s * dy, -s * dx + c * dy
            ex = qx - np.clip(qx, -hx, hx)
            ey = qy - np.clip(qy, -hy, hy)
            hit |= np.hypot(ex, ey) < radius
        return hit

    def raycast(self, origin, angles, max_range):
        """Distance from origin along each world-frame angle to the first surface."""
        ox, oy = origin
        dx, dy = np.cos(angles), np.sin(angles)
        with np.errstate(divide="ignore", invalid="ignore"):
            tx = np.where(dx > 0, (self.width - ox) / dx, np.where(dx < 0, -ox / dx, np.inf))
            ty = np.where(dy > 0, (self.depth - oy) / dy, np.where(dy < 0, -oy / dy, np.inf))
            t = np.minimum(tx, ty)
            for cx, cy, r in self.cylinders:
                fx, fy = ox - cx, oy - cy
                b = fx * dx + fy * dy
                cc = fx * fx + fy * fy - r * r
                disc = b * b - cc
                th = np.where(disc >= 0, -b - np.sqrt(np.maximum(disc, 0)), np.inf)
                th = np.where(cc <= 0, 0.0, np.where(th >= 0, th, np.inf))
                t = np.minimum(t, th)
            for cx, cy, yaw, hx, hy in self.boxes:
                c, s = np.cos(yaw), np.sin(yaw)
                px, py = c * (ox - cx) + s * (oy - cy), -s * (ox - cx) + c * (oy - cy)
                ux, uy = c * dx + s * dy, -s * dx + c * dy
                # slab method; a tiny component stands in for zero so parallel rays need no special case
                ux = np.where(np.abs(ux) < 1e-12, 1e-12, ux)
                uy = np.where(np.abs(uy) < 1e-12, 1e-12, uy)
                t1x, t2x = (-hx - px) / ux, (hx - px) / ux
                t1y, t2y = (-hy - py) / uy, (hy - py) / uy
                tmin = np.maximum(np.minimum(t1x, t2x), np.minimum(t1y, t2y))
                tmax = np.minimum(np.maximum(t1x, t2x), np.maximum(t1y, t2y))
                hit = tmax >= np.maximum(tmin, 0)
                t = np.minimum(t, np.where(hit, np.maximum(tmin, 0), np.inf))
        return np.minimum(t, max_range)

    def reachable(self, radius, resolution):
        """Whether the goal is reachable from the start on a grid of free robot positions."""
        xs = np.arange(resolution / 2, self.width, resolution)
        ys = np.arange(resolution / 2, self.depth, resolution)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        free = ~self.collides(np.stack([gx.ravel(), gy.ravel()], 1), radius).reshape(gx.shape)

        def cell(p):
            return (min(int(p[0] / resolution), len(xs) - 1), min(int(p[1] / resolution), len(ys) - 1))

        si, gi = cell(self.start[:2]), cell(self.goal)
        if not (free[si] and free[gi]):
            return False
        seen = np.zeros_like(free)
        seen[si] = True
        while True:  # flood fill, 4-connected
            grow = seen.copy()
            grow[1:] |= seen[:-1]
            grow[:-1] |= seen[1:]
            grow[:, 1:] |= seen[:, :-1]
            grow[:, :-1] |= seen[:, 1:]
            grow &= free
            if grow[gi]:
                return True
            if (grow == seen).all():
                return False
            seen = grow


def sample_scene(rng, cfg, robot_radius):
    """Sample a room, start, goal and obstacles, resampling until the goal is reachable."""
    sc = cfg
    margin = robot_radius + sc["clearance"]
    while True:
        w, d = rng.uniform(*sc["room_size_range"], 2)
        lo, hi = np.array([margin, margin]), np.array([w - margin, d - margin])
        start_xy = rng.uniform(lo, hi)
        goal = rng.uniform(lo, hi)
        if np.hypot(*(goal - start_xy)) < sc["min_goal_distance"]:
            continue
        scene = Scene(w, d, start=np.array([*start_xy, rng.uniform(-np.pi, np.pi)]), goal=goal)
        cyl, box = [], []
        n = rng.integers(sc["n_obstacles_range"][0], sc["n_obstacles_range"][1] + 1)
        for _ in range(n * 20):  # attempts
            if len(cyl) + len(box) == n:
                break
            kind = sc["obstacle_types"][rng.integers(len(sc["obstacle_types"]))]
            cx, cy = rng.uniform(0, w), rng.uniform(0, d)
            if kind == "cylinder":
                trial = Scene(np.inf, np.inf, cylinders=np.array([[cx, cy, rng.uniform(*sc["cylinder_radius_range"])]]))
            else:
                hx, hy = rng.uniform(*sc["box_half_size_range"], 2)
                trial = Scene(np.inf, np.inf, boxes=np.array([[cx, cy, rng.uniform(-np.pi, np.pi), hx, hy]]))
            # keep start and goal clear (an infinite room tests the obstacle only, not the walls)
            if trial.collides(np.stack([start_xy, goal]), margin).any():
                continue
            (cyl if kind == "cylinder" else box).append((trial.cylinders if kind == "cylinder" else trial.boxes)[0])
        scene.cylinders = np.array(cyl).reshape(-1, 3)
        scene.boxes = np.array(box).reshape(-1, 5)
        if scene.reachable(robot_radius, sc["grid_resolution"]):
            return scene

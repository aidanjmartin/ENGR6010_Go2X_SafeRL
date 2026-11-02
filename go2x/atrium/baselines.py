"""Baseline controllers for Task 4. Both read only the drawn top-down image the policy sees (one frame,
4 x S x S) and output a velocity target (vx, vy, wz) for the same walking policy.

- StraightController: the Phase 1 go-to-goal idea. Turn toward the target and drive at it; no obstacle use.
- AStarController: A* on the image's obstacle channels (grown by `inflate_m`), then pure pursuit.

The robot pose comes from the image: the centroid of the robot channel is the position, and the direction
from it to the centroid of the heading marker is the heading.
"""
import heapq

import numpy as np
from scipy import ndimage

from go2x.config import load_config


class ImageGeometry:
    def __init__(self, td_cfg=None):
        self.cfg = td_cfg or load_config("topdown")
        self.S, self.E = self.cfg["size"], self.cfg["half_extent"]
        self.res = 2 * self.E / self.S

    def rc_to_xy(self, r, c):
        return np.stack([-self.E + (np.asarray(c) + 0.5) * self.res, self.E - (np.asarray(r) + 0.5) * self.res], -1)

    def xy_to_rc(self, xy):
        xy = np.asarray(xy, float)
        return (np.clip(((self.E - xy[..., 1]) / self.res).astype(int), 0, self.S - 1),
                np.clip(((xy[..., 0] + self.E) / self.res).astype(int), 0, self.S - 1))

    def pose(self, img):
        robot = img[2]
        rows, cols = np.nonzero(robot > 0)
        if len(rows) == 0:
            return None
        p = self.rc_to_xy(rows, cols).mean(0)
        hr, hc = np.nonzero(robot >= self.cfg["heading_value"] - 1e-6)
        if len(hr) == 0:
            return None
        h = self.rc_to_xy(hr, hc).mean(0)
        return np.array([p[0], p[1], np.arctan2(h[1] - p[1], h[0] - p[0])])

    def target_cells(self, img):
        r, c = np.nonzero(img[3] > 0)
        return self.rc_to_xy(r, c) if len(r) else None


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _clip(cmd, limits):
    return np.array([np.clip(cmd[i], *limits[k]) for i, k in enumerate(("vx", "vy", "wz"))])


class StraightController:
    def __init__(self, nav_cfg=None, atrium_cfg=None, td_cfg=None):
        self.nav = nav_cfg or load_config("nav")
        self.front = (atrium_cfg or load_config("atrium"))["robot"]["front_offset"]
        self.geo = ImageGeometry(td_cfg)
        self.p = self.nav["baselines"]["straight"]

    def reset(self):
        pass

    def act(self, img):
        pose, tgt = self.geo.pose(img), self.geo.target_cells(img)
        if pose is None or tgt is None:
            return np.zeros(3)
        front = pose[:2] + self.front * np.array([np.cos(pose[2]), np.sin(pose[2])])
        d = np.hypot(*(tgt - front).T).min() - self.geo.res / 2
        c = tgt.mean(0)
        alpha = _wrap(np.arctan2(c[1] - pose[1], c[0] - pose[0]) - pose[2])
        if d <= self.p["stop_distance"]:
            return np.zeros(3) if abs(alpha) < 0.3 else _clip([0.0, 0.0, self.p["k_yaw"] * alpha], self.nav["action"])
        v = self.p["k_v"] * d * max(np.cos(alpha), 0.0)
        return _clip([v, 0.0, self.p["k_yaw"] * alpha], self.nav["action"])


def astar(free, start, goal_mask):
    """8-connected A* on a boolean grid. Returns the list of cells from start to the first goal reached."""
    goals = np.argwhere(goal_mask)
    if len(goals) == 0 or not free[start]:
        return None

    def h(rc):
        d = np.abs(goals - rc)
        return float((np.maximum(d[:, 0], d[:, 1]) + (np.sqrt(2) - 1) * np.minimum(d[:, 0], d[:, 1])).min())

    S0, S1 = free.shape
    g = {start: 0.0}
    came = {}
    heap = [(h(np.array(start)), 0.0, start)]
    steps = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    while heap:
        _, gc, cur = heapq.heappop(heap)
        if goal_mask[cur]:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        if gc > g.get(cur, np.inf):
            continue
        for dr, dc in steps:
            nb = (cur[0] + dr, cur[1] + dc)
            if not (0 <= nb[0] < S0 and 0 <= nb[1] < S1) or not free[nb]:
                continue
            if dr and dc and not (free[cur[0] + dr, cur[1]] and free[cur[0], cur[1] + dc]):
                continue  # no corner cutting
            ng = gc + (np.sqrt(2) if dr and dc else 1.0)
            if ng < g.get(nb, np.inf):
                g[nb] = ng
                came[nb] = cur
                heapq.heappush(heap, (ng + h(np.array(nb)), ng, nb))
    return None


class AStarController:
    def __init__(self, nav_cfg=None, atrium_cfg=None, td_cfg=None):
        self.nav = nav_cfg or load_config("nav")
        self.front = (atrium_cfg or load_config("atrium"))["robot"]["front_offset"]
        self.geo = ImageGeometry(td_cfg)
        self.p = self.nav["baselines"]["astar"]
        self.reset()

    def reset(self):
        self.path, self.k = None, 0
        self.plans = self.failed_plans = 0

    def plan(self, img, pose):
        geo = self.geo
        obst = (img[0] > 0) | (img[1] > 0) | (img[3] > 0)
        dist_obst = ndimage.distance_transform_edt(~obst) * geo.res
        free = dist_obst > self.p["inflate_m"]
        start = tuple(int(v) for v in geo.xy_to_rc(pose[:2]))
        rr, cc = np.ogrid[:geo.S, :geo.S]
        free[(rr - start[0]) ** 2 + (cc - start[1]) ** 2 <= (self.p["inflate_m"] / geo.res) ** 2] |= ~obst[
            (rr - start[0]) ** 2 + (cc - start[1]) ** 2 <= (self.p["inflate_m"] / geo.res) ** 2]
        dist_t = ndimage.distance_transform_edt(img[3] == 0) * geo.res
        goal = free & (dist_t <= self.p["stop_distance"] + self.front)
        self.plans += 1
        path = astar(free, start, goal)
        if path is None:
            self.failed_plans += 1
            return None
        r, c = np.array(path).T
        return geo.rc_to_xy(r, c)

    def act(self, img):
        pose, tgt = self.geo.pose(img), self.geo.target_cells(img)
        if pose is None or tgt is None:
            return np.zeros(3)
        if self.path is None or self.k % self.p["replan_every"] == 0:
            new = self.plan(img, pose)
            if new is not None:
                self.path = new
        self.k += 1
        front = pose[:2] + self.front * np.array([np.cos(pose[2]), np.sin(pose[2])])
        d = np.hypot(*(tgt - front).T).min() - self.geo.res / 2
        c = tgt.mean(0)
        if d <= self.p["stop_distance"] or self.path is None:
            alpha = _wrap(np.arctan2(c[1] - pose[1], c[0] - pose[0]) - pose[2])
            if self.path is None and d > self.p["stop_distance"]:
                return np.zeros(3)  # no plan: wait
            return np.zeros(3) if abs(alpha) < 0.3 else _clip([0.0, 0.0, self.p["k_yaw"] * alpha], self.nav["action"])
        dists = np.hypot(*(self.path - pose[:2]).T)
        i0 = int(dists.argmin())
        ahead = np.nonzero(dists[i0:] >= self.p["lookahead_m"])[0]
        goal_pt = self.path[i0 + ahead[0]] if len(ahead) else self.path[-1]
        alpha = _wrap(np.arctan2(goal_pt[1] - pose[1], goal_pt[0] - pose[0]) - pose[2])
        v = self.p["v_max"] * max(np.cos(alpha), 0.0)
        return _clip([v, 0.0, self.p["k_yaw"] * alpha], self.nav["action"])

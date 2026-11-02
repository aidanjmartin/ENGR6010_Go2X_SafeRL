"""Random atrium layouts: chairs, a table, people, the robot start pose and the target object.

Frame: aeb_level1 world frame, square centered on config `square.center`. A layout is accepted only when
- every static footprint lies inside the square and keeps `gap` from the others;
- the robot start footprint keeps `start_clearance` from every obstacle (people at their start positions);
- on a free-space grid (obstacles grown by `inflate`), a cell within `goal_distance + front_offset` of the
  target footprint is reachable from the start cell. Walking people are left out of the grid.
"""
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from go2x.atrium import geometry as G
from go2x.config import load_config


@dataclass
class Person:
    start: np.ndarray            # (2,)
    walking: bool
    a: np.ndarray = None         # path ends (walking only)
    b: np.ndarray = None
    script: list = field(default_factory=list)   # [(duration_s, speed_mps)]

    def trajectory(self, dt, n_steps):
        """Positions (n_steps, 2) at t = 0, dt, ...: back and forth along a-b, following the speed script."""
        if not self.walking:
            return np.repeat(self.start[None], n_steps, 0)
        L = float(np.linalg.norm(self.b - self.a))
        s0 = float(np.dot(self.start - self.a, self.b - self.a) / L)
        ends = np.cumsum([d for d, _ in self.script])
        out, s = np.empty((n_steps, 2)), s0
        for i in range(n_steps):
            phase = s % (2 * L)
            arc = phase if phase <= L else 2 * L - phase
            out[i] = self.a + (self.b - self.a) * (arc / L)
            t = (i + 0.5) * dt
            seg = min(int(np.searchsorted(ends, t, side="right")), len(self.script) - 1)
            s += self.script[seg][1] * dt
        return out


@dataclass
class Layout:
    robot_start: np.ndarray                     # (x, y, yaw)
    chairs: np.ndarray                          # (Nc, 4): x, y, yaw, color index
    table: np.ndarray                           # (0 or 1, 3): x, y, yaw
    people: list                                # [Person]
    target: tuple                               # ("chair", i) | ("table", 0) | ("person", i)
    cfg: dict = field(repr=False, default=None)

    def chair_rects(self):
        w, d = self.cfg["obstacles"]["chair"]["footprint"]
        return [(x, y, yaw, w / 2, d / 2) for x, y, yaw, _ in self.chairs]

    def table_rects(self):
        w, d = self.cfg["obstacles"]["table"]["footprint"]
        return [(x, y, yaw, w / 2, d / 2) for x, y, yaw in self.table]

    def static_rects(self):
        return self.chair_rects() + self.table_rects()

    def person_circles(self, positions=None):
        r = self.cfg["obstacles"]["person"]["radius"]
        pos = positions if positions is not None else [p.start for p in self.people]
        return [(x, y, r) for x, y in pos]

    def color_names(self):
        names = list(self.cfg["obstacles"]["chair"]["colors"])
        return [names[int(c)] for c in self.chairs[:, 3]]

    def objects(self):
        """Every obstacle as a dict: id, kind, color (chairs), footprint (rect or circle)."""
        out = []
        for i, (rect, color) in enumerate(zip(self.chair_rects(), self.color_names())):
            out.append({"id": f"chair_{i}", "kind": "chair", "color": color, "rect": rect})
        for rect in self.table_rects():
            out.append({"id": "table_0", "kind": "table", "color": None, "rect": rect})
        for i, (p, circle) in enumerate(zip(self.people, self.person_circles())):
            out.append({"id": f"person_{i}", "kind": "person", "color": None, "circle": circle, "walking": p.walking})
        return out

    @property
    def target_id(self):
        return f"{self.target[0]}_{self.target[1]}"


def make_speed_script(rng, total_time, speed_range, pause_probability, segment_duration):
    """Random (duration_s, speed_mps) segments covering total_time. Adapted from tourbot_rl pedestrian.py
    (aidan/ureca-rl 2a10b64) with the atrium's walking speeds."""
    script, t = [], 0.0
    while t < total_time:
        d = float(rng.uniform(*segment_duration))
        v = 0.0 if rng.random() < pause_probability else float(rng.uniform(*speed_range))
        script.append((d, v))
        t += d
    return script


def _square_chord(rng, half, inset):
    """A random straight path across the square (ends on the edges of the square shrunk by inset)."""
    h = half - inset
    while True:
        p = rng.uniform(-h, h, 2)
        th = rng.uniform(0, np.pi)
        d = np.array([np.cos(th), np.sin(th)])
        ts = []
        for k in range(2):
            if abs(d[k]) > 1e-9:
                ts += [(-h - p[k]) / d[k], (h - p[k]) / d[k]]
        pts = [p + t * d for t in ts if np.all(np.abs(p + t * d) <= h + 1e-9)]
        if len(pts) >= 2:
            a, b = min(pts, key=lambda q: np.dot(q, d)), max(pts, key=lambda q: np.dot(q, d))
            if np.linalg.norm(b - a) > 1.0:
                return a, b


def _path_clear(a, b, rects, radius, gap, n=40):
    pts = a + (b - a) * np.linspace(0, 1, n)[:, None]
    return all(np.all(G.point_rect_distance(pts, r) >= radius + gap) for r in rects)


def reachable(layout, cfg=None):
    """Is a goal cell reachable from the start on the free-space grid? Returns (bool, details)."""
    cfg = cfg or layout.cfg
    half = cfg["square"]["side"] / 2
    rc = cfg["reachability"]
    res, inflate = rc["grid_resolution"], rc["inflate"]
    n = int(np.floor(2 * half / res))
    g = -half + res * (np.arange(n) + 0.5)
    gx, gy = np.meshgrid(g, g, indexing="ij")
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    occ = np.zeros(len(pts), bool)
    for r in layout.static_rects():
        occ |= G.point_rect_distance(pts, r) < inflate
    for p, (cx, cy, rad) in zip(layout.people, layout.person_circles()):
        if not p.walking:
            occ |= np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) < rad + inflate
    kind, idx = layout.target
    if kind == "person":
        cx, cy, rad = layout.person_circles()[idx]
        d_t = np.maximum(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - rad, 0)
    else:
        rect = (layout.chair_rects() if kind == "chair" else layout.table_rects())[idx]
        d_t = G.point_rect_distance(pts, rect)
    goal = (~occ) & (d_t <= rc["goal_distance"] + cfg["robot"]["front_offset"])
    occ, goal = occ.reshape(n, n), goal.reshape(n, n)
    sx, sy = layout.robot_start[:2]
    si, sj = int((sx + half) // res), int((sy + half) // res)
    if not (0 <= si < n and 0 <= sj < n) or occ[si, sj]:
        return False, {"reason": "start cell blocked"}
    seen = np.zeros_like(occ)
    seen[si, sj] = True
    q = deque([(si, sj)])
    while q:
        i, j = q.popleft()
        if goal[i, j]:
            return True, {"goal_cells": int(goal.sum())}
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                a, b = i + di, j + dj
                if (di or dj) and 0 <= a < n and 0 <= b < n and not seen[a, b] and not occ[a, b]:
                    seen[a, b] = True
                    q.append((a, b))
    return False, {"reason": "no path", "goal_cells": int(goal.sum())}


def _target_distance(layout):
    pose = layout.robot_start
    front = pose[:2] + layout.cfg["robot"]["front_offset"] * np.array([np.cos(pose[2]), np.sin(pose[2])])
    kind, idx = layout.target
    if kind == "person":
        cx, cy, r = layout.person_circles()[idx]
        return max(np.hypot(*(front - (cx, cy))) - r, 0.0)
    rect = (layout.chair_rects() if kind == "chair" else layout.table_rects())[idx]
    return float(G.point_rect_distance(front[None], rect)[0])


def sample_layout(rng, cfg=None, target_kind=None):
    """Draw layouts until one passes every check. Returns (layout, stats)."""
    cfg = cfg or load_config("atrium")
    ob = cfg["obstacles"]
    half = cfg["square"]["side"] / 2
    stats = {"rejected": {}}

    def reject(why):
        stats["rejected"][why] = stats["rejected"].get(why, 0) + 1

    for _ in range(ob["max_tries"]):
        n_chairs = int(rng.integers(ob["chair"]["count"][0], ob["chair"]["count"][1] + 1))
        has_table = rng.random() < ob["table"]["probability"]
        n_people = int(rng.integers(ob["person"]["count"][0], ob["person"]["count"][1] + 1))
        if n_chairs + has_table + n_people == 0:
            reject("no objects")
            continue
        rects, ok = [], True
        table = np.zeros((0, 3))
        chairs = np.zeros((0, 4))
        specs = ([("table", ob["table"]["footprint"])] if has_table else []) + [("chair", ob["chair"]["footprint"])] * n_chairs
        for kind, (w, d) in specs:
            for _try in range(50):
                yaw = rng.uniform(-np.pi, np.pi)
                xy = rng.uniform(-half, half, 2)
                r = (xy[0], xy[1], yaw, w / 2, d / 2)
                if G.rect_inside_square(r, half, ob["inside_margin"]) and not any(
                        G.rects_overlap(r, o, ob["gap"]) for o in rects):
                    break
            else:
                ok = False
                break
            rects.append(r)
            if kind == "table":
                table = np.array([[xy[0], xy[1], yaw]])
            else:
                chairs = np.vstack([chairs, [xy[0], xy[1], yaw, rng.integers(len(ob["chair"]["colors"]))]])
        if not ok:
            reject("static placement")
            continue
        people, circles = [], []
        pr = ob["person"]["radius"]
        for _p in range(n_people):
            for _try in range(50):
                if rng.random() < ob["person"]["walking_probability"]:
                    a, b = _square_chord(rng, half, pr)
                    if not _path_clear(a, b, rects, pr, ob["gap"]):
                        continue
                    start = a + (b - a) * rng.uniform(0, 1)
                    script = make_speed_script(rng, cfg["episode"]["length_s"] + 1.0, ob["person"]["speed"],
                                               ob["person"]["pause_probability"], ob["person"]["segment_duration"])
                    person = Person(start, True, a, b, script)
                else:
                    start = rng.uniform(-half + pr, half - pr, 2)
                    if any(G.rect_circle_overlap(r, (*start, pr), ob["gap"]) for r in rects):
                        continue
                    person = Person(start, False)
                if all(np.hypot(*(start - c[:2])) >= 2 * pr + ob["gap"] for c in circles):
                    break
            else:
                ok = False
                break
            people.append(person)
            circles.append((*person.start, pr))
        if not ok:
            reject("person placement")
            continue

        rb = cfg["robot"]
        for _try in range(50):
            m = rb["start_margin"]
            pose = np.array([*rng.uniform(-half + m, half - m, 2), rng.uniform(-np.pi, np.pi)])
            rr = G.robot_rect(pose, rb["footprint"])
            if not any(G.rects_overlap(rr, o, rb["start_clearance"]) for o in rects) and not any(
                    G.rect_circle_overlap(rr, c, rb["start_clearance"]) for c in circles):
                break
        else:
            reject("start pose")
            continue

        candidates = [("chair", i) for i in range(len(chairs))] + [("table", 0)] * len(table) + \
                     [("person", i) for i in range(len(people))]
        if target_kind is not None:
            candidates = [c for c in candidates if c[0] == target_kind]
            if not candidates:
                reject("no object of the requested kind")
                continue
        target = candidates[int(rng.integers(len(candidates)))]
        layout = Layout(pose, chairs, table, people, target, cfg)
        if _target_distance(layout) <= cfg["reachability"]["goal_distance"] + cfg["reachability"]["min_start_extra"]:
            reject("start too close to target")
            continue
        ok, info = reachable(layout, cfg)
        if not ok:
            reject("target unreachable: " + info["reason"])
            continue
        stats["accepted_after"] = sum(stats["rejected"].values())
        return layout, stats
    raise RuntimeError(f"no valid layout in {ob['max_tries']} tries: {stats}")


def layout_to_dict(layout):
    """JSON-ready description (used for the shared evaluation scenarios)."""
    return {
        "robot_start": layout.robot_start.tolist(),
        "chairs": layout.chairs.tolist(),
        "table": layout.table.tolist(),
        "people": [{"start": p.start.tolist(), "walking": p.walking,
                    "a": None if p.a is None else p.a.tolist(), "b": None if p.b is None else p.b.tolist(),
                    "script": p.script} for p in layout.people],
        "target": list(layout.target),
    }


def layout_from_dict(d, cfg=None):
    cfg = cfg or load_config("atrium")
    people = [Person(np.array(p["start"]), p["walking"], None if p["a"] is None else np.array(p["a"]),
                     None if p["b"] is None else np.array(p["b"]), [tuple(s) for s in p["script"]]) for p in d["people"]]
    return Layout(np.array(d["robot_start"]), np.array(d["chairs"]).reshape(-1, 4), np.array(d["table"]).reshape(-1, 3),
                  people, (d["target"][0], int(d["target"][1])), cfg)

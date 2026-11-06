"""From camera frames to what the navigation policy needs: the robot's pose and the obstacles, in the square frame.

- Robot: the tag on its back (config/tags.yaml robot), every frame. Its center and FRONT arrow on the plane at the
  tag's height give (x, y, yaw); forward_offset_m moves the point to the middle of the robot's body.
- Obstacles: OWLv2 ("a chair", "a table", "a person"; model and revision of config/grounding.yaml) on the newest
  frame at about owl.rate_hz, in its own thread. A box's bottom center is projected to the floor and moved
  half_depth_m further from the camera (the box bottom is the object's near edge). Obstacle tags, when present,
  give exact position and heading and replace detections near them.
- The target: a phrase ("red chair", "table", "person", "chair") or a tag id. Every pass also asks for a chair of
  each color ("a red chair", ...); such a detection is a chair of that color (the best-scoring color wins when
  detections of one chair overlap), and a colored target picks the chair detected with that color.

`to_scenario()` writes the result in config/nav_eval_scenarios.json format so go2x.deploy.nav_runtime.SquareScene
draws it exactly as in training. The scene has slots for 12 chairs (3 per color), 1 table and 2 people; extra
objects are dropped (nearest to the robot kept) and reported.
"""
import math
import threading
import time

import numpy as np

from go2x.config import load_config
from go2x.vision.geometry import TagDetector, tag_pose_on_plane

KINDS = ("chair", "table", "person")


class Perception:
    def __init__(self, cam, source, target, obstacles="both", tags_cfg=None, atrium_cfg=None, device=None,
                 on_detections=None):
        self.cam, self.source = cam, source
        self.T = tags_cfg or load_config("tags")
        self.A = atrium_cfg or load_config("atrium")
        self.colors = list(self.A["obstacles"]["chair"]["colors"])
        self.half = self.A["square"]["side"] / 2
        self.mode = obstacles                     # "owl", "tags" or "both"
        self.target = str(target)
        self.tags = TagDetector(self.T["dictionary"])
        self.device = device
        self.on_detections = on_detections
        self._lock = threading.Lock()
        self.robot = None                         # (t, pose (3,), tag edge m)
        self.tag_obstacles = []
        self.owl_objects, self.owl_t, self.target_hint = [], None, None
        self.frame, self.frame_t, self.frame_tags = None, None, {}
        self._run = threading.Event()
        self._owl = None
        self.error = None
        self.tracks, self.owl_passes = [], 0

    # -- threads ---------------------------------------------------------------------
    def start(self):
        self._run.set()
        self._threads = [threading.Thread(target=self._camera_loop, daemon=True)]
        if self.mode in ("owl", "both"):
            self._threads.append(threading.Thread(target=self._owl_loop, daemon=True))
        for th in self._threads:
            th.start()

    def stop(self, timeout_s=15.0):
        """Stop and wait for the threads (a detector pass in progress finishes first)."""
        self._run.clear()
        for th in getattr(self, "_threads", []):
            th.join(timeout_s)

    def _camera_loop(self):
        try:
            self._camera_loop_inner()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error = f"camera thread stopped: {e!r}"

    def _camera_loop_inner(self):
        rob = self.T["robot"]
        while self._run.is_set():
            got = self.source.read()
            if got is None:
                time.sleep(0.005)
                continue
            frame, t = got
            found = self.tags.detect(frame)
            robot, obst = None, []
            if rob["id"] in found:
                pose, edge = tag_pose_on_plane(self.cam, found[rob["id"]], rob["height_m"])
                pose[:2] += rob["forward_offset_m"] * np.array([math.cos(pose[2]), math.sin(pose[2])])
                robot = (t, pose, edge)
            for tid, spec in self.T["obstacles"].items():
                if int(tid) in found:
                    pose, _ = tag_pose_on_plane(self.cam, found[int(tid)], spec["height_m"])
                    obst.append({"kind": spec["kind"], "color": spec.get("color"), "x": pose[0], "y": pose[1],
                                 "yaw": pose[2], "source": f"tag {tid}", "score": 1.0})
            with self._lock:
                self.frame, self.frame_t, self.frame_tags = frame, t, found
                if robot is not None:
                    self.robot = robot
                if obst or self.mode == "tags":
                    self.tag_obstacles = obst

    def _owl_loop(self):
        try:
            self._owl_loop_inner()
        except Exception as e:         # surfaced by the runner (CameraMode.wait_ready and pose)
            import traceback
            traceback.print_exc()
            self.error = f"obstacle detector stopped: {e!r}"

    def _owl_loop_inner(self):
        from go2x.vision.owl import OwlDetector
        self._owl = OwlDetector(device=self.device)
        period = 1.0 / self.T["owl"]["rate_hz"]
        queries = ["a chair", "a table", "a person"] + [f"a {c} chair" for c in self.colors]
        while self._run.is_set():
            t0 = time.monotonic()
            with self._lock:
                frame, ft, robot = self.frame, self.frame_t, self.robot
            if frame is None:
                time.sleep(0.05)
                continue
            boxes, scores, labels = self._owl.detect(frame, queries)
            objs, hint = self._objects_from_boxes(boxes, scores, labels, queries, robot)
            kept = self._update_tracks(objs, ft)
            with self._lock:
                self.owl_objects, self.owl_t, self.target_hint = kept, ft, hint
                self.owl_passes += 1
            if self.on_detections:
                self.on_detections({"t": ft, "boxes": boxes.tolist(), "scores": scores.tolist(),
                                    "labels": [queries[i] for i in labels], "objects": objs})
            end = time.monotonic() + max(0.0, period - (time.monotonic() - t0))
            while self._run.is_set() and time.monotonic() < end:
                time.sleep(0.02)

    def _update_tracks(self, objs, t):
        """Obstacles persist for owl.forget_s after their last detection; a detection within owl.match_m of a
        kept obstacle of the same kind updates it (position averaged with the new one)."""
        o = self.T["owl"]
        for d in objs:
            near = [k for k in self.tracks if k["kind"] == d["kind"] and np.hypot(k["x"] - d["x"], k["y"] - d["y"]) < o["match_m"]]
            if near:
                k = min(near, key=lambda k: np.hypot(k["x"] - d["x"], k["y"] - d["y"]))
                k["x"], k["y"] = 0.5 * (k["x"] + d["x"]), 0.5 * (k["y"] + d["y"])
                if d.get("color") and d["score"] >= k.get("color_score", 0.0):     # best-scoring color wins
                    k["color"], k["color_score"] = d["color"], d["score"]
                k.update(last_t=t, hits=k["hits"] + 1, score=max(k["score"], d["score"]), box=d.get("box"))
            else:
                self.tracks.append({**d, "last_t": t, "hits": 1, "color_score": d["score"] if d.get("color") else 0.0})
        self.tracks = [k for k in self.tracks if t - k["last_t"] <= o["forget_s"]]
        return [dict(k) for k in self.tracks]

    # -- detections to floor objects ---------------------------------------------------
    def _target_color(self):
        words = self.target.lower().split()
        return next((c for c in self.colors if c in words), None)

    def _objects_from_boxes(self, boxes, scores, labels, queries, robot):
        o = self.T["owl"]
        objs, hint = [], None
        C = self.cam.C
        for b, s, li in zip(boxes, scores, labels):
            q = queries[li]
            kind = next(k for k in KINDS if k in q)
            if s < o["score_threshold"][kind]:
                continue
            p = self.cam.pixel_to_plane(np.array([[(b[0] + b[2]) / 2, b[3]]]), 0.0)[0]
            away = p[:2] - C[:2]
            away /= max(np.linalg.norm(away), 1e-6)
            xy = p[:2] + o["half_depth_m"][kind] * away
            if np.any(np.abs(xy) > self.half + o["keep_margin_m"]):
                continue
            if robot is not None and np.hypot(*(xy - robot[1][:2])) < o["robot_exclusion_m"]:
                continue
            color = q.split()[1] if len(q.split()) == 3 else None      # "a red chair"
            objs.append({"kind": kind, "color": color, "x": float(xy[0]), "y": float(xy[1]), "yaw": 0.0,
                         "source": "owl", "score": float(s), "box": [float(v) for v in b]})
        return objs, hint

    # -- state for the runner -----------------------------------------------------------
    def robot_pose(self):
        """(time, pose) of the newest robot tag fix, or None."""
        with self._lock:
            return None if self.robot is None else (self.robot[0], self.robot[1].copy())

    def objects(self):
        with self._lock:
            tag_objs = list(self.tag_obstacles)
            owl_objs = list(self.owl_objects) if self.mode in ("owl", "both") else []
            hint = self.target_hint
        r = self.T["owl"]["tag_override_m"]
        keep = [d for d in owl_objs if all(np.hypot(d["x"] - t["x"], d["y"] - t["y"]) > r for t in tag_objs)]
        return tag_objs + keep, hint

    def to_scenario(self, robot_pose):
        """(scenario dict, notes) for go2x.deploy.nav_runtime.SquareScene, or (None, reason)."""
        objs, hint = self.objects()
        rp = np.asarray(robot_pose[:2])
        objs.sort(key=lambda d: np.hypot(d["x"] - rp[0], d["y"] - rp[1]))
        notes = []
        target_obj = self._pick_target(objs, hint)
        if target_obj is None:
            return None, f"target '{self.target}' not found"
        chairs, table, people, per_color = [], [], [], {i: 0 for i in range(len(self.colors))}
        cap = self.A["slots"]["chairs_per_color"]
        n_people = self.A["obstacles"]["person"]["count"][1]
        target = None
        for d in objs:
            if d["kind"] == "chair":
                c = self.colors.index(d["color"]) if d.get("color") in self.colors else None
                if c is None or per_color[c] >= cap:      # color unknown or slots full: any free color slot
                    c = next((k for k, v in per_color.items() if v < cap), None)
                if c is None:
                    notes.append("chair dropped (no slot)")
                    continue
                per_color[c] += 1
                if d is target_obj:
                    target = ["chair", len(chairs)]
                chairs.append([d["x"], d["y"], d["yaw"], float(c)])
            elif d["kind"] == "table":
                if table:
                    notes.append("table dropped (one table slot)")
                    continue
                if d is target_obj:
                    target = ["table", 0]
                table.append([d["x"], d["y"], d["yaw"]])
            else:
                if len(people) >= n_people:
                    notes.append("person dropped (two person slots)")
                    continue
                if d is target_obj:
                    target = ["person", len(people)]
                people.append({"start": [d["x"], d["y"]], "walking": False, "a": None, "b": None, "script": []})
        if target is None:
            return None, "target dropped for lack of a slot"
        scen = {"robot_start": [float(v) for v in robot_pose], "chairs": chairs, "table": table, "people": people,
                "target": target}
        return scen, "; ".join(notes)

    def _pick_target(self, objs, hint):
        t = self.target.lower().strip()
        if t.startswith("tag:"):
            src = f"tag {int(t[4:])}"
            return next((d for d in objs if d["source"] == src), None)
        kind = next((k for k in KINDS if k in t), None)
        if kind is None:
            raise ValueError(f"target must name chair, table or person, or be tag:<id>: {self.target!r}")
        cands = [d for d in objs if d["kind"] == kind]
        color = self._target_color()
        if color:
            same = [d for d in cands if d.get("color") == color]
            return max(same, key=lambda d: d["score"]) if same else None
        if not cands:
            return None
        return max(cands, key=lambda d: d["score"])

"""Camera mode for scripts/robot/run_nav_policy.py: pose from the robot tag (odometry between fixes), scene from
the detected obstacles, and the camera recording.

Pose: a fresh camera fix (younger than tracking.pose_max_age_s) is used as is and re-anchors the odometry
mapping (go2x.deploy.nav_runtime.OdomToSquare) at that pose; between fixes the odometry carries the pose from the
last fix. With no fix for tracking.camera_loss_stop_s the runner stops the robot.
Scene: rebuilt (go2x.deploy.nav_runtime.SquareScene) whenever the perceived layout changes by more than 5 cm.
Records in the run folder: camera.mp4 (frames at the policy rate with the square, tags, robot pose and obstacles
drawn), detections.jsonl (every OWLv2 pass), perception.jsonl (every layout given to the policy, with time).
"""
import json
import time

import numpy as np

from go2x.config import load_config
from go2x.deploy.nav_runtime import OdomToSquare, SquareScene
from go2x.vision import annotate
from go2x.vision.perception import Perception


class CameraMode:
    def __init__(self, cam, source, target, obstacles="both", tags_cfg=None, device=None):
        self.cam, self.source = cam, source
        self.T = tags_cfg or load_config("tags")
        self.A = load_config("atrium")
        self.det_file = None
        self.perc = Perception(cam, source, target, obstacles, self.T, self.A, device, on_detections=self._on_det)
        self.mapper, self.scene, self.scen, self._key = None, None, None, None
        self.last_fix_t = None
        self.writer = None
        self.pose_source = None

    def start(self, run_dir):
        import imageio
        self.dir = run_dir
        self.det_file = open(run_dir / "detections.jsonl", "w")
        self.perc_file = open(run_dir / "perception.jsonl", "w")
        self.writer = imageio.get_writer(run_dir / "camera.mp4", fps=10, quality=7, macro_block_size=8)
        self.perc.start()

    def _on_det(self, d):
        if self.det_file and not self.det_file.closed:
            self.det_file.write(json.dumps({"t_mono": time.monotonic(), **d}, default=float) + "\n")

    def wait_ready(self, check, timeout_s=90.0):
        """Block until the robot tag and the target are seen. Returns the first scenario dict."""
        t_end = time.monotonic() + timeout_s
        msg = ""
        while time.monotonic() < t_end:
            check()
            if self.perc.error:
                raise RuntimeError(self.perc.error)
            fix = self.perc.robot_pose()
            if self.perc.mode in ("owl", "both") and self.perc.owl_passes < self.T["owl"]["min_passes"]:
                msg = f"obstacle detector: {self.perc.owl_passes} of {self.T['owl']['min_passes']} passes"
            elif fix is None:
                msg = "robot tag not seen"
            else:
                scen, note = self.perc.to_scenario(fix[1])
                if scen is not None:
                    return scen
                msg = note
            time.sleep(0.2)
        raise TimeoutError(f"camera not ready after {timeout_s:.0f} s: {msg}")

    def pose(self, odom, now=None):
        """Robot pose (x, y, yaw) in the square frame and its source ('camera' or 'odometry'), or raise."""
        now = time.monotonic() if now is None else now
        fix = self.perc.robot_pose()
        tr = self.T["tracking"]
        if self.perc.error:
            raise LookupError(self.perc.error)
        if fix is not None and now - fix[0] <= tr["pose_max_age_s"]:
            self.mapper = OdomToSquare(fix[1], odom)
            self.last_fix_t = fix[0]
            self.pose_source = "camera"
            return fix[1], "camera"
        if self.mapper is None or self.last_fix_t is None or now - self.last_fix_t > tr["camera_loss_stop_s"]:
            raise LookupError("the camera has not seen the robot tag for "
                              f"{'ever' if self.last_fix_t is None else f'{now - self.last_fix_t:.1f} s'}")
        self.pose_source = "odometry"
        return self.mapper(odom), "odometry"

    def scene_for(self, pose, t):
        scen, note = self.perc.to_scenario(pose)
        if scen is None:
            return self.scene, f"keeping the last layout ({note})"
        key = np.round(np.array([v for c in scen["chairs"] for v in c[:2]] + [v for c in scen["table"] for v in c[:2]]
                                + [v for p in scen["people"] for v in p["start"]]) / 0.05).astype(int).tobytes() \
            + str(scen["target"]).encode()
        if key != self._key or self.scene is None:
            self._key, self.scen = key, scen
            self.scene = SquareScene(scen, self.A)
            self.perc_file.write(json.dumps({"t_mono": t, "scenario": scen, "note": note}, default=float) + "\n")
        return self.scene, note

    def record(self, pose, text):
        with self.perc._lock:
            frame, tags = self.perc.frame, dict(self.perc.frame_tags)
        if frame is None or self.writer is None:
            return
        img = annotate.draw(frame, self.cam, self.A["square"]["side"] / 2, tags, pose,
                            tuple(self.A["robot"]["footprint"]), self.scen, self.A, text)
        if img.shape[1] > 1280:
            import cv2
            s = 1280 / img.shape[1]
            img = cv2.resize(img, (1280, int(round(img.shape[0] * s)) // 2 * 2))
        self.writer.append_data(img)

    def close(self):
        self.perc.stop()
        for f in (self.det_file, getattr(self, "perc_file", None)):
            if f and not f.closed:
                f.close()
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        try:
            self.source.close()
        except Exception:
            pass

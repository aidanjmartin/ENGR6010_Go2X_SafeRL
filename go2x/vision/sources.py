"""Frame sources for go2x.vision.perception.Perception: a USB camera (OpenCV) or the MuJoCo stand-in's camera.

Both give `read()` -> (RGB frame, monotonic time) for each new frame, or None when there is no new frame.
"""
import threading
import time

import cv2
import numpy as np

from go2x.vision.synth import overlay_tag


class USBCamera:
    """OpenCV capture in its own thread, keeping only the newest frame (no queue, so no growing delay).
    On macOS the first run asks for camera permission for the terminal app."""

    def __init__(self, index=0, width=1920, height=1080, fps=30):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"camera {index} did not open")
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self._lock = threading.Lock()
        self._frame, self._t, self._seen = None, None, True
        self._run = threading.Event()
        self._run.set()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self._run.is_set():
            ok, bgr = self.cap.read()
            t = time.monotonic()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame, self._t, self._seen = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), t, False

    def read(self):
        with self._lock:
            if self._seen or self._frame is None:
                return None
            self._seen = True
            return self._frame, self._t

    def close(self):
        self._run.clear()
        time.sleep(0.05)
        self.cap.release()


class SimCamera:
    """Frames rendered by go2x.deploy.mujoco_robot.MujocoRobot from the overhead camera, with the printed tags
    drawn in (go2x.vision.synth): the four corner tags on the floor, the robot tag fixed to the robot's body
    (it tilts and bobs with the body) and, if `obstacle_tags`, a tag on each obstacle's top."""

    def __init__(self, cam, tags_cfg, robot_tag_local=(0.0, 0.0, 0.06), obstacle_tags=False):
        self.cam, self.T = cam, tags_cfg
        self.robot_tag_local = np.asarray(robot_tag_local, float)
        self.obstacle_tags = obstacle_tags
        self._lock = threading.Lock()
        self._frame, self._t, self._seen = None, None, True

    def push(self, frame, t, base_pos, base_R, obstacles):
        """Called by the MuJoCo backend's physics thread with a fresh render and the true poses."""
        T, size = self.T, self.T["tag_size_m"]
        for tid, (x, y) in zip(T["corners"]["ids"], T["corners"]["xy"]):
            overlay_tag(frame, self.cam, tid, x, y, 0.0, size, 0.0, T["dictionary"])
        c = base_pos + base_R @ self.robot_tag_local
        fwd = base_R[:, 0]
        overlay_tag(frame, self.cam, T["robot"]["id"], c[0], c[1], float(np.arctan2(fwd[1], fwd[0])), size, c[2],
                    T["dictionary"])
        if self.obstacle_tags:
            for o in obstacles:
                tid = next((int(k) for k, v in T["obstacles"].items() if v["kind"] == o["kind"]
                            and (o["kind"] != "chair" or v.get("color") == o["color"])), None)
                if tid is not None:
                    overlay_tag(frame, self.cam, tid, o["x"], o["y"], o["yaw"], size,
                                T["obstacles"][tid]["height_m"], T["dictionary"])
        with self._lock:
            self._frame, self._t, self._seen = frame, t, False

    def read(self):
        with self._lock:
            if self._seen or self._frame is None:
                return None
            self._seen = True
            return self._frame, self._t

    def close(self):
        pass

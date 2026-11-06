"""The calibration scripts (scripts/camera) on synthetic images from a known camera."""
import math
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from go2x.atrium.camera import Camera
from go2x.config import load_config
from go2x.vision.geometry import ARUCO_DICTS, RealCamera
from go2x.vision.synth import overlay_tag

ROOT = Path(__file__).resolve().parents[1]
ENV = {**os.environ, "PYTHONPATH": str(ROOT)}


def _rot(rx, ry, rz):
    return cv2.Rodrigues(np.array([rx, ry, rz], float))[0]


def test_lens_calibration_recovers_the_camera_matrix(tmp_path):
    c = load_config("tags")["charuco"]
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[c["dictionary"]])
    board = cv2.aruco.CharucoBoard((c["squares_x"], c["squares_y"]), c["square_m"], c["marker_m"], d)
    bw, bh = c["squares_x"] * c["square_m"], c["squares_y"] * c["square_m"]
    img_px = 1000
    bimg = board.generateImage((img_px, int(img_px * bh / bw)), marginSize=0)
    W, H = 1280, 720
    K = np.array([[900.0, 0, 650], [0, 900.0, 350], [0, 0, 1]])
    rng = np.random.default_rng(0)
    src = np.float32([[0, 0], [bimg.shape[1] - 1, 0], [bimg.shape[1] - 1, bimg.shape[0] - 1], [0, bimg.shape[0] - 1]])
    corners3 = np.array([[0, 0, 0], [bw, 0, 0], [bw, bh, 0], [0, bh, 0]], float)
    n = 0
    for i in range(40):
        R = _rot(*rng.uniform(-0.5, 0.5, 3))
        t = np.array([rng.uniform(-0.15, 0.05), rng.uniform(-0.15, 0.0), rng.uniform(0.45, 0.8)])
        Xc = corners3 @ R.T + t
        uv = (Xc @ K.T)[:, :2] / Xc[:, 2:3]
        if np.any(uv < 0) or np.any(uv[:, 0] >= W) or np.any(uv[:, 1] >= H):
            continue
        M = cv2.getPerspectiveTransform(src, uv.astype(np.float32))
        frame = cv2.warpPerspective(bimg, M, (W, H), borderValue=200)
        cv2.imwrite(str(tmp_path / f"v{i:02d}.png"), frame)
        n += 1
        if n == 15:
            break
    out = tmp_path / "cal.yaml"
    r = subprocess.run([sys.executable, str(ROOT / "scripts/camera/calibrate_intrinsics.py"), "--images", str(tmp_path),
                        "--out", str(out)], capture_output=True, text=True, env=ENV, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    Ke = np.array(yaml.safe_load(open(out))["intrinsics"]["K"])
    assert abs(Ke[0, 0] / 900 - 1) < 0.02 and abs(Ke[1, 1] / 900 - 1) < 0.02
    assert abs(Ke[0, 2] - 650) < 10 and abs(Ke[1, 2] - 350) < 10


def test_extrinsic_calibration_script(tmp_path):
    T = load_config("tags")
    truth = RealCamera.from_atrium_camera(Camera.from_config(load_config("camera")), scale=2.0)
    img = np.full((truth.height, truth.width, 3), 150, np.uint8)
    for tid, (x, y) in zip(T["corners"]["ids"], T["corners"]["xy"]):
        overlay_tag(img, truth, tid, x, y, 0.3 * tid, T["tag_size_m"], 0.0)     # any rotation of the corner tags
    cv2.imwrite(str(tmp_path / "frame.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    calib = tmp_path / "cal.yaml"
    intr = RealCamera(truth.K, truth.dist, truth.width, truth.height)
    calib.write_text(yaml.safe_dump(intr.to_dict()))
    r = subprocess.run([sys.executable, str(ROOT / "scripts/camera/calibrate_extrinsics.py"), "--image",
                        str(tmp_path / "frame.png"), "--calib", str(calib), "--check-out", str(tmp_path / "check.png")],
                       capture_output=True, text=True, env=ENV, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    est = RealCamera.load(calib)
    assert np.linalg.norm(est.C - truth.C) < 0.05
    ang = math.degrees(math.acos(np.clip((np.trace(est.R_wc.T @ truth.R_wc) - 1) / 2, -1, 1)))
    assert ang < 1.0 and (tmp_path / "check.png").exists()

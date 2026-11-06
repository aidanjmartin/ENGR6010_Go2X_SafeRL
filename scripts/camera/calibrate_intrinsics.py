"""Lens calibration of the overhead webcam with the printed ChArUco board (config/tags.yaml charuco).

Live: hold the board in front of the camera (any distance where it fills a good part of the image) and move it:
tilt it, turn it, cover the corners and the middle of the image. A frame is kept automatically when the board is
seen well and differs from the kept ones; at --frames kept frames the calibration runs. Press q to stop early.
Writes the intrinsics (camera matrix, distortion, image size) to --out, keeping any extrinsics already there.

    python scripts/camera/calibrate_intrinsics.py --camera 0 --width 1920 --height 1080
    python scripts/camera/calibrate_intrinsics.py --images folder_of_pngs --out cal.yaml      (offline)
"""
import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from go2x.config import load_config  # noqa: E402
from go2x.vision.geometry import ARUCO_DICTS  # noqa: E402


def board_and_detector():
    c = load_config("tags")["charuco"]
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[c["dictionary"]])
    board = cv2.aruco.CharucoBoard((c["squares_x"], c["squares_y"]), c["square_m"], c["marker_m"], d)
    return board, cv2.aruco.CharucoDetector(board)


def detect(det, board, gray, min_corners=10):
    cc, ci, _, _ = det.detectBoard(gray)
    if ci is None or len(ci) < min_corners:
        return None
    obj, img = board.matchImagePoints(cc, ci)
    return obj.reshape(-1, 3).astype(np.float32), img.reshape(-1, 2).astype(np.float32)


def calibrate(views, size):
    obj = [v[0] for v in views]
    img = [v[1] for v in views]
    rms, K, dist, _, _ = cv2.calibrateCamera(obj, img, size, None, None)
    return rms, K, dist.ravel()


def save(out, K, dist, size, rms, n):
    d = {}
    if Path(out).exists():
        d = yaml.safe_load(open(out)) or {}
    d["intrinsics"] = {"K": K.tolist(), "dist": [float(v) for v in dist], "width": int(size[0]), "height": int(size[1]),
                       "rms_px": float(rms), "views": int(n)}
    d.pop("extrinsics", None)          # the camera pose depends on the intrinsics: solve it again afterwards
    Path(out).write_text("# written by scripts/camera/calibrate_intrinsics.py\n" + yaml.safe_dump(d, sort_keys=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--images", default=None, help="calibrate from the images in this folder instead of the camera")
    p.add_argument("--frames", type=int, default=25)
    p.add_argument("--out", default=str(ROOT / "config/camera_real.yaml"))
    args = p.parse_args()
    board, det = board_and_detector()
    views, size = [], None
    if args.images:
        for f in sorted(glob.glob(str(Path(args.images) / "*.png")) + glob.glob(str(Path(args.images) / "*.jpg"))):
            g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            size = (g.shape[1], g.shape[0])
            v = detect(det, board, g)
            if v is not None:
                views.append(v)
        print(f"{len(views)} usable images")
    else:
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        while len(views) < args.frames:
            ok, bgr = cap.read()
            if not ok:
                continue
            g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            size = (g.shape[1], g.shape[0])
            v = detect(det, board, g, min_corners=12)
            show = bgr.copy()
            if v is not None:
                for u in v[1]:
                    cv2.circle(show, tuple(int(a) for a in u), 6, (0, 255, 0), 2)
                centroid = v[1].mean(0)
                spread = np.ptp(v[1], axis=0).mean()
                if all(np.linalg.norm(centroid - w[1].mean(0)) > 0.08 * size[0] or
                       abs(spread - np.ptp(w[1], axis=0).mean()) > 0.15 * spread for w in views):
                    views.append(v)
            cv2.putText(show, f"kept {len(views)}/{args.frames}: move and tilt the board; q to stop", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.imshow("lens calibration", cv2.resize(show, (size[0] // 2, size[1] // 2)))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
    if len(views) < 8:
        raise SystemExit(f"only {len(views)} usable views; need at least 8")
    rms, K, dist = calibrate(views, size)
    save(args.out, K, dist, size, rms, len(views))
    print(f"image {size[0]} x {size[1]}, {len(views)} views, reprojection error {rms:.3f} px (under 1 px is good)")
    print(f"fx {K[0, 0]:.1f} fy {K[1, 1]:.1f} cx {K[0, 2]:.1f} cy {K[1, 2]:.1f}; distortion {np.round(dist, 4).tolist()}")
    print(f"wrote {args.out}; next: scripts/camera/calibrate_extrinsics.py (the camera must not move or refocus now)")


if __name__ == "__main__":
    main()

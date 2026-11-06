"""Camera pose over the square from the four corner tags (config/tags.yaml corners), after the lens calibration.

Place the corner tags flat with their centers exactly on the square's corners and keep the square clear. The
script averages the tag centers over --frames frames, solves the camera pose, and writes it to --calib. It prints
the camera position (height above the floor), the reprojection error, and how large the robot tag will look,
and saves calib_check.png with the square and axes drawn over the image: the square must sit on the tape.

    python scripts/camera/calibrate_extrinsics.py --camera 0
    python scripts/camera/calibrate_extrinsics.py --image frame.png --calib cal.yaml      (offline)
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from go2x.config import load_config  # noqa: E402
from go2x.vision import annotate  # noqa: E402
from go2x.vision.geometry import RealCamera, TagDetector, tag_center_px  # noqa: E402


def solve(cam, centers, T):
    ids = T["corners"]["ids"]
    uv = np.array([np.mean(centers[i], 0) for i in ids])
    X = np.array([[x, y, 0.0] for x, y in T["corners"]["xy"]])
    return cam.solve_extrinsics(uv, X)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--image", default=None, help="one saved frame instead of the camera")
    p.add_argument("--frames", type=int, default=30)
    p.add_argument("--calib", default=str(ROOT / "config/camera_real.yaml"))
    p.add_argument("--check-out", default="calib_check.png")
    args = p.parse_args()
    T = load_config("tags")
    cam = RealCamera.load(args.calib)
    det = TagDetector(T["dictionary"])
    ids = T["corners"]["ids"]
    centers = {i: [] for i in ids}
    last = None
    if args.image:
        last = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
        frames = [last]
    else:
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam.height)
        frames = []
        for _ in range(args.frames * 3):
            ok, bgr = cap.read()
            if ok:
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            if len(frames) >= args.frames:
                break
        cap.release()
    for f in frames:
        if (f.shape[1], f.shape[0]) != (cam.width, cam.height):
            raise SystemExit(f"frame is {f.shape[1]} x {f.shape[0]}, the lens calibration is for {cam.width} x {cam.height}")
        found = det.detect(f)
        for i in ids:
            if i in found:
                centers[i].append(tag_center_px(found[i]))
        last = f
    missing = [i for i in ids if not centers[i]]
    if missing:
        raise SystemExit(f"corner tags not seen: {missing}. Check that all four are in the image, flat and lit.")
    mean_err, max_err = solve(cam, centers, T)
    d = yaml.safe_load(open(args.calib))
    d["extrinsics"] = {"R_wc": cam.R_wc.tolist(), "C": cam.C.tolist(), "reprojection_px": [mean_err, max_err],
                       "frames": len(frames)}
    Path(args.calib).write_text("# written by scripts/camera/calibrate_*.py\n" + yaml.safe_dump(d, sort_keys=False))
    # how big the robot tag looks in the middle of the square
    h = T["robot"]["height_m"]
    s = T["tag_size_m"] / 2
    uv = cam.project(np.array([[-s, -s, h], [s, -s, h], [s, s, h], [-s, s, h]]))
    edge_px = np.mean([np.linalg.norm(uv[i] - uv[(i + 1) % 4]) for i in range(4)])
    half = load_config("atrium")["square"]["side"] / 2
    corners_uv = cam.project(np.array([[x, y, 0] for x in (-half, half) for y in (-half, half)]))
    inside = np.all((corners_uv >= 0) & (corners_uv < [cam.width, cam.height]))
    print(f"camera at x {cam.C[0]:+.2f} y {cam.C[1]:+.2f} m, {cam.C[2]:.2f} m above the floor; "
          f"distance to the square center {np.linalg.norm(cam.C):.2f} m")
    print(f"reprojection error of the corner tags: mean {mean_err:.2f} px, max {max_err:.2f} px (under 2 px is good)")
    print(f"robot tag in the square's center: about {edge_px:.0f} px wide (above 25 px is comfortable for detection)")
    print("whole square in view" if inside else "WARNING: part of the square is outside the image")
    img = annotate.draw(last, cam, half, det.detect(last), None)
    cv2.imwrite(args.check_out, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"wrote {args.calib} and {args.check_out} (the white square must sit on the square's tape)")


if __name__ == "__main__":
    main()

"""One command for a camera run on the Go2 X: checks, floor preview, the run, then the digital-twin video.

Steps (each one stops with a plain message if something is wrong; nothing moves before step 6's 'yes'):
1. Calibration: config/camera_real.yaml must have the lens calibration and the camera pose; its `controls`
   (fixed exposure, go2x/vision/uvc.py) are applied to the webcam, since they reset when it loses power.
2. Overhead camera: tries each camera index and takes the one that sees the four corner tags. Camera indices
   on macOS change when cameras are plugged in again, so the index is never trusted.
3. Camera drift: the corner tags must still be where the calibration puts them (--max-drift-px). If the camera
   or a corner tag moved, it stops; --recalibrate-floor solves the camera pose again first.
4. OWLv2 weights in the local cache (the robot's hotspot has no internet); runs offline from here on.
5. Robot link: the robot's WebRTC port answers (hotspot 192.168.12.1 with --method ap, or --ip / GO2_IP).
6. Preview (scripts/camera/preview.py): place the obstacles and the robot until the picture matches the floor,
   then press q. Skip with --no-preview.
7. The run (scripts/robot/run_nav_policy.py, camera mode); it asks for 'yes' before anything is sent.
8. The digital-twin video of the run (scripts/robot/render_twin.py).

    python scripts/robot/start_run.py --target "red chair"
    python scripts/robot/start_run.py --target "red chair" --check-only      (steps 1 to 5, no robot needed for 1-4)
"""
import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from go2x.config import load_config  # noqa: E402
from go2x.vision.geometry import RealCamera, TagDetector, tag_center_px  # noqa: E402
from go2x.vision.uvc import apply_controls, load_controls  # noqa: E402

PY = sys.executable
HOTSPOT_IP = "192.168.12.1"        # unitree_webrtc_connect e0abc57, webrtc_driver.py:88 (LocalAP)
SIGNAL_PORTS = (9991, 8081)        # unitree_auth.py:71 (new firmware, legacy firmware)


def step(n, text):
    print(f"\n[{n}] {text}", flush=True)


def fail(text):
    print(f"\nSTOPPED: {text}", flush=True)
    sys.exit(1)


def corner_centers(index, cam, T, frames=15, warmup_s=1.5):
    """Mean pixel center of each corner tag seen by camera `index` ({id: (2,)}), or None if it does not open."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam.height)
    det, ids = TagDetector(T["dictionary"]), T["corners"]["ids"]
    t0, seen, n, size = time.monotonic(), {i: [] for i in ids}, 0, None
    while time.monotonic() - t0 < warmup_s:          # auto exposure settles
        cap.read()
    while n < frames and time.monotonic() - t0 < warmup_s + 5:
        ok, bgr = cap.read()
        if not ok:
            continue
        n += 1
        size = (bgr.shape[1], bgr.shape[0])
        found = det.detect(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        for i in ids:
            if i in found:
                seen[i].append(tag_center_px(found[i]))
    cap.release()
    return {"size": size, "centers": {i: np.mean(v, 0) for i, v in seen.items() if v}}


def find_overhead_camera(cam, T, max_index=4):
    results = {}
    for i in range(max_index):
        r = corner_centers(i, cam, T)
        if r is None:
            continue
        results[i] = r
        print(f"    camera {i}: {r['size'][0] if r['size'] else '?'} x {r['size'][1] if r['size'] else '?'}, "
              f"corner tags seen {sorted(r['centers'])}", flush=True)
    full = [i for i, r in results.items() if len(r["centers"]) == len(T["corners"]["ids"])]
    if not full:
        best = max(results, key=lambda i: len(results[i]["centers"]), default=None)
        missing = sorted(set(T["corners"]["ids"]) - set(results[best]["centers"])) if best is not None else "all"
        fail(f"no camera sees all four corner tags (best: camera {best}, missing {missing}). Check that the webcam "
             "is plugged in, that every corner tag is in its picture, flat, lit and not covered.")
    if len(full) > 1:
        fail(f"cameras {full} all see the four corner tags; unplug the extra one or pass --camera-index.")
    return full[0], results[full[0]]


def link_ok(ip, timeout=2.0):
    for port in SIGNAL_PORTS:
        try:
            with socket.create_connection((ip, port), timeout=timeout):
                return port
        except OSError:
            continue
    return None


def env_value(key):
    if os.environ.get(key):
        return os.environ[key]
    f = ROOT / ".env"
    if f.exists():
        for line in f.read_text().splitlines():
            if line.strip().startswith(f"{key}="):
                return line.split("=", 1)[1].strip() or None
    return None


def run(cmd, env=None):
    print("    $ " + " ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
    return subprocess.call(cmd, cwd=ROOT, env=env)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--target", required=True, help='"red chair", "chair", "table", "person" or tag:<id>')
    p.add_argument("--method", choices=["ap", "sta"], default="ap", help="ap: laptop on the robot's hotspot")
    p.add_argument("--ip", default=None, help="robot address for --method sta (default GO2_IP in .env)")
    p.add_argument("--camera-index", type=int, default=None, help="skip the camera search")
    p.add_argument("--calib", default=str(ROOT / "config/camera_real.yaml"))
    p.add_argument("--max-drift-px", type=float, default=3.0)
    p.add_argument("--recalibrate-floor", action="store_true", help="solve the camera pose again before the run")
    p.add_argument("--obstacles", choices=["owl", "tags", "both"], default="both")
    p.add_argument("--no-preview", action="store_true")
    p.add_argument("--no-twin", action="store_true")
    p.add_argument("--check-only", action="store_true", help="steps 1 to 5 only")
    p.add_argument("--auto-exposure", action="store_true", help="leave the webcam on auto exposure")
    args = p.parse_args()
    T = load_config("tags")

    step(1, "calibration")
    if not Path(args.calib).exists():
        fail(f"{args.calib} missing: run scripts/camera/calibrate_intrinsics.py, then calibrate_extrinsics.py")
    cam = RealCamera.load(args.calib)
    print(f"    lens: {cam.width} x {cam.height}, fx {cam.K[0, 0]:.0f}; camera pose "
          f"{'set, %.2f m above the floor' % cam.C[2] if cam.R_wc is not None else 'NOT SET'}")
    r = T["robot"]
    print(f"    config/tags.yaml: robot tag height {r['height_m']:.3f} m, {r['forward_offset_m']:+.3f} m ahead of the "
          f"body's middle, turned {r.get('yaw_offset_deg', 0.0):+.0f} deg")
    controls = load_controls(args.calib)
    if controls and not args.auto_exposure:
        try:
            got = apply_controls(controls)
        except RuntimeError as e:
            fail(f"{e}. Plug the webcam in, or pass --auto-exposure.")
        print(f"    exposure locked on '{controls['device']}': " + ", ".join(f"{k} {v}" for k, v in got.items()))
    else:
        print("    webcam on auto exposure" + ("" if controls else f" (no `controls` in {args.calib})"))

    step(2, "overhead camera")
    if args.camera_index is not None:
        idx, seen = args.camera_index, corner_centers(args.camera_index, cam, T)
        if seen is None:
            fail(f"camera {idx} does not open")
    else:
        idx, seen = find_overhead_camera(cam, T)
    if seen["size"] != (cam.width, cam.height):
        fail(f"camera {idx} gives {seen['size']}, the calibration is for {(cam.width, cam.height)}")
    print(f"    using camera {idx}")

    step(3, "camera drift")
    if cam.R_wc is None or args.recalibrate_floor:
        print("    solving the camera pose from the corner tags (square must be clear) ...")
        if run([PY, "scripts/camera/calibrate_extrinsics.py", "--camera", str(idx), "--calib", args.calib,
                "--check-out", str(ROOT / "data" / "calib_check.png")]) != 0:
            fail("camera pose calibration failed (message above)")
        cam = RealCamera.load(args.calib)
    ids = T["corners"]["ids"]
    import yaml
    ex = (yaml.safe_load(open(args.calib)) or {}).get("extrinsics") or {}
    if "corner_px" in ex:       # where the floor calibration saw each tag
        ref = np.array([ex["corner_px"][i] for i in ids], float)
    else:                       # older calibration: where it puts the corners (includes the fit's own error)
        ref = cam.project(np.array([[x, y, 0.0] for x, y in T["corners"]["xy"]]))
    drift = np.linalg.norm(ref - np.array([seen["centers"][i] for i in ids]), axis=1)
    print(f"    floor calibration fit: {ex.get('reprojection_px', ['?'])[0]:.1f} px mean" if ex.get("reprojection_px")
          else "    floor calibration fit: not recorded")
    print("    corner tags vs. calibration: " + ", ".join(f"{i}: {d:.1f} px" for i, d in zip(ids, drift)))
    if drift.max() > args.max_drift_px:
        fail(f"a corner tag is {drift.max():.1f} px from where the calibration puts it (limit {args.max_drift_px} px): "
             "the camera or a corner tag moved. Put the tag back on its corner, or run again with "
             "--recalibrate-floor (square clear).")

    step(4, "OWLv2 weights (offline)")
    if args.obstacles != "tags":
        from huggingface_hub import snapshot_download
        dc = load_config("grounding")["detector"]
        try:
            path = snapshot_download(dc["model"], revision=dc["revision"], local_files_only=True)
            print(f"    cached: {path}")
        except Exception as e:
            fail(f"OWLv2 weights not in the cache ({e.__class__.__name__}). Connect to the internet once and run "
                 "scripts/camera/preview.py --owl, then come back.")
    else:
        print("    not needed (--obstacles tags)")
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")

    step(5, "robot link")
    ip = HOTSPOT_IP if args.method == "ap" else (args.ip or env_value("GO2_IP"))
    if not ip:
        fail("--method sta needs --ip or GO2_IP in .env")
    port = link_ok(ip)
    if port is None:
        hint = ("join this Mac to the robot's Wi-Fi hotspot" if args.method == "ap"
                else "check that the robot and this Mac are on the same network")
        if args.check_only:
            print(f"    robot not reachable at {ip} (ports {SIGNAL_PORTS}): {hint}")
        else:
            fail(f"robot not reachable at {ip} (ports {SIGNAL_PORTS}): {hint}")
    else:
        print(f"    robot answers at {ip}:{port}")
    if args.check_only:
        print("\ncheck only: done.")
        return

    if not args.no_preview:
        step(6, "preview: place the obstacles and the robot until the picture matches the floor, then CLICK THE "
                "PREVIEW WINDOW and press q there (keys typed in this terminal are discarded)")
        if run([PY, "scripts/camera/preview.py", "--camera", str(idx), "--calib", args.calib, "--owl",
                "--target", args.target], env=env) != 0:
            fail("preview ended with an error")

    step(7, "run")
    before = set((ROOT / "data").glob("*_nav_policy_webrtc"))
    cmd = [PY, "scripts/robot/run_nav_policy.py", "--backend", "webrtc", "--method", args.method, "--camera", "usb",
           "--camera-index", str(idx), "--calib", args.calib, "--target", args.target, "--obstacles", args.obstacles]
    if args.method == "sta":
        cmd += ["--ip", ip]
    code = run(cmd, env=env)
    new = sorted(set((ROOT / "data").glob("*_nav_policy_webrtc")) - before)
    if not new:
        fail(f"the run made no log folder (exit code {code})")
    run_dir = new[-1]

    if not args.no_twin:
        step(8, "digital-twin video")
        run([PY, "scripts/robot/render_twin.py", str(run_dir), "--model_dir", "third_party/unitree_mujoco"], env=env)
    print(f"\nrun folder: {run_dir}")
    for f in ("camera.mp4", "twin.mp4"):
        if (run_dir / f).exists():
            print(f"    {f}")
    sys.exit(code)


if __name__ == "__main__":
    main()

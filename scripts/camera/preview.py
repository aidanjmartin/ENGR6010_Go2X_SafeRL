"""Live view for setting up the floor from the laptop: what the camera pipeline sees, before any robot run.

Draws the calibrated square and axes, every tag, the robot's pose from its tag, and (with --owl) the OWLv2
obstacles and the layout the policy would get for --target. Sends nothing to the robot.
Keys: q quits, s saves a snapshot (preview_<time>.png).

--owl also downloads the OWLv2 weights on first use (about 0.6 GB): run it once with internet, before joining the
robot's Wi-Fi hotspot.

    python scripts/camera/preview.py --camera 0 --owl --target "red chair"
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from go2x.config import load_config  # noqa: E402
from go2x.vision import annotate  # noqa: E402
from go2x.vision.geometry import RealCamera  # noqa: E402
from go2x.vision.perception import Perception  # noqa: E402
from go2x.vision.sources import USBCamera  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--calib", default=str(ROOT / "config/camera_real.yaml"))
    p.add_argument("--owl", action="store_true", help="run OWLv2 obstacle detection too")
    p.add_argument("--target", default="chair")
    p.add_argument("--scale", type=float, default=0.5, help="window size relative to the camera image")
    args = p.parse_args()
    T, A = load_config("tags"), load_config("atrium")
    cam = RealCamera.load(args.calib)
    if cam.R_wc is None:
        raise SystemExit(f"{args.calib} has no extrinsics: run scripts/camera/calibrate_extrinsics.py first")
    src = USBCamera(args.camera, cam.width, cam.height)
    perc = Perception(cam, src, args.target, "both" if args.owl else "tags", T, A)
    perc.start()
    half = A["square"]["side"] / 2
    t_fps, n = time.monotonic(), 0
    try:
        while True:
            with perc._lock:
                frame, tags = perc.frame, dict(perc.frame_tags)
            if frame is None:
                time.sleep(0.02)
                continue
            fix = perc.robot_pose()
            pose = None if fix is None or time.monotonic() - fix[0] > 0.5 else fix[1]
            scen, note = perc.to_scenario(pose if pose is not None else np.zeros(3))
            objs, _ = perc.objects()
            n += 1
            fps = n / max(time.monotonic() - t_fps, 1e-6)
            text = (f"robot {'x %+.2f y %+.2f yaw %+.0f deg' % (pose[0], pose[1], np.degrees(pose[2])) if pose is not None else 'tag NOT seen'}"
                    f" | {len(objs)} obstacles | target: {'%s %d' % tuple(scen['target']) if scen else note} | {fps:.0f} fps")
            img = annotate.draw(frame, cam, half, tags, pose, tuple(A["robot"]["footprint"]), scen, A, text)
            for o in objs:
                if "box" in o:
                    b = np.array(o["box"]).astype(int)
                    cv2.rectangle(img, tuple(b[:2]), tuple(b[2:]), (255, 128, 0), 2)
                    cv2.putText(img, f"{o['kind']} {o['score']:.2f}", (b[0], b[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (255, 128, 0), 2)
            if perc.error:
                cv2.putText(img, perc.error[:100], (10, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow("overhead camera", cv2.resize(bgr, None, fx=args.scale, fy=args.scale))
            k = cv2.waitKey(30) & 0xFF
            if k == ord("q"):
                break
            if k == ord("s"):
                name = f"preview_{time.strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(name, bgr)
                print("saved", name)
    finally:
        perc.stop()
        src.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

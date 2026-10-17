"""Phase 1 motion test 3: rotate in place 90 degrees (counterclockwise), then stop.

--wz sets the yaw rate (default 0.3 rad/s). The safety layer still clamps it to max_wz
(0.5 rad/s); a value above that is refused here so the timing stays right. Added after the
first robot run at 0.3 rad/s turned only 13 degrees: the robot stepped but barely rotated,
which suggests a minimum yaw rate.
"""
import argparse
import math
import sys

from common import run
from go2x.config import load_config

ANGLE = math.pi / 2

pre = argparse.ArgumentParser(add_help=False)
pre.add_argument("--wz", type=float, default=0.3)
opts, rest = pre.parse_known_args()
sys.argv = [sys.argv[0]] + rest
W = opts.wz
if not 0 < W <= load_config("safety_limits")["max_wz"]:
    raise SystemExit(f"--wz must be in (0, max_wz]; got {W}")


def steps(robot):
    robot.drive(0.0, 0.0, W, duration=ANGLE / W)


if __name__ == "__main__":
    run("03_rotate", f"rotate in place 90 deg at {W} rad/s, then stop", steps, max_duration_s=20)

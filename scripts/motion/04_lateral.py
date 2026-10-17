"""Phase 1 motion test 4: lateral 0.3 m to the robot's left (+y), then stop.

--vy sets the speed (default 0.15 m/s). The safety layer still clamps it to max_vy
(0.2 m/s); a value above that is refused here so the timing stays right. Added after the
first robot run at 0.15 m/s did not move the robot at all (below its apparent minimum speed).
"""
import argparse
import sys

from common import run
from go2x.config import load_config

D = 0.3

pre = argparse.ArgumentParser(add_help=False)
pre.add_argument("--vy", type=float, default=0.15)
opts, rest = pre.parse_known_args()
sys.argv = [sys.argv[0]] + rest
V = opts.vy
if not 0 < V <= load_config("safety_limits")["max_vy"]:
    raise SystemExit(f"--vy must be in (0, max_vy]; got {V}")


def steps(robot):
    robot.drive(0.0, V, 0.0, duration=D / V)


if __name__ == "__main__":
    run("04_lateral", f"lateral 0.3 m left at {V} m/s, then stop", steps, max_duration_s=20)

"""Phase 1 motion test 3: rotate in place 90 degrees at 0.3 rad/s (counterclockwise), then stop."""
import math

from common import run

W, ANGLE = 0.3, math.pi / 2


def steps(robot):
    robot.drive(0.0, 0.0, W, duration=ANGLE / W)


if __name__ == "__main__":
    run("03_rotate", "rotate in place 90 deg at 0.3 rad/s, then stop", steps, max_duration_s=20)

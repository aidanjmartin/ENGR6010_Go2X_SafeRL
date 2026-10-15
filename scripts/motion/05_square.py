"""Phase 1 motion test 5: a 1 m square at 0.2 m/s.

Four times: forward 1 m at 0.2 m/s, pause, turn 90 degrees counterclockwise at 0.3 rad/s,
pause. Open loop (timed), so the measured square will not close exactly; the summary reports
the odometry displacement at the end, which would be zero for a perfect square.
"""
import math

from common import run

V, SIDE, W = 0.2, 1.0, 0.3


def steps(robot):
    for _ in range(4):
        robot.drive(V, 0.0, 0.0, duration=SIDE / V)
        robot.hold(1.0)
        robot.drive(0.0, 0.0, W, duration=(math.pi / 2) / W)
        robot.hold(1.0)


if __name__ == "__main__":
    run("05_square", "1 m square at 0.2 m/s, 90 deg turns at 0.3 rad/s", steps, max_duration_s=75)

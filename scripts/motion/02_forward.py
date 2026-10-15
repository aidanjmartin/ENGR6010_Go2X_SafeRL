"""Phase 1 motion test 2: forward 0.5 m at 0.2 m/s, then stop.

The rate limit ramps the command up and down symmetrically, so a 0.2 m/s target held for
0.5 / 0.2 = 2.5 s integrates to 0.5 m of commanded travel.
"""
from common import run

V, D = 0.2, 0.5


def steps(robot):
    robot.drive(V, 0.0, 0.0, duration=D / V)


if __name__ == "__main__":
    run("02_forward", "forward 0.5 m at 0.2 m/s, then stop", steps, max_duration_s=20)

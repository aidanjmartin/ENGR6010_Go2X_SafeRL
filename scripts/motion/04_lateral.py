"""Phase 1 motion test 4: lateral 0.3 m at 0.15 m/s (to the robot's left, +y), then stop."""
from common import run

V, D = 0.15, 0.3


def steps(robot):
    robot.drive(0.0, V, 0.0, duration=D / V)


if __name__ == "__main__":
    run("04_lateral", "lateral 0.3 m left at 0.15 m/s, then stop", steps, max_duration_s=20)

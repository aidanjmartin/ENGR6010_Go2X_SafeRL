"""Phase 1 motion test 1: stand up and hold for 10 s. Logs state; sends zero velocity only."""
from common import run


def steps(robot):
    robot.hold(10.0)


if __name__ == "__main__":
    run("01_stand", "stand up and hold 10 s", steps, max_duration_s=25)

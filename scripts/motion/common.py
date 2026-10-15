"""Shared runner for the Phase 1 motion scripts.

Each script calls run(name, description, steps, max_duration_s). It parses --backend,
asks for the safety confirmation on a real backend, drives the robot through SafeRobot
(config/safety_limits.yaml), and prints the log summary.
"""
import argparse
import sys
from pathlib import Path

from go2x.robot.interface import make_backend
from go2x.robot.safety import SafeRobot

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_log import summarize  # noqa: E402

CONFIRM = """
About to run '{name}' on the {backend} backend: {description}
Confirm ALL of the following:
  - 2 m of clear floor around the robot in every direction
  - the handheld remote is in your hand and on
  - the robot is standing or lying normally, not on a stand
Any key or Ctrl+C stops the robot during the run.
Type 'yes' to start: """


def run(name, description, steps, max_duration_s):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--backend", choices=["mock", "dds", "webrtc"], required=True)
    p.add_argument("--skip-stand", action="store_true", help="robot is already in balanced stand")
    args = p.parse_args()

    if args.backend != "mock":
        if input(CONFIRM.format(name=name, backend=args.backend, description=description)).strip() != "yes":
            raise SystemExit("not confirmed; nothing sent")

    backend = make_backend(args.backend)
    robot = SafeRobot(backend, script=name, max_duration_s=max_duration_s,
                      extra_header={"description": description})
    with robot:
        if not args.skip_stand:
            robot.stand_up()
        steps(robot)
        robot.hold(2.0)  # ramp down through the rate limit and let the robot settle
    print("")
    print(f"log: {robot.log.dir}")
    summarize(robot.log.dir)

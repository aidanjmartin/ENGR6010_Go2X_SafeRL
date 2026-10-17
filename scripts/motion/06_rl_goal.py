"""Phase 1 RL demo: the PPO policy trained in Go2NavEnv drives the robot to a goal 1 m ahead.

The policy was trained in simulation with UNFITTED dynamics. On the robot:
- pose comes from the robot's odometry (SportModeState.position) and IMU yaw, relative to
  where the policy starts; the goal is goal_distance ahead along the starting heading;
- the LiDAR inputs are all set to max range ("nothing nearby"): the robot's LiDAR is not
  connected, so THE POLICY CANNOT SEE OBSTACLES and the floor must be clear;
- every policy command goes through SafeRobot (config/safety_limits.yaml): clamped to
  0.2 m/s and 0.5 rad/s (the policy was trained with up to 0.5 m/s, 0.3 m/s, 0.8 rad/s),
  rate limited, watchdog, any-key stop, time cap;
- extra stops: at the goal (sim goal tolerance, 0.3 m) and if odometry puts the robot more
  than --geofence metres from its start.
--velocity-frame says how to read SportModeState.velocity (body or world); it is required on a
real backend because the frame is only known after the scripted rotate-then-forward test.

Usage:
    python scripts/motion/06_rl_goal.py --backend mock --velocity-frame body
    python scripts/motion/06_rl_goal.py --backend webrtc --velocity-frame <body|world> --skip-stand
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from go2x.config import load_config
from go2x.robot.interface import ROOT, make_backend
from go2x.robot.safety import SafeRobot, SafetyStop

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CONFIRM  # noqa: E402
from summarize_log import summarize  # noqa: E402


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--backend", choices=["mock", "dds", "webrtc"], required=True)
    p.add_argument("--velocity-frame", choices=["body", "world"], required=True)
    p.add_argument("--model", default=str(ROOT / "models" / "full_mac.zip"))
    p.add_argument("--goal-distance", type=float, default=1.0)
    p.add_argument("--geofence", type=float, default=1.5)
    p.add_argument("--max-policy-s", type=float, default=20.0)
    p.add_argument("--skip-stand", action="store_true")
    args = p.parse_args()

    sim = load_config("sim")
    sim_limits = np.array([sim["limits"]["vx"], sim["limits"]["vy"], sim["limits"]["wz"]])
    n_beams = sim["lidar"]["n_beams"]
    dist_scale = np.hypot(*([sim["scene"]["room_size_range"][1]] * 2))
    tol = sim["goal_tolerance"]
    model = PPO.load(args.model, device="cpu")

    description = (f"RL policy ({Path(args.model).name}) drives to a goal {args.goal_distance} m ahead; "
                   f"blind to obstacles; stops at goal, at {args.geofence} m from start, or after {args.max_policy_s} s")
    if args.backend != "mock":
        if input(CONFIRM.format(name="06_rl_goal", backend=args.backend, description=description)).strip() != "yes":
            raise SystemExit("not confirmed; nothing sent")

    robot = SafeRobot(make_backend(args.backend), script="06_rl_goal",
                      max_duration_s=args.max_policy_s + 15.0,
                      extra_header={"description": description, "model": Path(args.model).name,
                                    "velocity_frame": args.velocity_frame, "goal_distance": args.goal_distance,
                                    "geofence": args.geofence, "lidar": "all beams at max range (not connected)"})
    outcome = "not started"
    with robot:
        if not args.skip_stand:
            robot.stand_up()
        s = robot.backend.latest_state()
        x0, y0, yaw0 = s.position[0], s.position[1], s.rpy[2]
        goal = np.array([x0 + args.goal_distance * np.cos(yaw0), y0 + args.goal_distance * np.sin(yaw0)])
        robot.log.event("policy_start", start=[x0, y0, yaw0], goal=goal.tolist())
        prev_action = np.zeros(3)
        outcome = "time limit"
        for k in range(int(args.max_policy_s / robot.period)):
            s = robot.backend.latest_state()
            x, y, yaw = s.position[0], s.position[1], s.rpy[2]
            v = np.asarray(s.velocity[:2], float)
            if args.velocity_frame == "world":  # rotate world -> body
                c, sn = np.cos(yaw), np.sin(yaw)
                v = np.array([c * v[0] + sn * v[1], -sn * v[0] + c * v[1]])
            vel = np.array([v[0], v[1], s.yaw_speed])
            dx, dy = goal - (x, y)
            dist, bearing = np.hypot(dx, dy), wrap(np.arctan2(dy, dx) - yaw)
            if dist <= tol:
                outcome = f"goal reached ({dist:.2f} m from goal)"
                break
            if np.hypot(x - x0, y - y0) > args.geofence:
                raise SafetyStop(f"geofence: {np.hypot(x - x0, y - y0):.2f} m from start")
            obs = np.concatenate([[min(dist / dist_scale, 1.0), np.cos(bearing), np.sin(bearing)],
                                  np.clip(vel / sim_limits, -1, 1), prev_action, np.ones(n_beams)]).astype(np.float32)
            action = np.clip(model.predict(obs, deterministic=True)[0], -1, 1)
            cmd = action * sim_limits
            robot.log.event("policy_step", k=k, dist=dist, bearing=bearing, action=action.tolist(), cmd_policy=cmd.tolist())
            robot.drive(*cmd, duration=robot.period)  # one command at 20 Hz, through the safety limits
            prev_action = action
        robot.log.event("policy_end", outcome=outcome)
        robot.hold(2.0)
    print("")
    print(f"policy outcome: {outcome if robot.stop_reason == 'end of script' else 'stopped: ' + str(robot.stop_reason)}")
    print(f"log: {robot.log.dir}")
    summarize(robot.log.dir)


if __name__ == "__main__":
    main()

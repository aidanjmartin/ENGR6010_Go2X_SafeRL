"""L-path repeatability test: forward 0.5 m, turn left 90 degrees, forward 0.5 m.

Closed loop on the robot's own sensors, because the robot under-delivers small commands (LOG.md,
Mac robot session): each forward leg runs at --vx until odometry shows --distance travelled
since the leg started, and the turn runs at --wz until the IMU yaw has changed --angle degrees.
Each segment stops commanding early by the predicted coast (measured speed times --t-resp, plus
the rate-limit ramp-down), and has a timeout. Everything goes through SafeRobot (limits, watchdog, any-key stop,
time cap). The robot's odometry and IMU are both the controller's feedback and the measurement,
so external ground truth (tape measure) is worth recording alongside.

Run once per trial; scripts/motion/summarize_trials.py collects the trials into one table.
Trials chain: each one ends facing 90 degrees left of where it started, so successive trials
trace a square of 1 m sides (the robot stays inside about a 1 m box).

Usage:
    python scripts/motion/07_l_path.py --backend webrtc --skip-stand --trial 1
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

from go2x.config import load_config
from go2x.robot.interface import make_backend
from go2x.robot.safety import SafeRobot

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CONFIRM  # noqa: E402


def pose(robot):
    s = robot.backend.latest_state()
    return np.array([s.position[0], s.position[1]]), s.rpy[2]


def coast(v_meas, v_cmd, accel, t_resp):
    """Predicted travel after commands go to zero: the robot's response time at the measured speed,
    plus the safety layer's ramp-down from the commanded speed."""
    return abs(v_meas) * t_resp + v_cmd ** 2 / (2 * accel)


def forward_leg(robot, vx, distance, timeout_s, name, t_resp):
    p0, _ = pose(robot)
    robot.log.event("segment_start", segment=name, target_m=distance)
    accel = robot.cfg["max_lin_accel"]
    k, travelled, lead = 0, 0.0, 0.0
    while travelled + lead < distance and k * robot.period < timeout_s:
        robot.drive(vx, 0.0, 0.0, duration=robot.period)
        travelled = float(np.hypot(*(pose(robot)[0] - p0)))
        lead = coast(robot.backend.latest_state().velocity[0], vx, accel, t_resp)
        k += 1
    reached = travelled + lead >= distance
    robot.hold(1.0)  # ramp down and settle; the measurement below includes the coast
    final = float(np.hypot(*(pose(robot)[0] - p0)))
    robot.log.event("segment_end", segment=name, reached=reached, at_cutoff_m=travelled, final_m=final,
                    drive_s=k * robot.period)
    return {"reached": reached, "final": final, "drive_s": k * robot.period}


def turn(robot, wz, angle_deg, timeout_s, name, t_resp):
    _, yaw_prev = pose(robot)
    robot.log.event("segment_start", segment=name, target_deg=angle_deg)
    accel = robot.cfg["max_yaw_accel"]
    target = math.radians(angle_deg)
    turned, k, lead = 0.0, 0, 0.0
    while abs(turned) + lead < target and k * robot.period < timeout_s:
        robot.drive(0.0, 0.0, wz, duration=robot.period)
        _, yaw = pose(robot)
        turned += (yaw - yaw_prev + math.pi) % (2 * math.pi) - math.pi  # unwrap
        yaw_prev = yaw
        lead = coast(robot.backend.latest_state().yaw_speed, wz, accel, t_resp)
        k += 1
    reached = abs(turned) + lead >= target
    robot.hold(1.0)
    _, yaw = pose(robot)
    turned += (yaw - yaw_prev + math.pi) % (2 * math.pi) - math.pi
    robot.log.event("segment_end", segment=name, reached=reached, final_deg=math.degrees(turned), drive_s=k * robot.period)
    return {"reached": reached, "final": math.degrees(turned), "drive_s": k * robot.period}


def main():
    lim = load_config("safety_limits")
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--backend", choices=["mock", "dds", "webrtc"], required=True)
    p.add_argument("--trial", type=int, required=True, help="trial number, recorded in the log")
    p.add_argument("--distance", type=float, default=0.5)
    p.add_argument("--angle", type=float, default=90.0)
    p.add_argument("--vx", type=float, default=lim["max_vx"])
    p.add_argument("--wz", type=float, default=lim["max_wz"])
    p.add_argument("--leg-timeout", type=float, default=8.0)
    p.add_argument("--turn-timeout", type=float, default=10.0)
    p.add_argument("--t-resp", type=float, default=0.2,
                   help="robot response time used to stop early, s (measured 0.13 to 0.19 s to 50 %% on 2026-10-17)")
    p.add_argument("--skip-stand", action="store_true")
    args = p.parse_args()
    if not (0 < args.vx <= lim["max_vx"] and 0 < args.wz <= lim["max_wz"]):
        raise SystemExit("--vx and --wz must be positive and within config/safety_limits.yaml")

    description = (f"trial {args.trial}: forward {args.distance} m at {args.vx} m/s, turn left {args.angle} deg "
                   f"at {args.wz} rad/s, forward {args.distance} m (closed loop on odometry and IMU)")
    if args.backend != "mock":
        if input(CONFIRM.format(name="07_l_path", backend=args.backend, description=description)).strip() != "yes":
            raise SystemExit("not confirmed; nothing sent")

    cap = 2 * args.leg_timeout + args.turn_timeout + 3 * 1.0 + 2.0 + (0 if args.skip_stand else 5.0)
    robot = SafeRobot(make_backend(args.backend), script=f"07_l_path_t{args.trial}", max_duration_s=cap,
                      extra_header={"description": description, "trial": args.trial, "args": vars(args)})
    res = {}
    with robot:
        if not args.skip_stand:
            robot.stand_up()
        p0, yaw0 = pose(robot)
        robot.log.event("trial_start", trial=args.trial, start=[*p0.tolist(), yaw0])
        res["leg1"] = forward_leg(robot, args.vx, args.distance, args.leg_timeout, "leg1", args.t_resp)
        res["turn"] = turn(robot, args.wz, args.angle, args.turn_timeout, "turn", args.t_resp)
        res["leg2"] = forward_leg(robot, args.vx, args.distance, args.leg_timeout, "leg2", args.t_resp)
        p1, yaw1 = pose(robot)
        c, s = math.cos(yaw0), math.sin(yaw0)
        d = p1 - p0
        rel = [c * d[0] + s * d[1], -s * d[0] + c * d[1]]  # end position in the start's body frame
        dyaw = math.degrees((yaw1 - yaw0 + math.pi) % (2 * math.pi) - math.pi)
        robot.log.event("trial_end", trial=args.trial, end_rel=rel, dyaw_deg=dyaw)

    print("")
    print(f"trial {args.trial}  stop reason: {robot.stop_reason}")
    for k in ("leg1", "turn", "leg2"):
        if k in res:
            r = res[k]
            unit = "deg" if k == "turn" else "m"
            tgt = args.angle if k == "turn" else args.distance
            print(f"  {k}: {r['final']:.3f} {unit} of {tgt} ({'reached' if r['reached'] else 'TIMEOUT'}), driven {r['drive_s']:.1f} s")
    if "leg2" in res:
        print(f"  end vs start (start's frame, odometry): forward {rel[0]:+.3f} m, left {rel[1]:+.3f} m "
              f"(ideal +{args.distance} / +{args.distance}), heading change {dyaw:+.1f} deg (ideal +{args.angle})")
    print(f"log: {robot.log.dir}")


if __name__ == "__main__":
    main()

"""Run the Task 4 navigation policy (job 050 attempt 4) through a robot backend and record everything.

Backends:
- mujoco: the MuJoCo stand-in (go2x/deploy/mujoco_robot.py), with the Task 1 walking policy in place of
  Unitree's walking controller. --viewer opens the live view, --record writes an mp4.
- mock: the Phase 1 mock (first-order lag, no legs), for a quick dry run.
- webrtc: the Go2 X over Wi-Fi (go2x/deploy/wireless.py). --method sta: robot and laptop on the same Wi-Fi network,
  robot address from --ip or GO2_IP in .env; --method ap: laptop joined to the robot's hotspot.

What runs: every 0.1 s (10 Hz, the training rate) the robot pose comes from odometry and IMU yaw, mapped into the
square frame by anchoring the first reading at the scenario's start pose (the robot must be placed there). The
scenario's obstacles and target are drawn around that pose (the policy's robot view, no camera), the policy picks
a velocity target, and go2x/robot/safety.py (config/safety_limits_nav.yaml) clamps, rate-limits and sends it.
There is no camera on the real robot run: the layout comes from the scenario file and must match the floor.

Stops: success (robot front within 0.5 m of the target at under 0.1 m/s), the footprint touching an obstacle's
footprint (a virtual bumper, the training's collision check), the robot center leaving the square, roll or pitch
above 0.8 rad, the time limit (20 s, the training episode), plus all of SafeRobot's stops (any key, Ctrl+C, stale
state, script time cap). Every stop is zero velocity then StopMove, never Damp.

Recorded in data/<UTC>_nav_policy_<backend>/: header.json, state.jsonl (odometry, 20 Hz), commands.jsonl,
events.jsonl (SafeRobot's logger), lowstate.jsonl (joint angles), run.json (scenario, odometry mapping, outcome)
and policy.npz (every policy step: time, pose, speed, policy input, action, command, distance, people).
scripts/robot/render_twin.py turns a run folder into the digital-twin video.

    python scripts/robot/run_nav_policy.py --backend mujoco --scenario 0 --viewer --record --model_dir <unitree_mujoco>
    python scripts/robot/run_nav_policy.py --backend webrtc --method sta --scenario 12
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from go2x.config import load_config  # noqa: E402
from go2x.deploy.nav_runtime import NavPolicy, OdomToSquare, SquareScene, load_scenario, odom_pose  # noqa: E402
from go2x.robot.logger import scrub  # noqa: E402
from go2x.robot.safety import SafeRobot, SafetyStop  # noqa: E402

CONFIRM = """
About to run the navigation policy on the Go2 X ({method}).
Scenario: {source}. Start pose (square frame): x {sx:+.2f} m, y {sy:+.2f} m, heading {syaw:+.0f} deg.
Confirm ALL of the following:
  - the robot stands at the start pose above, facing the heading above
  - the chairs, table and people are where the setup sheet above puts them, and nothing else is in the square
  - nobody walks into the square during the run
  - the handheld remote is in your hand and on
Any key or Ctrl+C stops the robot.
Type 'yes' to start: """


def setup_sheet(scene):
    """Where everything goes, in the square frame and from the square's (-x, -y) corner."""
    h = scene.cfg["square"]["side"] / 2
    L = scene.batch.layouts[0]
    rows = []

    def row(name, x, y, yaw=None):
        rows.append(f"  {name:<22} x {x:+.2f} m  y {y:+.2f} m   from corner: {x + h:.2f} m along x, {y + h:.2f} m along y"
                    + (f"   heading {math.degrees(yaw):+.0f} deg" if yaw is not None else ""))

    sx, sy, syaw = scene.start_pose
    row("robot start (base)", sx, sy, syaw)
    colors = list(scene.cfg["obstacles"]["chair"]["colors"])
    for i, (x, y, yaw, c) in enumerate(L.chairs):
        row(f"chair {i} ({colors[int(c)]})", x, y, yaw)
    for x, y, yaw in L.table:
        row("table", x, y, yaw)
    for i, p in enumerate(L.people):
        row(f"person {i} ({'walking' if p.walking else 'standing'})", *p.trajectory(0.02, 1)[0])
    kind, idx = L.target
    print("Square: 3.048 m (10 ft) side, origin at its center, x and y along its edges, heading 0 = +x, "
          "positive = counterclockwise seen from above.")
    print("\n".join(rows))
    print(f"  target: {kind} {idx}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--backend", choices=["mujoco", "mock", "webrtc"], required=True)
    p.add_argument("--scenario", type=int, default=0, help="index into config/nav_eval_scenarios.json")
    p.add_argument("--layout", default=None, help="a scenario JSON file instead of --scenario")
    p.add_argument("--actor", default=str(ROOT / "results/phase2/nav_policy/nav_actor.pt"))
    p.add_argument("--limits", default="safety_limits_nav", help="config file name for the safety layer")
    p.add_argument("--max-policy-s", type=float, default=20.0)
    p.add_argument("--skip-stand", action="store_true", help="robot already in balanced stand")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--sheet-only", action="store_true", help="print the setup sheet and exit; connects to nothing")
    # webrtc
    p.add_argument("--method", choices=["sta", "ap"], default="sta")
    p.add_argument("--ip", default=None)
    # mujoco
    p.add_argument("--model_dir", default=None, help="unitree_mujoco checkout (pinned commit, config/mujoco.yaml)")
    p.add_argument("--walk-policy", default=str(ROOT / "results/phase2/walk_train/policy.pt"))
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--record", action="store_true", help="mujoco: write sim_video.mp4 in the run folder")
    args = p.parse_args()

    scenario, source = load_scenario(None if args.layout else args.scenario, args.layout)
    scene = SquareScene(scenario)
    policy = NavPolicy(args.actor)
    limits = load_config(args.limits)
    print(f"{source}; policy {Path(args.actor).name} ({policy.view} view, {policy.meta.get('note', '')})")
    setup_sheet(scene)
    if args.sheet_only:
        return

    real = args.backend == "webrtc"
    if real and scene.walking_people:
        raise SystemExit("This scenario has walking people, which the robot run cannot reproduce. Pick a scenario "
                         "without them (scripts/robot/list_scenarios.py).")
    if args.backend == "mujoco":
        from go2x.deploy.mujoco_robot import MujocoRobot
        if not args.model_dir:
            raise SystemExit("--model_dir is required for the mujoco backend")
        backend = MujocoRobot(args.model_dir, args.walk_policy, scene=scene, viewer=args.viewer, record=None)
    elif args.backend == "mock":
        from go2x.robot.mock import MockRobot
        backend = MockRobot()
    else:
        from go2x.deploy.wireless import WirelessWebRTCRobot
        backend = WirelessWebRTCRobot(method=args.method, ip=args.ip)
    if real:
        sx, sy, syaw = scene.start_pose
        if input(CONFIRM.format(method=backend.info()["method"], source=source, sx=sx, sy=sy,
                                syaw=math.degrees(syaw))).strip() != "yes":
            raise SystemExit("not confirmed; nothing sent")

    robot = SafeRobot(backend, script="nav_policy", max_duration_s=args.max_policy_s + 25.0, config=limits,
                      data_dir=args.data_dir,
                      extra_header={"description": f"navigation policy, {source}", "actor": Path(args.actor).name,
                                    "actor_meta": {k: v for k, v in policy.meta.items() if k != "topdown"}})
    # The log folder exists only once SafeRobot has entered; low state and the recording start there.
    low_file = [None]

    def on_low(msg):
        f = low_file[0]
        if f is not None and not f.closed:
            f.write(json.dumps({"t_mono_ns": time.monotonic_ns(), "t_wall_ns": time.time_ns(),
                                "data": scrub(msg)}, default=float) + "\n")

    if hasattr(backend, "add_low_state_listener"):
        backend.add_low_state_listener(on_low)

    rec = {k: [] for k in ("t", "pose", "odom", "speed", "action", "cmd_policy", "cmd_sent", "distance", "frame", "people")}
    outcome, mapper = "not started", None
    n_people = scene.batch.n_people
    with robot:
        low_file[0] = open(robot.log.dir / "lowstate.jsonl", "w")
        if args.backend == "mujoco" and args.record:
            backend.record_path = str(robot.log.dir / "sim_video.mp4")
            backend.start_recording()
        try:
            if not args.skip_stand:
                robot.stand_up()
            s = backend.latest_state()
            mapper = OdomToSquare(scene.start_pose, odom_pose(s))
            if hasattr(backend, "start_scene_clock"):
                backend.start_scene_clock()
            robot.log.event("policy_start", mapping=mapper.to_dict())
            policy.reset()
            t0 = time.monotonic()
            dt = 1.0 / policy.meta["nav_hz"]
            outcome = "time limit"
            for k in range(int(round(args.max_policy_s / dt))):
                robot.check()
                s = backend.latest_state()
                odom = odom_pose(s)
                pose = mapper(odom)
                speed = float(np.hypot(*s.velocity[:2]))
                ts = backend.scene_time() if hasattr(backend, "scene_time") else time.monotonic() - t0
                scene.set_time(ts if ts is not None else time.monotonic() - t0)
                if s.rpy is not None and max(abs(s.rpy[0]), abs(s.rpy[1])) > 0.8:
                    raise SafetyStop(f"roll or pitch above 0.8 rad: {s.rpy[0]:.2f}, {s.rpy[1]:.2f}")
                chk = scene.checks(pose, speed)
                frame = scene.draw(pose, policy.draw_cfg)
                if chk["at_target"] and not chk["collision_footprint"]:
                    outcome = f"success: front {chk['target_distance']:.2f} m from the target, speed {speed:.2f} m/s"
                    break
                if chk["collision_footprint"]:
                    raise SafetyStop("virtual bumper: the footprint touches an obstacle's footprint")
                if chk["out_of_square"]:
                    raise SafetyStop("the robot center left the square")
                a, cmd = policy.act(frame)
                people = np.full((n_people, 2), np.nan)
                xy = scene.people_xy()
                if xy:
                    people[: len(xy)] = xy
                for key, val in (("t", time.monotonic() - t0), ("pose", pose), ("odom", odom), ("speed", speed),
                                 ("action", a), ("cmd_policy", cmd), ("distance", chk["target_distance"]),
                                 ("frame", frame.numpy().astype(np.float16)), ("people", people)):
                    rec[key].append(val)
                robot.log.event("policy_step", k=k, pose=pose.tolist(), action=a.tolist(), cmd_policy=cmd.tolist(),
                                distance=chk["target_distance"], speed=speed)
                if hasattr(backend, "annotate"):
                    backend.annotate(frame.numpy(), f"policy step {k}  cmd vx {cmd[0]:+.2f} vy {cmd[1]:+.2f} "
                                                    f"wz {cmd[2]:+.2f}  target {chk['target_distance']:.2f} m")
                robot.drive(*cmd, duration=dt)
                rec["cmd_sent"].append(robot.cmd.copy())
            robot.log.event("policy_end", outcome=outcome)
            robot.hold(1.5)
        finally:
            run = {"source": source, "scenario": scenario, "backend": args.backend,
                   "mapping": mapper.to_dict() if mapper else None, "outcome": outcome,
                   "limits": limits, "actor": args.actor, "policy_steps": len(rec["t"])}
            (robot.log.dir / "run.json").write_text(json.dumps(run, indent=1, default=float))
            n = len(rec["t"])
            arrays = {k: np.asarray(v[:n]) for k, v in rec.items() if len(v)}
            np.savez_compressed(robot.log.dir / "policy.npz", **arrays)
            if hasattr(backend, "fell"):
                robot.log.event("mujoco_fell", fell=bool(backend.fell))
    if low_file[0]:
        low_file[0].close()
    final = outcome if robot.stop_reason == "end of script" else f"stopped: {robot.stop_reason}"
    with open(robot.log.dir / "run.json") as f:
        run = json.load(f)
    run["final"] = final
    (robot.log.dir / "run.json").write_text(json.dumps(run, indent=1, default=float))
    print("")
    print(f"outcome: {final}")
    print(f"policy steps: {len(rec['t'])}; log: {robot.log.dir}")
    print(f"digital twin video: python scripts/robot/render_twin.py {robot.log.dir} --model_dir <unitree_mujoco>")


if __name__ == "__main__":
    main()

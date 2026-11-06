"""Read-only check of the wireless link to the Go2 X: connect, record sport state and low state, send nothing.

Sends no motion request of any kind (no stand, no Move, no StopMove), so it is safe with the robot lying down.
Prints message rates, the largest gap between messages, odometry, IMU yaw, the 12 joint angles and the battery
charge, and keeps the messages in data/<UTC>_check_link_webrtc/.

    python scripts/robot/check_link.py --method sta [--ip <robot Wi-Fi address>] --seconds 10
    python scripts/robot/check_link.py --method ap --seconds 10
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from go2x.deploy.wireless import WirelessWebRTCRobot  # noqa: E402
from go2x.robot.logger import SessionLogger, scrub  # noqa: E402
from go2x.sim2sim import SDK_JOINTS  # noqa: E402


def gaps(ts):
    t = np.asarray(ts) / 1e9
    return (len(t) / (t[-1] - t[0]) if len(t) > 1 else 0.0), (np.diff(t).max() if len(t) > 1 else float("nan"))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--method", choices=["sta", "ap"], default="sta")
    p.add_argument("--ip", default=None)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--data-dir", default=None)
    args = p.parse_args()

    robot = WirelessWebRTCRobot(method=args.method, ip=args.ip)
    log = SessionLogger("check_link", robot.info(), {"read_only": True}, args.data_dir,
                        {"description": "read-only wireless link check; no request sent"})
    sport_t, low_t, low_last = [], [], [None]
    low_f = open(log.dir / "lowstate.jsonl", "w")

    def on_sport(s):
        sport_t.append(s.t_mono_ns)
        log.state(s)

    def on_low(msg):
        low_t.append(time.monotonic_ns())
        low_last[0] = msg
        low_f.write(json.dumps({"t_mono_ns": low_t[-1], "t_wall_ns": time.time_ns(), "data": scrub(msg)}, default=float) + "\n")

    robot.add_state_listener(on_sport)
    robot.add_low_state_listener(on_low)
    t0 = time.monotonic()
    try:
        robot.connect()
        log.event("connected", seconds_to_connect=time.monotonic() - t0)
        print(f"connected in {time.monotonic() - t0:.1f} s; recording {args.seconds:.0f} s, sending nothing", flush=True)
        time.sleep(args.seconds)
    finally:
        robot.close()
        low_f.close()
        log.event("closed")
        log.close()

    r, g = gaps(sport_t)
    print(f"sport state: {len(sport_t)} messages, {r:.1f} per second, largest gap {g * 1000:.0f} ms")
    r, g = gaps(low_t)
    print(f"low state:   {len(low_t)} messages, {r:.1f} per second, largest gap {g * 1000:.0f} ms")
    s = robot.latest_state()
    if s is not None:
        print(f"odometry x {s.position[0]:+.3f} y {s.position[1]:+.3f} m, IMU yaw {s.rpy[2]:+.3f} rad, "
              f"body height {s.body_height:.3f} m, mode {s.mode}")
    m = low_last[0]
    if m:
        q = [x.get("q") for x in m.get("motor_state", [])[:12]]
        print("joint angles (rad, SDK order): " + ", ".join(f"{n.replace('_joint', '')} {v:+.2f}" for n, v in zip(SDK_JOINTS, q)))
        bms = m.get("bms_state") or {}
        if "soc" in bms:
            print(f"battery charge {bms['soc']} %")
    ok = len(sport_t) > 0 and len(low_t) > 0
    print("LINK OK: both streams arrived" if ok else "LINK PROBLEM: a stream is missing (see above)")
    print(f"log: {log.dir}")


if __name__ == "__main__":
    main()

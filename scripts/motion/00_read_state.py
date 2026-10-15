"""Read-only check: connect with a backend, log sport-mode state for a few seconds, send nothing.

Run this before any motion script on a new backend. The safety layer's 0.5 s watchdog depends
on this stream, and Exam 1 never read rt/lf/sportmodestate over DDS (only rt/lf/lowstate).
No velocity, stand or stop request is sent (the DDS backend constructs its SportClient but
calls no API on it; the WebRTC backend sends one read-only motion-switcher status query).

Usage:
    python scripts/motion/00_read_state.py --backend dds --seconds 10
"""
import argparse
import sys
import time
from pathlib import Path

from go2x.config import load_config
from go2x.robot.interface import make_backend
from go2x.robot.logger import SessionLogger

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_log import load, state_arrays  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--backend", choices=["mock", "dds", "webrtc"], required=True)
    p.add_argument("--seconds", type=float, default=10.0)
    args = p.parse_args()

    backend = make_backend(args.backend)
    log = SessionLogger("00_read_state", backend.info(), load_config("safety_limits"),
                        extra_header={"read_only": True})
    backend.add_state_listener(log.state)
    try:
        backend.connect()
        log.event("connected")
        time.sleep(args.seconds)
    finally:
        backend.close()
        log.event("closed", backend_info=backend.info())
        log.close()

    _, _, states, _ = load(log.dir)
    print(f"log: {log.dir}")
    if not states:
        print("NO STATE RECEIVED. Do not run motion scripts on this backend.")
        return
    t, v, _, wz, yaw, pos = state_arrays(states)
    span = t[-1] - t[0] if len(t) > 1 else float("nan")
    gaps = (t[1:] - t[:-1]) if len(t) > 1 else [float("nan")]
    last = states[-1]["data"]
    print(f"{len(t)} messages in {span:.1f} s: {len(t) / span:.1f} per second, largest gap {max(gaps) * 1000:.0f} ms")
    print(f"last: mode {last.get('mode')}, body_height {last.get('body_height')}, velocity {v[-1].round(3).tolist()}, "
          f"yaw_speed {wz[-1]:.3f}, IMU yaw {yaw[-1]:.3f}, position {pos[-1].round(3).tolist()}")
    if max(gaps) > load_config("safety_limits")["state_timeout_s"]:
        print("WARNING: a gap exceeded the 0.5 s state timeout; the watchdog would have stopped a motion script.")


if __name__ == "__main__":
    main()

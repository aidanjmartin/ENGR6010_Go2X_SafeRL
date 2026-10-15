"""Session logger in the format of ENGR6010_Exam1 q4_data_sim/data_collection_plan.md (commit 90150da).

That plan's format, kept here:
- each stream in JSON Lines, one object per message;
- every message stamped on arrival with time.monotonic_ns() (ordering, intervals) and
  time.time_ns() (alignment with other machines), keeping the robot's own time stamp;
- the serial number field `sn` dropped before writing;
- a session header file with start time, library versions and topic list.

A session is a folder data/<UTC time>_<script>_<backend>/ holding header.json,
state.jsonl, commands.jsonl and events.jsonl. data/ is gitignored.
"""
import json
import platform
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from go2x.robot.interface import ROOT

DROP_KEYS = {"sn"}


def scrub(obj):
    """Copy obj without any key in DROP_KEYS, at any depth."""
    if isinstance(obj, dict):
        return {k: scrub(v) for k, v in obj.items() if k not in DROP_KEYS}
    if isinstance(obj, (list, tuple)):
        return [scrub(v) for v in obj]
    return obj


class SessionLogger:
    def __init__(self, script, backend_info, config, data_dir=None, extra_header=None):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        base = Path(data_dir) if data_dir else ROOT / "data"
        self.dir = base / f"{stamp}_{script}_{backend_info['backend']}"
        self.dir.mkdir(parents=True, exist_ok=False)
        self._lock = threading.Lock()
        self._files = {n: open(self.dir / f"{n}.jsonl", "w") for n in ("state", "commands", "events")}
        header = {
            "script": script,
            "start_wall_ns": time.time_ns(),
            "start_mono_ns": time.monotonic_ns(),
            "start_utc": stamp,
            "host": platform.node(),
            "python": platform.python_version(),
            "backend": scrub(backend_info),
            "safety_config": config,
            "streams": {"state.jsonl": "sport-mode state, one object per message",
                        "commands.jsonl": "every command sent: requested (before limits) and sent (after)",
                        "events.jsonl": "stand, stop and abort events with reasons"},
            "firmware": "unknown (Exam 1 handshake data2 = 2: 1.1.11 to 1.1.14)",
            **(extra_header or {}),
        }
        (self.dir / "header.json").write_text(json.dumps(scrub(header), indent=2))

    def _write(self, stream, record):
        line = json.dumps(scrub(record), default=float)
        with self._lock:
            f = self._files.get(stream)
            if f and not f.closed:
                f.write(line + "\n")

    def state(self, s):
        self._write("state", {"t_mono_ns": s.t_mono_ns, "t_wall_ns": s.t_wall_ns, "topic": s.topic,
                              "robot_stamp": s.robot_stamp, "data": s.raw})

    def command(self, requested, sent):
        self._write("commands", {"t_mono_ns": time.monotonic_ns(), "t_wall_ns": time.time_ns(),
                                 "requested": list(requested), "sent": list(sent)})

    def event(self, kind, **detail):
        self._write("events", {"t_mono_ns": time.monotonic_ns(), "t_wall_ns": time.time_ns(),
                               "event": kind, **detail})

    def close(self):
        with self._lock:
            for f in self._files.values():
                f.close()

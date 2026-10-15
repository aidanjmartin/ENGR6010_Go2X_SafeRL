"""One interface for Go2 body-velocity commands and sport-mode state.

Backends: MockRobot (mock.py), DDSRobot (dds.py), WebRTCRobot (webrtc.py). Never use a
backend directly in a motion script: wrap it in safety.SafeRobot.

read_env() is adapted from ENGR6010_Exam1 q3_apps/webrtc_read_state.py (commit 90150da):
same parsing, path changed to this repository's root.
"""
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class State:
    """One sport-mode state message, stamped on arrival.

    `raw` is the message as a dict, as the backend received it (the logger drops `sn`).
    The named fields are copied from it for convenience. The frame of `velocity` on the real
    robot is not yet confirmed (body or world); see LOG.md.
    """
    t_mono_ns: int
    t_wall_ns: int
    topic: str
    raw: dict
    robot_stamp: dict = field(default_factory=dict)   # {"sec", "nanosec"} if the robot sends it
    mode: int = None
    position: list = None       # [x, y, z], odometry
    velocity: list = None       # [vx, vy, vz]
    yaw_speed: float = None
    rpy: list = None            # IMU roll, pitch, yaw
    body_height: float = None


def state_from_dict(raw, topic):
    """Build a State from a SportModeState-shaped dict (the same field names over DDS and WebRTC)."""
    t_mono, t_wall = time.monotonic_ns(), time.time_ns()
    imu = raw.get("imu_state") or {}
    return State(t_mono, t_wall, topic, raw,
                 robot_stamp=raw.get("stamp") or {}, mode=raw.get("mode"),
                 position=raw.get("position"), velocity=raw.get("velocity"),
                 yaw_speed=raw.get("yaw_speed"), rpy=imu.get("rpy"), body_height=raw.get("body_height"))


class RobotInterface(ABC):
    """Velocity commands and state. Subclasses call self._on_state(state) for every message."""

    name = "base"

    def __init__(self):
        self._lock = threading.Lock()
        self._latest = None
        self._listeners = []

    # -- state ---------------------------------------------------------------------
    def add_state_listener(self, fn):
        self._listeners.append(fn)

    def latest_state(self):
        with self._lock:
            return self._latest

    def _on_state(self, state):
        with self._lock:
            self._latest = state
        for fn in self._listeners:
            fn(state)

    # -- commands ------------------------------------------------------------------
    @abstractmethod
    def connect(self):
        """Open the link and start receiving state."""

    @abstractmethod
    def close(self):
        """Close the link. Does not command the robot."""

    @abstractmethod
    def send_velocity(self, vx, vy, wz):
        """Send one body-velocity command (m/s, m/s, rad/s). Non-blocking."""

    @abstractmethod
    def stand_up(self):
        """Bring the robot to a balanced stand that accepts velocity commands."""

    @abstractmethod
    def stop_move(self):
        """Ask the robot to stop moving and hold its stance (sport StopMove). Never Damp."""

    def info(self):
        """Backend details for the session header."""
        return {"backend": self.name}


def read_env():
    """Read the untracked .env file at the repository root into a dict."""
    env = {}
    try:
        with open(ROOT / ".env") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    env[key.strip()] = value.strip()
    except OSError:
        pass
    return env


def make_backend(name, **kwargs):
    """Construct a backend by name: mock, dds or webrtc."""
    if name == "mock":
        from go2x.robot.mock import MockRobot
        return MockRobot(**kwargs)
    if name == "dds":
        from go2x.robot.dds import DDSRobot
        return DDSRobot(**kwargs)
    if name == "webrtc":
        from go2x.robot.webrtc import WebRTCRobot
        return WebRTCRobot(**kwargs)
    raise ValueError(f"unknown backend {name!r}")

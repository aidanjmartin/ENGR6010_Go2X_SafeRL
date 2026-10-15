"""Mock Go2: emits sport-mode-shaped state at 20 Hz from a first-order lag plus dead time.

Everything in Phase 1 is tested against this before it touches the robot. The dynamics are
go2x.sim.dynamics.LagDelayDynamics with fixed, UNFITTED parameters (assumptions, not Go2 X
measurements). Like the real robot (see webrtc.py), Move is ignored until stand_up() has run,
and the last command is held until a new one arrives.
"""
import threading
import time

import numpy as np

from go2x.robot.interface import RobotInterface, state_from_dict
from go2x.sim.dynamics import LagDelayDynamics

TOPIC = "mock/sportmodestate"


class MockRobot(RobotInterface):
    name = "mock"

    def __init__(self, rate_hz=20.0, tau=0.25, delay=0.1, drop_state_after_s=None, seed=0):
        """drop_state_after_s: stop publishing state after this many seconds (watchdog tests)."""
        super().__init__()
        self.rate_hz = rate_hz
        self.dt = 1.0 / rate_hz
        self.dyn = LagDelayDynamics(substeps=10, dt=self.dt)
        self.dyn.reset(np.random.default_rng(seed), [0, 0, 0], tau=[tau] * 3, delay=[delay] * 3)
        self.tau, self.delay = tau, delay
        self.drop_state_after_s = drop_state_after_s
        self.standing = False
        self.cmd = np.zeros(3)
        self.calls = []                 # every command received, for tests
        self._run = threading.Event()
        self._thread = None

    def connect(self):
        self._t0 = time.monotonic()
        self._run.set()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def close(self):
        self._run.clear()
        if self._thread:
            self._thread.join(timeout=1.0)

    def send_velocity(self, vx, vy, wz):
        self.calls.append(("move", time.monotonic(), (vx, vy, wz)))
        if self.standing:
            self.cmd = np.array([vx, vy, wz], dtype=float)

    def stand_up(self):
        self.calls.append(("stand_up", time.monotonic(), None))
        self.standing = True

    def stop_move(self):
        self.calls.append(("stop_move", time.monotonic(), None))
        self.cmd = np.zeros(3)

    def info(self):
        return {"backend": self.name, "rate_hz": self.rate_hz,
                "dynamics": f"LagDelayDynamics tau={self.tau} s, delay={self.delay} s (unfitted assumption)"}

    def _loop(self):
        next_t = time.monotonic()
        while self._run.is_set():
            self.dyn.step(self.cmd, self.dt)
            elapsed = time.monotonic() - self._t0
            if self.drop_state_after_s is None or elapsed < self.drop_state_after_s:
                x, y, yaw = self.dyn.pose
                now = time.time_ns()
                raw = {
                    "stamp": {"sec": now // 1_000_000_000, "nanosec": now % 1_000_000_000},
                    "mode": 1 if self.standing else 0,
                    "position": [float(x), float(y), 0.0],
                    "velocity": [float(self.dyn.vel[0]), float(self.dyn.vel[1]), 0.0],  # body frame in the mock
                    "yaw_speed": float(self.dyn.vel[2]),
                    "imu_state": {"rpy": [0.0, 0.0, float(yaw)]},
                    "body_height": 0.32 if self.standing else 0.08,
                    "sn": "MOCK-SERIAL",  # stands in for LowState's serial number; the logger must drop it
                }
                self._on_state(state_from_dict(raw, TOPIC))
            next_t += self.dt
            time.sleep(max(0.0, next_t - time.monotonic()))

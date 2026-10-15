"""Safety layer that every motion script drives the robot through.

    with SafeRobot(make_backend("mock"), script="forward") as robot:
        robot.stand_up()
        robot.drive(0.2, 0, 0, duration=2.5)

What it enforces (limits from config/safety_limits.yaml only):
- clamps every command to max_vx, max_vy, max_wz;
- limits the command's rate of change (max_lin_accel, max_yaw_accel);
- sends commands at command_hz;
- zero velocity and stop if no state arrives for state_timeout_s, checked both in the
  command loop and by an independent watchdog thread;
- zero velocity and stop on Ctrl+C, on any key press (when stdin is a terminal), on any
  exception, and when the script's total time reaches max_script_s;
- "stop" means: stop_repeats zero-velocity commands, then the backend's StopMove. It never
  sends Damp, which drops the robot.
SafetyStop and KeyboardInterrupt end the `with` block quietly so the script can still print
its summary; any other exception is re-raised after the stop.
"""
import os
import select
import sys
import threading
import time

import numpy as np

from go2x.config import load_config
from go2x.robot.logger import SessionLogger


class SafetyStop(Exception):
    pass


class _KeyListener:
    """Sets an event on any key press. Puts the terminal in cbreak mode (Ctrl+C still works)."""

    def __init__(self, event):
        self.event = event
        self._old = None
        self._run = threading.Event()

    def start(self):
        if not sys.stdin.isatty():
            return False
        import termios
        import tty
        fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        self._run.set()
        threading.Thread(target=self._loop, args=(fd,), daemon=True).start()
        return True

    def _loop(self, fd):
        while self._run.is_set():
            r, _, _ = select.select([fd], [], [], 0.05)
            if r:
                os.read(fd, 1)
                self.event.set()
                return

    def stop(self):
        self._run.clear()
        if self._old is not None:
            import termios
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old)
            self._old = None


class SafeRobot:
    def __init__(self, backend, script, max_duration_s=None, listen_keys=True, data_dir=None,
                 extra_header=None, config=None):
        self.backend = backend
        self.cfg = config or load_config("safety_limits")
        c = self.cfg
        self.max_cmd = np.array([c["max_vx"], c["max_vy"], c["max_wz"]], float)
        self.max_rate = np.array([c["max_lin_accel"], c["max_lin_accel"], c["max_yaw_accel"]], float)
        self.period = 1.0 / c["command_hz"]
        self.max_duration_s = min(max_duration_s or c["max_script_s"], c["max_script_s"])
        self.script = script
        self.listen_keys = listen_keys
        self.data_dir = data_dir
        self.extra_header = extra_header
        self.cmd = np.zeros(3)
        self.stop_reason = None
        self._abort = threading.Event()
        self._key = threading.Event()
        self._send_lock = threading.Lock()
        self._stopped = False
        self.log = None

    # -- context --------------------------------------------------------------------
    def __enter__(self):
        self.t_start = time.monotonic()
        self.log = SessionLogger(self.script, self.backend.info(), self.cfg, self.data_dir,
                                 {"max_duration_s": self.max_duration_s, **(self.extra_header or {})})
        self.backend.add_state_listener(self.log.state)
        self._keys = _KeyListener(self._key)
        if self.listen_keys and self._keys.start():
            print("Press any key (or Ctrl+C) to stop the robot.", flush=True)
        self._watch_run = threading.Event()
        self._watch_run.set()
        try:
            self.backend.connect()
            self.log.event("connected")
            self._wait_first_state()
        except BaseException:
            self._teardown()
            raise
        threading.Thread(target=self._watchdog, daemon=True).start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            reason = "end of script"
        elif issubclass(exc_type, SafetyStop):
            reason = str(exc)
        elif issubclass(exc_type, KeyboardInterrupt):
            reason = "Ctrl+C"
        else:
            reason = f"exception: {exc_type.__name__}: {exc}"
        self.stop(reason)
        self._teardown()
        if exc_type is not None:
            print(f"STOPPED: {reason}", flush=True)
        return exc_type is not None and issubclass(exc_type, (SafetyStop, KeyboardInterrupt))

    def _teardown(self):
        self._watch_run.clear()
        self._keys.stop()
        try:
            self.backend.close()
        finally:
            if self.log:
                self.log.event("closed")
                self.log.close()

    # -- checks ---------------------------------------------------------------------
    def state_age_s(self):
        s = self.backend.latest_state()
        return None if s is None else (time.monotonic_ns() - s.t_mono_ns) / 1e9

    def _wait_first_state(self):
        t_end = time.monotonic() + self.cfg["first_state_timeout_s"]
        while self.backend.latest_state() is None:
            if time.monotonic() > t_end:
                raise SafetyStop(f"no state within {self.cfg['first_state_timeout_s']} s of connecting")
            time.sleep(0.01)
        self.log.event("first_state")

    def check(self):
        """Raise SafetyStop if any stop condition holds."""
        if self._abort.is_set():
            raise SafetyStop(self.stop_reason or "aborted")
        if self._key.is_set():
            raise SafetyStop("key pressed")
        age = self.state_age_s()
        if age is None or age > self.cfg["state_timeout_s"]:
            raise SafetyStop(f"state timeout: last state {age if age is None else round(age, 3)} s ago")
        if time.monotonic() - self.t_start > self.max_duration_s:
            raise SafetyStop(f"script time cap {self.max_duration_s} s reached")

    def _watchdog(self):
        """Independent of the command loop: zero velocity as soon as state goes stale."""
        while self._watch_run.is_set():
            age = self.state_age_s()
            if age is not None and age > self.cfg["state_timeout_s"] and not self._abort.is_set():
                self.stop_reason = f"state timeout (watchdog): last state {age:.3f} s ago"
                self._abort.set()
                with self._send_lock:
                    self.backend.send_velocity(0.0, 0.0, 0.0)
                    self.cmd = np.zeros(3)
                self.log.event("watchdog_zero", state_age_s=age)
            time.sleep(0.02)

    # -- commands -------------------------------------------------------------------
    def limit(self, target, dt):
        """Clamp target to the limits, then move self.cmd toward it at the allowed rate."""
        target = np.clip(np.asarray(target, float), -self.max_cmd, self.max_cmd)
        step = np.clip(target - self.cmd, -self.max_rate * dt, self.max_rate * dt)
        return self.cmd + step

    def _send(self, requested, dt):
        with self._send_lock:
            if self._abort.is_set():
                raise SafetyStop(self.stop_reason or "aborted")
            self.cmd = self.limit(requested, dt)
            self.backend.send_velocity(*self.cmd)
            self.log.command(requested, self.cmd)

    def drive(self, vx, vy, wz, duration):
        """Command (vx, vy, wz) for `duration` seconds through the limits, at command_hz."""
        t_end = time.monotonic() + duration
        next_t = time.monotonic()
        while time.monotonic() < t_end:
            self.check()
            self._send((vx, vy, wz), self.period)
            next_t += self.period
            time.sleep(max(0.0, next_t - time.monotonic()))

    def hold(self, duration):
        """Command zero velocity (ramped down through the rate limit) for `duration` seconds."""
        self.drive(0.0, 0.0, 0.0, duration)

    def stand_up(self, settle_s=3.0):
        self.check()
        self.log.event("stand_up_request")
        self.backend.stand_up()
        self.log.event("stand_up_done")
        self.hold(settle_s)

    def stop(self, reason):
        """Zero velocity, then StopMove. Safe to call more than once."""
        if self._stopped:
            return
        self._stopped = True
        self.stop_reason = self.stop_reason or reason
        self.log.event("stop", reason=reason)
        for _ in range(self.cfg["stop_repeats"]):
            try:
                with self._send_lock:
                    self.backend.send_velocity(0.0, 0.0, 0.0)
                    self.cmd = np.zeros(3)
                    self.log.command((0.0, 0.0, 0.0), self.cmd)
            except Exception as e:  # keep trying StopMove even if a send fails
                self.log.event("stop_send_error", error=repr(e))
            time.sleep(self.period)
        try:
            self.backend.stop_move()
            self.log.event("stop_move_sent")
        except Exception as e:
            self.log.event("stop_move_error", error=repr(e))
            print(f"WARNING: StopMove failed: {e!r}. Use the handheld remote.", flush=True)

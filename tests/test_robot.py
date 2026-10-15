import json
import time

import numpy as np
import pytest

from go2x.config import load_config
from go2x.robot.mock import MockRobot
from go2x.robot.safety import SafeRobot

CFG = load_config("safety_limits")


def safe(tmp_path, **kw):
    mock_kw = {k: kw.pop(k) for k in ("drop_state_after_s",) if k in kw}
    return SafeRobot(MockRobot(**mock_kw), script="test", listen_keys=False, data_dir=tmp_path, **kw)


def read_jsonl(path):
    return [json.loads(line) for line in open(path)]


def test_clamp_and_rate_limit(tmp_path):
    with safe(tmp_path) as r:
        r.stand_up(settle_s=0.1)
        r.drive(1.0, -1.0, 2.0, duration=1.0)
        logdir = r.log.dir
    sent = np.array([c["sent"] for c in read_jsonl(logdir / "commands.jsonl")])
    assert np.all(np.abs(sent[:, 0]) <= CFG["max_vx"] + 1e-12)
    assert np.all(np.abs(sent[:, 1]) <= CFG["max_vy"] + 1e-12)
    assert np.all(np.abs(sent[:, 2]) <= CFG["max_wz"] + 1e-12)
    dt = 1.0 / CFG["command_hz"]
    # stop() zeroes immediately by design; the rate limit applies to the commands before it
    steps = np.abs(np.diff(sent[:-CFG["stop_repeats"]], axis=0))
    assert np.all(steps[:, :2] <= CFG["max_lin_accel"] * dt + 1e-12)
    assert steps[:, 2].max() <= CFG["max_yaw_accel"] * dt + 1e-12
    assert np.isclose(sent[:, 0].max(), CFG["max_vx"])  # it does reach the limit


def test_stop_sends_zero_then_stop_move(tmp_path):
    with safe(tmp_path) as r:
        r.stand_up(settle_s=0.1)
        r.drive(0.2, 0, 0, duration=0.5)
        mock = r.backend
    kinds = [c[0] for c in mock.calls]
    assert kinds[-1] == "stop_move"
    tail = mock.calls[-1 - CFG["stop_repeats"]:-1]
    assert all(c[0] == "move" and c[2] == (0.0, 0.0, 0.0) for c in tail)


def test_state_timeout_triggers_zero_within_limit(tmp_path):
    with safe(tmp_path, drop_state_after_s=1.0) as r:
        r.stand_up(settle_s=0.1)
        r.drive(0.2, 0, 0, duration=5.0)
    assert "state timeout" in r.stop_reason
    mock = r.backend
    last_nonzero = max(c[1] for c in mock.calls if c[0] == "move" and any(c[2]))
    first_zero_after = min(c[1] for c in mock.calls if c[0] == "move" and not any(c[2]) and c[1] > last_nonzero)
    # state stops at 1.0 s after connect; zero must follow within the timeout plus a few command periods
    assert first_zero_after - mock._t0 < 1.0 + CFG["state_timeout_s"] + 0.15
    assert mock.calls[-1][0] == "stop_move"


def test_exception_stops_and_reraises(tmp_path):
    with pytest.raises(ValueError):
        with safe(tmp_path) as r:
            r.stand_up(settle_s=0.1)
            r.drive(0.2, 0, 0, duration=0.3)
            raise ValueError("boom")
    assert r.backend.calls[-1][0] == "stop_move"
    assert r.stop_reason.startswith("exception: ValueError")


def test_ctrl_c_stops_quietly(tmp_path):
    with safe(tmp_path) as r:
        r.stand_up(settle_s=0.1)
        raise KeyboardInterrupt
    assert r.stop_reason == "Ctrl+C"
    assert r.backend.calls[-1][0] == "stop_move"


def test_key_press_stops(tmp_path):
    with safe(tmp_path) as r:
        r.stand_up(settle_s=0.1)
        r._key.set()  # what the key listener does on a key press
        r.drive(0.2, 0, 0, duration=2.0)
    assert r.stop_reason == "key pressed"


def test_duration_cap(tmp_path):
    t0 = time.monotonic()
    with safe(tmp_path, max_duration_s=1.0) as r:
        r.drive(0.1, 0, 0, duration=10.0)
    assert "time cap" in r.stop_reason
    assert time.monotonic() - t0 < 2.0


def test_logger_format_and_sn_dropped(tmp_path):
    with safe(tmp_path) as r:
        r.stand_up(settle_s=0.5)
        logdir = r.log.dir
    header = json.loads((logdir / "header.json").read_text())
    assert header["backend"]["backend"] == "mock" and "safety_config" in header
    states = read_jsonl(logdir / "state.jsonl")
    assert len(states) >= 5
    for s in states:
        assert {"t_mono_ns", "t_wall_ns", "robot_stamp", "data"} <= s.keys()
        assert "sec" in s["robot_stamp"]
    text = (logdir / "state.jsonl").read_text()
    assert '"sn"' not in text and "MOCK-SERIAL" not in text


def test_mock_forward_distance(tmp_path):
    with safe(tmp_path) as r:
        r.stand_up(settle_s=0.1)
        r.drive(0.2, 0, 0, duration=2.5)
        r.hold(1.5)
        x = r.backend.latest_state().position[0]
    assert abs(x - 0.5) < 0.03

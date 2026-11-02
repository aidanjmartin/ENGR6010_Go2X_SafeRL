import numpy as np
import pytest

from go2x.transitions import STATE_FIELDS, TransitionLogger


def _run(T=40, B=3, reset_at=None):
    log = TransitionLogger(0.02, [0.1, 0.5], B)
    for t in range(T):
        state = np.zeros((B, len(STATE_FIELDS)))
        state[:, 0] = t          # x carries the step index
        state[:, 1] = np.arange(B)
        target = np.ones((B, 3)) * (t // 20)
        new = np.zeros(B, bool)
        if reset_at is not None and t == reset_at:
            new[0] = True
        log.record(state, target, new)
    return log


def test_offsets_and_pairs():
    d = _run().build()
    # 0.1 s = 5 steps, 0.5 s = 25 steps; 40 steps leave 15 start times per env
    assert d["state"].shape == (15 * 3, len(STATE_FIELDS))
    assert np.all(d["state_next_0.1s"][:, 0] - d["state"][:, 0] == 5)
    assert np.all(d["state_next_0.5s"][:, 0] - d["state"][:, 0] == 25)
    assert np.all(d["state"][:, 1] == d["env"])


def test_reset_windows_are_dropped():
    d = _run(reset_at=10).build()
    env0 = d["env"] == 0
    # env 0 restarts at step 10: no window of env 0 may contain both step 9 and step 10
    starts = d["state"][env0, 0]
    assert not np.any((starts < 10) & (starts + 25 >= 10))
    assert np.sum(~env0) == 30


def test_target_constant_flag():
    d = _run().build()
    # target changes at step 20; windows starting at 0..14 reach steps up to 39
    t0 = d["state"][:, 0]
    assert np.all(d["target_constant"] == (t0 + 25 < 20))


def test_bad_horizon():
    with pytest.raises(ValueError):
        TransitionLogger(0.02, [0.11], 2)


def test_save_roundtrip(tmp_path):
    log = _run()
    shapes = log.save(tmp_path / "t.npz", policy="test")
    z = np.load(tmp_path / "t.npz")
    assert z["state"].shape == shapes["state"]
    assert list(z["state_fields"]) == list(STATE_FIELDS)
    assert str(z["meta_policy"]) == "test"

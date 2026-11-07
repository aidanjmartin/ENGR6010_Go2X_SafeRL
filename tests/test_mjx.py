"""MuJoCo training code on the CPU: the training model, position-actuator PD, the TorchScript export, the MJX
environment (if JAX and Playground are installed), and the MuJoCo navigation environment."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
torch = pytest.importorskip("torch")

from go2x.config import CONFIG_DIR, load_config  # noqa: E402
from go2x.sim2sim import DEFAULT_POS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = os.environ.get("GO2X_UNITREE_MUJOCO", str(ROOT / "third_party/unitree_mujoco"))
needs_model = pytest.mark.skipif(not os.path.exists(os.path.join(MODEL_DIR, "unitree_robots/go2/go2.xml")),
                                 reason="unitree_mujoco not found (scripts/robot/setup_laptop.sh, or GO2X_UNITREE_MUJOCO)")


def _hold(m, d, ix, steps, python_pd):
    for _ in range(steps):
        if python_pd:
            d.ctrl[ix["act"]] = np.clip(25 * (DEFAULT_POS - d.qpos[ix["qadr"]]) - 0.5 * d.qvel[ix["vadr"]], -ix["limit"], ix["limit"])
        mujoco.mj_step(m, d)


@needs_model
def test_training_model_contacts_and_standing():
    from go2x.mjx import go2_model as G
    m = G.training_model(MODEL_DIR)
    colliding = {m.geom(i).name for i in range(m.ngeom) if m.geom_contype[i] or m.geom_conaffinity[i]}
    assert colliding == {"floor", "FL", "FR", "RL", "RR", "FL_knee", "FR_knee", "RL_knee", "RR_knee"}
    assert m.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL and m.opt.iterations == 4
    full = G.eval_model(MODEL_DIR)
    assert full.nbody == m.nbody and np.allclose(full.body_mass, m.body_mass)     # same robot
    ix = G.indices(m)
    d = mujoco.MjData(m)
    d.qpos[:] = G.home_qpos(m, G.standing_height(m))
    mujoco.mj_forward(m, d)
    x0 = d.qpos[:2].copy()
    _hold(m, d, ix, 800, True)                       # 4 s of PD on the default pose: stands, does not drift
    assert d.qpos[2] > 0.2 and np.linalg.norm(d.qpos[:2] - x0) < 0.05 and np.linalg.norm(d.qvel[:2]) < 0.02


@needs_model
def test_position_actuators_equal_python_pd():
    from go2x.mjx import go2_model as G
    m1, m2 = G.eval_model(MODEL_DIR), G.eval_model(MODEL_DIR)
    ix = G.indices(m1)
    G.use_position_actuators(m2, 25.0, 0.5, ix["act"])
    d1, d2 = mujoco.MjData(m1), mujoco.MjData(m2)
    for m, d in ((m1, d1), (m2, d2)):
        d.qpos[:] = G.home_qpos(m, G.standing_height(m))
        mujoco.mj_forward(m, d)
    for t in range(100):
        tgt = DEFAULT_POS + 0.3 * np.sin(0.05 * t + np.arange(12))
        for _ in range(4):
            d1.ctrl[ix["act"]] = np.clip(25 * (tgt - d1.qpos[ix["qadr"]]) - 0.5 * d1.qvel[ix["vadr"]], -ix["limit"], ix["limit"])
            mujoco.mj_step(m1, d1)
        d2.ctrl[ix["act"]] = tgt
        for _ in range(4):
            mujoco.mj_step(m2, d2)
    assert np.abs(d1.qpos - d2.qpos).max() < 1e-8


def test_export_matches_numpy_forward(tmp_path):
    from go2x.mjx.export import export_torchscript
    rng = np.random.default_rng(0)
    sizes = [45, 32, 16, 24]
    layers = {f"hidden_{i}": {"kernel": rng.normal(size=(a, b)) * 0.3, "bias": rng.normal(size=b) * 0.1}
              for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:]))}
    mean, std = rng.normal(size=45), rng.uniform(0.5, 2.0, 45)
    params = ({"mean": {"state": mean}, "std": {"state": std}}, {"params": layers})
    cfg = load_config("walk_mjx")
    ts = export_torchscript(params, [32, 16], cfg, tmp_path / "p.pt")
    x = rng.normal(size=(5, 45))
    h = (x - mean) / std
    for i in range(3):
        h = h @ layers[f"hidden_{i}"]["kernel"] + layers[f"hidden_{i}"]["bias"]
        if i < 2:
            h = h / (1 + np.exp(-h))          # swish
    want = np.tanh(h[:, :12]) * cfg["actuator"]["action_scale"] / cfg["actuator"]["interface_scale"]
    with torch.no_grad():
        got = torch.jit.load(str(tmp_path / "p.pt"))(torch.tensor(x, dtype=torch.float32)).numpy()
    assert np.abs(got - want).max() < 1e-5
    assert ts is not None


@needs_model
def test_mjx_walk_env_stands_with_zero_action():
    jax = pytest.importorskip("jax")
    pytest.importorskip("mujoco_playground")
    import jax.numpy as jp

    from go2x.mjx.walk_env import Go2Walk
    env = Go2Walk(MODEL_DIR, randomize=False)
    st = jax.jit(env.reset)(jax.random.PRNGKey(1))
    step = jax.jit(env.step)
    assert st.obs["state"].shape == (45,)
    for _ in range(100):
        st = step(st, jp.zeros(12))
    assert float(st.done) == 0.0 and float(st.data.qpos[2]) > 0.2


@needs_model
def test_mujoco_nav_env_runs_a_baseline():
    pytest.importorskip("tensordict")
    from go2x.atrium.baselines import StraightController
    from go2x.atrium.rewards import normalize_action
    from go2x.mjnav.env import MujocoNavEnv
    walker = ROOT / "results/phase2/walk_train/policy.pt"
    sc = json.load(open(CONFIG_DIR / "nav_eval_scenarios.json"))["scenarios"][:2]
    env = MujocoNavEnv(2, MODEL_DIR, str(walker), seed=0, scenarios=sc, view="world", threads=2)
    obs = env.get_observations()
    assert obs["policy"].shape == (2, 8 * 64 * 64 + 3) and obs["critic"].shape == (2, 6)
    ctl = [StraightController(), StraightController()]
    nav = load_config("nav")
    for _ in range(5):
        img = env.image().numpy()
        cmd = torch.tensor(np.stack([c.act(img[i]) for i, c in enumerate(ctl)]), dtype=torch.float32)
        obs, rew, done, ex = env.step(normalize_action(cmd, nav["action"]))
    assert rew.shape == (2,) and torch.isfinite(rew).all() and "time_outs" in ex
    pose, *_ = env._pose_speed()
    assert (pose[:, :2].abs() < 1.6).all()       # still in the square, walking
    env.close()

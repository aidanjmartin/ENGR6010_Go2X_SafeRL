import numpy as np
from gymnasium.utils.env_checker import check_env

from go2x.sim.baselines import GoToGoalPolicy
from go2x.sim.dynamics import LagDelayDynamics
from go2x.sim.nav_env import Go2NavEnv
from go2x.sim.scene import Scene


def test_check_env():
    check_env(Go2NavEnv(), skip_render_check=True)


def test_lag_delay_step_response():
    dt, tau, delay = 0.05, 0.2, 0.1
    dyn = LagDelayDynamics(substeps=10, dt=dt)
    dyn.reset(np.random.default_rng(0), [0, 0, 0], tau=[tau] * 3, delay=[delay] * 3)
    t, v = [], []
    for k in range(40):
        dyn.step([1.0, 0, 0], dt)
        t.append((k + 1) * dt)
        v.append(dyn.vel[0])
    t, v = np.array(t), np.array(v)
    assert v[t <= delay - 1e-9].max() == 0.0  # nothing before the dead time ends
    expected = 1 - np.exp(-(t - delay) / tau)
    assert np.allclose(v[t > delay], expected[t > delay], atol=1e-9)


def test_raycast_and_collision():
    sc = Scene(4.0, 3.0, cylinders=np.array([[3.0, 1.5, 0.5]]),
               boxes=np.array([[1.0, 2.5, 0.0, 0.2, 0.2]]))
    r = sc.raycast((1.0, 1.5), np.array([0.0, np.pi / 2, np.pi, -np.pi / 2]), 10.0)
    assert np.allclose(r, [1.5, 0.8, 1.0, 1.5])
    assert sc.raycast((1.0, 1.5), np.array([0.0]), 1.0)[0] == 1.0  # max range clips
    assert sc.collides([[2.2, 1.5]], 0.35)[0]  # near cylinder
    assert not sc.collides([[1.0, 1.0]], 0.35)[0]
    assert sc.collides([[0.2, 1.5]], 0.35)[0]  # near wall


def test_seeded_reset_is_deterministic():
    a, b = Go2NavEnv(), Go2NavEnv()
    oa, _ = a.reset(seed=7)
    ob, _ = b.reset(seed=7)
    assert np.array_equal(oa, ob)
    assert np.allclose(a.dynamics.tau, b.dynamics.tau)


def test_filter_hook_is_applied():
    seen = []

    def zero_filter(cmd, env):
        seen.append(cmd.copy())
        return np.zeros(3)

    env = Go2NavEnv(action_filter=zero_filter)
    env.reset(seed=0)
    pose0 = env.dynamics.pose.copy()
    _, _, _, _, info = env.step(np.ones(3))
    assert np.allclose(seen[0], env.limits)
    assert np.allclose(info["cmd_applied"], 0)
    assert np.allclose(env.dynamics.pose, pose0)


def test_go_to_goal_reaches_goal_in_empty_room():
    cfg = Go2NavEnv().cfg
    cfg["scene"]["n_obstacles_range"] = [0, 0]
    env = Go2NavEnv(config=cfg)
    pol = GoToGoalPolicy()
    obs, _ = env.reset(seed=3)
    for _ in range(env.max_steps):
        obs, _, term, trunc, info = env.step(pol(obs, env))
        if term or trunc:
            break
    assert info["is_success"]

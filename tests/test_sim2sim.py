"""Low-level interface conventions used outside Isaac Lab."""
import numpy as np

from go2x.sim2sim import (DEFAULT_POS, JOINT_IDS_MAP, POLICY_JOINTS, SDK_JOINTS, joint_targets, observation, pd_torque,
                          quat_rotate_inverse, tilt)


def test_joint_ids_map_matches_unitree_export():
    # deploy.yaml written by unitree_rl_lab export_deploy_cfg in job 020
    assert JOINT_IDS_MAP == [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]
    assert all(SDK_JOINTS[JOINT_IDS_MAP[i]] == POLICY_JOINTS[i] for i in range(12))


def test_observation_layout_and_gravity():
    q = DEFAULT_POS + 0.1
    obs = observation([1.0, 0.0, 0.0], [1, 0, 0, 0], [0.5, 0.0, 0.2], q, np.ones(12), np.zeros(12))
    assert obs.shape == (45,)
    assert np.allclose(obs[0:3], [0.2, 0, 0])
    assert np.allclose(obs[3:6], [0, 0, -1])
    assert np.allclose(obs[6:9], [0.5, 0, 0.2])
    assert np.allclose(obs[9:21], 0.1)
    assert np.allclose(obs[21:33], 0.05)
    # rolled 90 degrees about x: gravity points along -y in the body frame... (body y axis now points up)
    c = np.cos(np.pi / 4)
    assert np.allclose(quat_rotate_inverse([c, c, 0, 0], [0, 0, -1]), [0, -1, 0], atol=1e-9)
    assert np.isclose(tilt([c, c, 0, 0]), np.pi / 2)
    assert np.isclose(tilt([1, 0, 0, 0]), 0.0)


def test_targets_and_pd():
    assert np.allclose(joint_targets(np.ones(12)), DEFAULT_POS + 0.25)
    tau = pd_torque(np.array([1.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 10.0]), 25.0, 0.5, np.array([20.0, 23.7]))
    assert np.allclose(tau, [20.0, -5.0])

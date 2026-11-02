"""Policy-side conventions shared by any non-Isaac runner of the walking policy (MuJoCo now, the robot later).

Implements docs/LOWLEVEL_INTERFACE.md: joint order mapping, the 45-value observation, joint targets, PD torque.
Pure numpy so it is tested on the CPU.
"""
import numpy as np

POLICY_JOINTS = ["FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
                 "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
                 "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"]
SDK_JOINTS = ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
              "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint", "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"]
JOINT_IDS_MAP = [SDK_JOINTS.index(n) for n in POLICY_JOINTS]   # policy index -> SDK index
DEFAULT_POS = np.array([0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5])  # policy order
OBS_SCALES = {"ang_vel": 0.2, "joint_vel": 0.05}


def quat_rotate_inverse(q_wxyz, v):
    """Rotate a world vector into the body frame given the body orientation quaternion (w, x, y, z)."""
    w, x, y, z = q_wxyz
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return R.T @ np.asarray(v, float)


def observation(gyro_b, quat_wxyz, command, q_policy, dq_policy, last_action):
    """The walking policy's 45-value input, in policy joint order."""
    g = quat_rotate_inverse(quat_wxyz, [0.0, 0.0, -1.0])
    return np.concatenate([np.asarray(gyro_b) * OBS_SCALES["ang_vel"], g, command, np.asarray(q_policy) - DEFAULT_POS,
                           np.asarray(dq_policy) * OBS_SCALES["joint_vel"], last_action]).astype(np.float32)


def joint_targets(action, scale=0.25):
    return DEFAULT_POS + scale * np.asarray(action)


def pd_torque(q_target, q, dq, kp, kd, limit):
    return np.clip(kp * (q_target - q) - kd * dq, -limit, limit)


def tilt(quat_wxyz):
    g = quat_rotate_inverse(quat_wxyz, [0.0, 0.0, -1.0])
    return float(np.arccos(np.clip(-g[2], -1, 1)))

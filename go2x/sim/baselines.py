"""Baseline controllers for Go2NavEnv. Each maps (obs, env) to a normalized action."""
import numpy as np


class RandomPolicy:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs, env):
        return self.rng.uniform(-1, 1, 3)


class GoToGoalPolicy:
    """Proportional go-to-goal controller with no obstacle awareness.

    vx = k_v d cos(b), vy = k_v d sin(b), wz = k_w b, with d and b the goal distance and
    body-frame bearing; forward speed is not allowed negative. Saturates at the env limits.
    """

    def __init__(self, k_v=1.0, k_w=2.0):
        self.k_v, self.k_w = k_v, k_w

    def __call__(self, obs, env):
        d, b = env.goal_polar()
        cmd = np.array([max(self.k_v * d * np.cos(b), 0.0), self.k_v * d * np.sin(b), self.k_w * b])
        return np.clip(cmd / env.limits, -1, 1)

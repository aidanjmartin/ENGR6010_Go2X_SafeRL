"""Transition logger for the learned dynamics model (safety filter, later phase).

Records, every policy step and for every environment, the base state and the velocity target. `build()`
turns the record into tuples (state at t, target at t, state at t + h for each horizon h) and drops any
tuple whose window crosses an episode reset. Pure numpy, so it is tested on the CPU.
"""
import numpy as np

STATE_FIELDS = ("x", "y", "z", "roll", "pitch", "yaw", "vx_b", "vy_b", "vz_b", "wx_b", "wy_b", "wz_b")
TARGET_FIELDS = ("vx_cmd", "vy_cmd", "wz_cmd")


class TransitionLogger:
    def __init__(self, dt, horizons_s, num_envs, state_dim=len(STATE_FIELDS), target_dim=len(TARGET_FIELDS)):
        self.dt = float(dt)
        self.horizons_s = [float(h) for h in horizons_s]
        self.offsets = []
        for h in self.horizons_s:
            k = h / self.dt
            if abs(k - round(k)) > 1e-6 or round(k) < 1:
                raise ValueError(f"horizon {h} s is not a positive multiple of the step {dt} s")
            self.offsets.append(int(round(k)))
        self.num_envs = num_envs
        self.state_dim, self.target_dim = state_dim, target_dim
        self._states, self._targets, self._episode = [], [], []
        self._episode_id = np.zeros(num_envs, dtype=np.int64)

    def record(self, state, target, new_episode=None):
        """state (num_envs, state_dim), target (num_envs, target_dim) at the same step.

        new_episode: bool (num_envs,), True where this state is the first one of a new episode
        (the environment was reset since the previous record)."""
        state, target = np.asarray(state, np.float32), np.asarray(target, np.float32)
        if state.shape != (self.num_envs, self.state_dim) or target.shape != (self.num_envs, self.target_dim):
            raise ValueError(f"bad shapes {state.shape} {target.shape}")
        if new_episode is not None and len(self._states):
            self._episode_id += np.asarray(new_episode, bool)
        self._states.append(state)
        self._targets.append(target)
        self._episode.append(self._episode_id.copy())

    def __len__(self):
        return len(self._states)

    def build(self):
        S = np.stack(self._states)      # (T, B, D)
        U = np.stack(self._targets)     # (T, B, 3)
        E = np.stack(self._episode)     # (T, B)
        T = S.shape[0]
        kmax = max(self.offsets)
        if T <= kmax:
            raise ValueError("record is shorter than the longest horizon")
        t_idx = np.arange(T - kmax)
        same = np.ones((len(t_idx), self.num_envs), bool)
        steady = np.ones_like(same)
        for k in range(1, kmax + 1):
            same &= E[t_idx + k] == E[t_idx]
            steady &= np.all(U[t_idx + k] == U[t_idx], axis=-1)
        tt, bb = np.nonzero(same)
        out = {
            "state": S[tt, bb],
            "target": U[tt, bb],
            "target_constant": steady[tt, bb],   # target unchanged over the longest window
            "env": bb.astype(np.int32),
            "episode": E[tt, bb].astype(np.int32),
            "t_s": (tt * self.dt).astype(np.float32),
        }
        for h, k in zip(self.horizons_s, self.offsets):
            out[f"state_next_{h:g}s"] = S[tt + k, bb]
        return out

    def save(self, path, **meta):
        data = self.build()
        np.savez_compressed(path, **data, state_fields=np.array(STATE_FIELDS), target_fields=np.array(TARGET_FIELDS),
                            horizons_s=np.array(self.horizons_s), dt=np.array(self.dt),
                            **{f"meta_{k}": np.array(v) for k, v in meta.items()})
        return {k: v.shape for k, v in data.items()}

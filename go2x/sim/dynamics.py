"""Planar body-velocity dynamics models.

A model holds the planar pose (x, y, yaw) in the world frame and the body velocity
(vx, vy, wz), and advances them given a commanded body velocity. The navigation
environment and the mock robot both use this class, so fitted and learned models in later
phases plug in by subclassing DynamicsModel.
"""
from abc import ABC, abstractmethod

import numpy as np


def integrate_pose(pose, vel, dt):
    """Advance (x, y, yaw) by body velocity (vx, vy, wz) over dt (forward Euler)."""
    x, y, yaw = pose
    vx, vy, wz = vel
    c, s = np.cos(yaw), np.sin(yaw)
    yaw = (yaw + wz * dt + np.pi) % (2 * np.pi) - np.pi
    return np.array([x + (c * vx - s * vy) * dt, y + (s * vx + c * vy) * dt, yaw])


class DynamicsModel(ABC):
    """Interface: reset() samples any per-episode parameters, step() advances by dt."""

    def __init__(self):
        self.pose = np.zeros(3)
        self.vel = np.zeros(3)

    def reset(self, rng, pose):
        self.pose = np.asarray(pose, dtype=float).copy()
        self.vel = np.zeros(3)

    @abstractmethod
    def step(self, cmd, dt):
        """Apply commanded body velocity cmd = (vx, vy, wz) for dt seconds."""


class LagDelayDynamics(DynamicsModel):
    """Per-axis first-order lag with dead time.

    ASSUMPTION, NOT FITTED: the time constants and delays are drawn uniformly from
    tau_range and delay_range, separately for each axis, at every reset. They are not
    measured on the Go2 X.

    Each control step is split into `substeps`. The delay is a FIFO of substep commands,
    so its resolution is dt / substeps. Each substep uses the exact discretization
    v <- v + (1 - exp(-h / tau)) (u_delayed - v).
    """

    def __init__(self, tau_range=(0.1, 0.4), delay_range=(0.0, 0.15), substeps=10, dt=0.05):
        super().__init__()
        self.tau_range = tau_range
        self.delay_range = delay_range
        self.substeps = substeps
        self.h = dt / substeps
        self.tau = np.full(3, np.mean(tau_range))
        self.delay = np.full(3, np.mean(delay_range))
        self._buf = None

    def reset(self, rng, pose, tau=None, delay=None):
        super().reset(rng, pose)
        self.tau = np.asarray(tau if tau is not None else rng.uniform(*self.tau_range, 3), float)
        self.delay = np.asarray(delay if delay is not None else rng.uniform(*self.delay_range, 3), float)
        self._lag = np.rint(self.delay / self.h).astype(int)
        self._buf = np.zeros((int(self._lag.max()) + 1, 3))  # row 0 newest
        self._alpha = 1.0 - np.exp(-self.h / self.tau)

    def step(self, cmd, dt):
        n = max(1, int(round(dt / self.h)))
        cmd = np.asarray(cmd, dtype=float)
        axes = np.arange(3)
        for _ in range(n):
            self._buf = np.roll(self._buf, 1, axis=0)
            self._buf[0] = cmd
            u = self._buf[self._lag, axes]
            self.vel = self.vel + self._alpha * (u - self.vel)
            self.pose = integrate_pose(self.pose, self.vel, self.h)

"""Go2 actuator with a per-episode motor strength factor.

Extends unitree_rl_lab's UnitreeActuator (assets/robots/unitree_actuators.py at 4960b84), which is a
DelayedPDActuator with the Go2 torque-speed limit and joint friction. The strength factor multiplies the
final joint torque, so a weak motor also has a lower torque limit.
"""
from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from unitree_rl_lab.assets.robots.unitree_actuators import UnitreeActuator, UnitreeActuatorCfg_Go2HV


class StrengthUnitreeActuator(UnitreeActuator):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.strength = torch.ones(self._num_envs, 1, device=self._device)

    def compute(self, control_action, joint_pos, joint_vel):
        control_action = super().compute(control_action, joint_pos, joint_vel)
        self.applied_effort *= self.strength
        control_action.joint_efforts = self.applied_effort
        return control_action


@configclass
class StrengthUnitreeActuatorCfg_Go2HV(UnitreeActuatorCfg_Go2HV):
    class_type: type = StrengthUnitreeActuator


def randomize_motor_strength(env, env_ids, asset_cfg: SceneEntityCfg, strength_range: tuple[float, float]):
    """Event term: draw a new strength factor for the given environments."""
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    lo, hi = strength_range
    for actuator in asset.actuators.values():
        if isinstance(actuator, StrengthUnitreeActuator):
            actuator.strength[env_ids] = lo + (hi - lo) * torch.rand(len(env_ids), 1, device=env.device)

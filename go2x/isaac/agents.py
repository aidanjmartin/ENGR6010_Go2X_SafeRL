"""RSL-RL PPO settings. The walking settings are unitree_rl_lab's BasePPORunnerCfg
(tasks/locomotion/agents/rsl_rl_ppo_cfg.py at 4960b84); only the experiment name differs."""
from isaaclab.utils import configclass
from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg


@configclass
class WalkPPORunnerCfg(BasePPORunnerCfg):
    experiment_name = "go2x_walk"

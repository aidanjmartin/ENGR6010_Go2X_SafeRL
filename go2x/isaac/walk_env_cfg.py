"""Task 1 walking environment: unitree_rl_lab's Go2 velocity task on a flat floor, with Phase 2 settings.

Starts from unitree_rl_lab 4960b84 `tasks/locomotion/robots/go2/velocity_env_cfg.py` (RobotEnvCfg) and changes:
- the Go2 model is loaded from the unitree_ros URDF (path from $UNITREE_ROS_DIR), as the unitree_rl_lab
  README recommends for Isaac Sim 5.x;
- flat plane instead of the generated cobblestone terrain; no height scanner; plain floor and light
  (no Nucleus material or sky texture);
- velocity target ranges, randomization ranges, observation noise, gains, action delay and timing from
  config/walk.yaml; added motor strength and PD gain randomization.
Rewards, terminations, observations and actions are unitree_rl_lab's.
"""
from __future__ import annotations

import os

import isaaclab.envs.mdp as il_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from unitree_rl_lab.assets.robots.unitree import UNITREE_GO2_CFG, UnitreeUrdfFileCfg
from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import RobotEnvCfg

from go2x.config import load_config
from go2x.isaac.actuators import StrengthUnitreeActuatorCfg_Go2HV, randomize_motor_strength

WALK = load_config("walk")


def go2_urdf_path() -> str:
    root = os.environ.get("UNITREE_ROS_DIR")
    if not root:
        raise RuntimeError("UNITREE_ROS_DIR is not set (jobs/common.sh sets it)")
    return os.path.join(root, "robots", "go2_description", "urdf", "go2_description.urdf")


def go2_robot_cfg(prim_path: str = "{ENV_REGEX_NS}/Robot"):
    act, rnd = WALK["actuator"], WALK["randomization"]
    lo, hi = rnd["action_delay_physics_steps"]
    return UNITREE_GO2_CFG.replace(
        prim_path=prim_path,
        spawn=UnitreeUrdfFileCfg(asset_path=go2_urdf_path()),
        actuators={
            "GO2HV": StrengthUnitreeActuatorCfg_Go2HV(
                joint_names_expr=[".*"],
                stiffness=act["kp"],
                damping=act["kd"],
                friction=0.01,  # unitree_rl_lab value
                min_delay=int(lo),
                max_delay=int(hi),
            )
        },
    )


def _range(r):
    return (float(r[0]), float(r[1]))


@configclass
class WalkEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        t, c, rnd = WALK["timing"], WALK["commands"], WALK["randomization"]

        # timing
        self.sim.dt = t["sim_dt"]
        self.decimation = t["decimation"]
        self.sim.render_interval = self.decimation
        self.episode_length_s = t["episode_length_s"]
        self.scene.contact_forces.update_period = self.sim.dt

        # flat floor, plain look
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.55, 0.52))
        self.scene.height_scanner = None
        self.scene.sky_light = AssetBaseCfg(
            prim_path="/World/skyLight", spawn=sim_utils.DomeLightCfg(intensity=1500.0, color=(1.0, 1.0, 1.0))
        )
        self.curriculum.terrain_levels = None

        # robot, gains, delay
        self.scene.robot = go2_robot_cfg()
        self.actions.JointPositionAction.scale = WALK["actuator"]["action_scale"]

        # velocity targets
        cmd = self.commands.base_velocity
        cmd.resampling_time_range = _range(c["resampling_time_s"])
        cmd.rel_standing_envs = c["rel_standing_envs"]
        cmd.ranges = mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=_range(c["curriculum_start"]["vx"]),
            lin_vel_y=_range(c["curriculum_start"]["vy"]),
            ang_vel_z=_range(c["curriculum_start"]["wz"]),
        )
        cmd.limit_ranges = mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=_range(c["train_range"]["vx"]),
            lin_vel_y=_range(c["train_range"]["vy"]),
            ang_vel_z=_range(c["train_range"]["wz"]),
        )

        # randomization
        ev = self.events
        ev.physics_material.params.update(
            static_friction_range=_range(rnd["static_friction"]),
            dynamic_friction_range=_range(rnd["dynamic_friction"]),
            restitution_range=_range(rnd["restitution"]),
            num_buckets=int(rnd["friction_buckets"]),
        )
        ev.add_base_mass.params["mass_distribution_params"] = _range(rnd["base_mass_add_kg"])
        ev.push_robot.interval_range_s = _range(rnd["push_interval_s"])
        ev.push_robot.params["velocity_range"] = {"x": _range(rnd["push_velocity_xy"]), "y": _range(rnd["push_velocity_xy"])}
        ev.motor_strength = EventTerm(
            func=randomize_motor_strength,
            mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot"), "strength_range": _range(rnd["motor_strength_scale"])},
        )
        ev.pd_gains = EventTerm(
            func=il_mdp.randomize_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "stiffness_distribution_params": _range(rnd["kp_scale"]),
                "damping_distribution_params": _range(rnd["kd_scale"]),
                "operation": "scale",
                "distribution": "uniform",
            },
        )

        # observation noise
        n, pol = rnd["obs_noise"], self.observations.policy
        pol.base_ang_vel.noise = Unoise(n_min=-n["base_ang_vel"], n_max=n["base_ang_vel"])
        pol.projected_gravity.noise = Unoise(n_min=-n["projected_gravity"], n_max=n["projected_gravity"])
        pol.joint_pos_rel.noise = Unoise(n_min=-n["joint_pos"], n_max=n["joint_pos"])
        pol.joint_vel_rel.noise = Unoise(n_min=-n["joint_vel"], n_max=n["joint_vel"])


@configclass
class WalkEvalEnvCfg(WalkEnvCfg):
    """Evaluation: full target range from the start, no curriculum, no pushes, fixed target hold time."""

    def __post_init__(self):
        super().__post_init__()
        ev = WALK["evaluation"]
        cmd = self.commands.base_velocity
        cmd.ranges = cmd.limit_ranges
        cmd.rel_standing_envs = 0.0
        cmd.resampling_time_range = (ev["command_hold_s"], ev["command_hold_s"])
        cmd.debug_vis = False
        self.curriculum.lin_vel_cmd_levels = None
        self.events.push_robot = None
        self.episode_length_s = ev["episode_s"]
        self.scene.num_envs = ev["num_episodes"]

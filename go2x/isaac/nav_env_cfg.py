"""Task 4 navigation environment in Isaac Lab: a navigation policy at 10 Hz over the frozen walking policy at
50 Hz, with full physics at 200 Hz, in the atrium square.

Structure follows Isaac Lab v2.3.2 `isaaclab_tasks/manager_based/navigation/config/anymal_c/navigation_env_cfg.py`
(Isaac-Navigation-Flat-Anymal-C-v0): a ManagerBasedRLEnv whose action term (`PreTrainedPolicyAction`) runs a
TorchScript walking policy on its own observation group. Changes: the walking policy and robot are Task 1's,
the action is a velocity target scaled to config/nav.yaml, the observation is the stacked drawn top-down
image plus the previous action, and the scene is the atrium with kinematic obstacles.

The per-environment atrium logic lives in go2x/atrium/state.py and go2x/atrium/topdown.py (CPU-tested); this
file is the Isaac glue. `get_runtime(env)` holds that state on the env.
"""
from __future__ import annotations

import json
import math
import os

import isaaclab.envs.mdp as il_mdp
import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, yaw_quat
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_tasks.manager_based.navigation.mdp import PreTrainedPolicyAction, PreTrainedPolicyActionCfg
from unitree_rl_lab.tasks.locomotion import mdp as ul_mdp

from go2x.atrium.camera import Camera
from go2x.atrium.rewards import TERMS, front_point, reward_terms, scale_action
from go2x.atrium.state import AtriumBatch, object_parts
from go2x.atrium.topdown import CameraFeed, view_config
from go2x.config import CONFIG_DIR, load_config
from go2x.isaac.actuators import randomize_motor_strength
from go2x.isaac.walk_env_cfg import WALK, WalkEnvCfg, go2_robot_cfg

ATRIUM, NAV, TD, CAM = load_config("atrium"), load_config("nav"), load_config("topdown"), load_config("camera")
SIM_DT = WALK["timing"]["sim_dt"]
WALK_DEC = WALK["timing"]["decimation"]
WALK_DT = SIM_DT * WALK_DEC
NAV_DEC = WALK_DEC * NAV["timing"]["walk_steps_per_nav"]
PARTS = object_parts(ATRIUM)
_N = WALK["randomization"]["obs_noise"]
_RND = WALK["randomization"]


# ------------------------------------------------------------------------------------------------ runtime state
class AtriumRuntime:
    def __init__(self, env):
        self.env = env
        B, dev = env.num_envs, env.device
        scen = getattr(env.cfg, "go2x_scenarios", None)
        seed = int(env.cfg.seed if env.cfg.seed is not None else 0)
        self.batch = AtriumBatch(B, dev, seed=seed, dt=WALK_DT, scenarios=scen)
        g = torch.Generator(device=dev)
        g.manual_seed(seed + 1)
        self.view = getattr(env.cfg, "go2x_image_view", "world")
        self.feed = CameraFeed(B, view_config(TD, self.view), tuple(ATRIUM["robot"]["footprint"]), dev, g)
        S = TD["size"]
        self.cur = torch.zeros(B, 4, S, S, device=dev)
        self.prev = torch.zeros_like(self.cur)
        self.first = torch.ones(B, dtype=torch.bool, device=dev)
        self.action_n = torch.zeros(B, 3, device=dev)
        self.prev_action_n = torch.zeros(B, 3, device=dev)
        self.prev_dist = torch.zeros(B, device=dev)
        self.robot = env.scene["robot"]
        self.obstacles = env.scene["obstacles"]
        self.contact = env.scene["contact_forces"]
        self.obstacle_contacts = [env.scene[f"obs_contact_{b}"] for b in ATRIUM["collision"]["robot_bodies"]]
        self.base_id = self.contact.find_bodies("base")[0]
        self.person_ids = torch.tensor(self.batch.person_part_ids(), device=dev)
        self._obs_step = -1
        self._m_step = -1
        self.m = None
        self.terms = None
        z = lambda: torch.zeros(B, device=dev)  # noqa: E731
        self.ep_min_person_gap = torch.full((B,), 1e3, device=dev)
        self.ep_fp_steps, self.ep_force_steps, self.ep_fp_only, self.ep_force_only = z(), z(), z(), z()

    # robot pose in the square frame
    def robot_pose(self):
        p = self.robot.data.root_pos_w[:, :2] - self.env.scene.env_origins[:, :2]
        q = yaw_quat(self.robot.data.root_quat_w)
        yaw = 2 * torch.atan2(q[:, 3], q[:, 0])
        return torch.cat([p, torch.atan2(torch.sin(yaw), torch.cos(yaw))[:, None]], -1)

    def write_parts(self, env_ids=None, part_ids=None):
        poses = self.batch.part_poses()
        ids = torch.arange(self.env.num_envs, device=self.env.device) if env_ids is None else env_ids
        poses = poses[ids]
        poses[..., :3] += self.env.scene.env_origins[ids][:, None, :]
        if part_ids is not None:
            poses = poses[:, part_ids]
        self.obstacles.write_object_link_pose_to_sim(poses, env_ids=ids, object_ids=part_ids)

    def reset(self, env_ids):
        start = self.batch.reset(env_ids)
        n = len(env_ids)
        root = self.robot.data.default_root_state[env_ids].clone()
        root[:, 0:2] = start[:, :2] + self.env.scene.env_origins[env_ids, :2]
        root[:, 2] += self.env.scene.env_origins[env_ids, 2]
        zeros = torch.zeros(n, device=self.env.device)
        root[:, 3:7] = quat_from_euler_xyz(zeros, zeros, start[:, 2])
        root[:, 7:] = 0.0
        self.robot.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(root[:, 7:], env_ids=env_ids)
        jp = self.robot.data.default_joint_pos[env_ids].clone()
        self.robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=env_ids)
        self.write_parts(env_ids)
        self.feed.reset(env_ids)
        self.first[env_ids] = True
        self._obs_step = -1            # an explicit env.reset() does not advance the step counter
        self.action_n[env_ids] = 0.0
        self.prev_action_n[env_ids] = 0.0
        pose = torch.cat([start[:, :2], start[:, 2:3]], -1)
        full = self.robot_pose()
        full[env_ids] = pose
        self.prev_dist[env_ids] = self.batch.target_distance(full)[env_ids]
        self.ep_min_person_gap[env_ids] = 1e3
        for t in (self.ep_fp_steps, self.ep_force_steps, self.ep_fp_only, self.ep_force_only):
            t[env_ids] = 0.0

    def walk_tick(self, k):
        """Before walking step k (0..4) of the current navigation step: move people, feed the camera."""
        idx = self.env.episode_length_buf * NAV["timing"]["walk_steps_per_nav"] + k
        self.batch.move_people(idx)
        self.write_parts(part_ids=self.person_ids)
        self.feed.observe(idx.float() * WALK_DT, self.batch.perception_state(self.robot_pose()))

    def image(self):
        """Stacked (current, previous) delivered frames, flattened. Computed once per environment step."""
        step = int(self.env.common_step_counter)
        if step != self._obs_step:
            t = self.env.episode_length_buf.float() * self.env.step_dt
            frame = self.feed.observe(t, self.batch.perception_state(self.robot_pose()))
            self.prev = torch.where(self.first[:, None, None, None], frame, self.cur)
            self.cur = frame
            self.first[:] = False
            self._obs_step = step
        return torch.cat([self.cur, self.prev], 1).flatten(1)

    def metrics(self):
        step = int(self.env.common_step_counter)
        if step == self._m_step:
            return self.m
        pose = self.robot_pose()
        speed = torch.linalg.norm(self.robot.data.root_lin_vel_b[:, :2], dim=-1)
        m = self.batch.metrics(pose, speed)
        thr = ATRIUM["collision"]["contact_force_threshold"]
        force_hit = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        for sensor in self.obstacle_contacts:                      # force_matrix_w: (B, 1, parts, 3)
            force_hit |= (torch.linalg.norm(sensor.data.force_matrix_w, dim=-1) > thr).flatten(1).any(-1)
        m["collision_force"] = force_hit
        base_f = torch.linalg.norm(self.contact.data.net_forces_w_history[:, :, self.base_id], dim=-1).amax((1, 2))
        tilt = torch.acos((-self.robot.data.projected_gravity_b[:, 2]).clamp(-1, 1))
        m["fall"] = (base_f > NAV["termination"]["fall_contact_n"]) | (tilt > NAV["termination"]["fall_tilt_rad"])
        m["collision"] = m["collision_footprint"] | m["collision_force"]
        m["success"] = m["at_target"] & ~m["collision"]
        self.terms = reward_terms(self.prev_dist, m["target_distance"], m["success"], m["collision"], m["person_gap"],
                                  m["fall"], m["out_of_square"], self.action_n, self.prev_action_n, NAV["reward"],
                                  ATRIUM["collision"]["person_near_distance"])
        self.prev_dist = m["target_distance"].clone()
        self.ep_min_person_gap = torch.minimum(self.ep_min_person_gap, m["person_gap"])
        fp, fo = m["collision_footprint"].float(), m["collision_force"].float()
        self.ep_fp_steps += fp
        self.ep_force_steps += fo
        self.ep_fp_only += fp * (1 - fo)
        self.ep_force_only += fo * (1 - fp)
        self.m, self._m_step = m, step
        return m


def get_runtime(env) -> AtriumRuntime:
    if not hasattr(env, "_go2x_atrium"):
        env._go2x_atrium = AtriumRuntime(env)
    return env._go2x_atrium


# ------------------------------------------------------------------------------------------------ MDP terms
def reset_atrium(env, env_ids):
    get_runtime(env).reset(env_ids)


def obs_image(env):
    return get_runtime(env).image()


def obs_prev_action(env):
    return get_runtime(env).action_n


def obs_critic_state(env):
    """Critic-only training aid (the policy never sees it): front-to-target vector in the robot frame, distance
    to the target footprint, planar base velocity and yaw rate in the robot frame."""
    rt = get_runtime(env)
    pose = rt.robot_pose()
    b = rt.batch
    center = torch.where(b.target_is_circle[:, None], b.target_circle[:, :2], b.target_rect[:, :2])
    front = front_point(pose, ATRIUM["robot"]["front_offset"])
    d = center - front
    c, s = torch.cos(pose[:, 2]), torch.sin(pose[:, 2])
    rel = torch.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], -1)
    v = rt.robot.data.root_lin_vel_b[:, :2]
    w = rt.robot.data.root_ang_vel_b[:, 2:3]
    return torch.cat([rel, b.target_distance(pose)[:, None], v, w], -1)


def _done(name):
    def f(env):
        return get_runtime(env).metrics()[name]
    f.__name__ = f"done_{name}"
    return f


def _reward(name):
    def f(env):
        get_runtime(env).metrics()
        return get_runtime(env).terms[name]
    f.__name__ = f"reward_{name}"
    return f


class AtriumNavAction(PreTrainedPolicyAction):
    """Isaac Lab's PreTrainedPolicyAction with the navigation output scaled to config/nav.yaml limits and the
    atrium updated (people, camera frames) before every walking-policy step."""

    def process_actions(self, actions):
        rt = get_runtime(self._env)
        a = actions.clamp(-1.0, 1.0)
        rt.prev_action_n = rt.action_n.clone()
        rt.action_n = a.clone()
        self._raw_actions[:] = scale_action(a, NAV["action"])
        self._tick = 0
        self._counter = 0

    def apply_actions(self):
        if self._tick % self.cfg.low_level_decimation == 0:
            get_runtime(self._env).walk_tick(self._tick // self.cfg.low_level_decimation)
        self._tick += 1
        super().apply_actions()


@configclass
class AtriumNavActionCfg(PreTrainedPolicyActionCfg):
    class_type: type = AtriumNavAction


# ------------------------------------------------------------------------------------------------ configs
@configclass
class WalkPolicyObsCfg(ObsGroup):
    """The walking policy's observation, same terms and order as unitree_rl_lab's PolicyCfg, with the last
    term named `actions` as PreTrainedPolicyAction requires."""

    base_ang_vel = ObsTerm(func=ul_mdp.base_ang_vel, scale=0.2, clip=(-100, 100),
                           noise=Unoise(n_min=-_N["base_ang_vel"], n_max=_N["base_ang_vel"]))
    projected_gravity = ObsTerm(func=ul_mdp.projected_gravity, clip=(-100, 100),
                                noise=Unoise(n_min=-_N["projected_gravity"], n_max=_N["projected_gravity"]))
    velocity_commands = ObsTerm(func=ul_mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"})
    joint_pos_rel = ObsTerm(func=ul_mdp.joint_pos_rel, clip=(-100, 100), noise=Unoise(n_min=-_N["joint_pos"], n_max=_N["joint_pos"]))
    joint_vel_rel = ObsTerm(func=ul_mdp.joint_vel_rel, scale=0.05, clip=(-100, 100),
                            noise=Unoise(n_min=-_N["joint_vel"], n_max=_N["joint_vel"]))
    actions = ObsTerm(func=ul_mdp.last_action, clip=(-100, 100))

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


def _part_spawn(p):
    rigid = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True)
    look = sim_utils.PreviewSurfaceCfg(diffuse_color=p["color"])
    common = dict(rigid_props=rigid, mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
                  collision_props=sim_utils.CollisionPropertiesCfg(), visual_material=look)
    if p["shape"] == "box":
        return sim_utils.CuboidCfg(size=p["size"], **common)
    return sim_utils.CylinderCfg(radius=p["size"][0], height=p["size"][1], axis="Z", **common)


def _obstacles_cfg():
    ab_park = AtriumBatch(1, scenarios=None)
    objs = {}
    for p in PARTS:
        x, y = ab_park.park_xy[p["slot"]].tolist()
        objs[p["name"]] = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Obs_" + p["name"], spawn=_part_spawn(p),
                                         init_state=RigidObjectCfg.InitialStateCfg(pos=(x + p["offset"][0], y + p["offset"][1], p["offset"][2])))
    return RigidObjectCollectionCfg(rigid_objects=objs)


@configclass
class AtriumSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="plane", collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(friction_combine_mode="multiply", restitution_combine_mode="multiply",
                                                        static_friction=1.0, dynamic_friction=1.0),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.62, 0.60, 0.56)),
    )
    robot = go2_robot_cfg()
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    obstacles = _obstacles_cfg()
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=1500.0, color=(1.0, 1.0, 1.0)))


@configclass
class ActionsCfg:
    nav = AtriumNavActionCfg(
        asset_name="robot",
        policy_path=os.environ.get("GO2X_WALK_POLICY", "set GO2X_WALK_POLICY"),
        low_level_decimation=WALK_DEC,
        low_level_actions=WalkEnvCfg().actions.JointPositionAction,
        low_level_observations=WalkPolicyObsCfg(),
        debug_vis=False,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        image = ObsTerm(func=obs_image)
        prev_action = ObsTerm(func=obs_prev_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Critic-only state. The critic reads the policy group followed by this one (obs_groups in
        nav_agent_dict), so the image is stored once in the rollout buffer (a copy here ran out of GPU memory)."""
        state = ObsTerm(func=obs_critic_state)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    physics_material = EventTerm(func=il_mdp.randomize_rigid_body_material, mode="startup", params={
        "asset_cfg": SceneEntityCfg("robot", body_names=".*"), "static_friction_range": tuple(_RND["static_friction"]),
        "dynamic_friction_range": tuple(_RND["dynamic_friction"]), "restitution_range": tuple(_RND["restitution"]),
        "num_buckets": int(_RND["friction_buckets"])})
    add_base_mass = EventTerm(func=il_mdp.randomize_rigid_body_mass, mode="startup", params={
        "asset_cfg": SceneEntityCfg("robot", body_names="base"), "mass_distribution_params": tuple(_RND["base_mass_add_kg"]),
        "operation": "add"})
    atrium = EventTerm(func=reset_atrium, mode="reset")
    motor_strength = EventTerm(func=randomize_motor_strength, mode="reset", params={
        "asset_cfg": SceneEntityCfg("robot"), "strength_range": tuple(_RND["motor_strength_scale"])})
    pd_gains = EventTerm(func=il_mdp.randomize_actuator_gains, mode="reset", params={
        "asset_cfg": SceneEntityCfg("robot"), "stiffness_distribution_params": tuple(_RND["kp_scale"]),
        "damping_distribution_params": tuple(_RND["kd_scale"]), "operation": "scale", "distribution": "uniform"})


NAV_STEP_DT = SIM_DT * NAV_DEC


@configclass
class RewardsCfg:
    # The reward manager multiplies each term by weight x step_dt; weight 1 / step_dt keeps the config's
    # per-step weights (applied inside go2x.atrium.rewards).
    progress = RewTerm(func=_reward("progress"), weight=1.0 / NAV_STEP_DT)
    success = RewTerm(func=_reward("success"), weight=1.0 / NAV_STEP_DT)
    collision = RewTerm(func=_reward("collision"), weight=1.0 / NAV_STEP_DT)
    person_near = RewTerm(func=_reward("person_near"), weight=1.0 / NAV_STEP_DT)
    fall = RewTerm(func=_reward("fall"), weight=1.0 / NAV_STEP_DT)
    out_of_square = RewTerm(func=_reward("out_of_square"), weight=1.0 / NAV_STEP_DT)
    time = RewTerm(func=_reward("time"), weight=1.0 / NAV_STEP_DT)
    smoothness = RewTerm(func=_reward("smoothness"), weight=1.0 / NAV_STEP_DT)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=il_mdp.time_out, time_out=True)
    success = DoneTerm(func=_done("success"))
    collision_footprint = DoneTerm(func=_done("collision_footprint"))
    collision_force = DoneTerm(func=_done("collision_force"))
    fall = DoneTerm(func=_done("fall"))
    out_of_square = DoneTerm(func=_done("out_of_square"))


@configclass
class CommandsCfg:
    pass


@configclass
class AtriumNavEnvCfg(ManagerBasedRLEnvCfg):
    scene: AtriumSceneCfg = AtriumSceneCfg(num_envs=NAV["training"]["num_envs"], env_spacing=ATRIUM["slots"]["env_spacing"])
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands: CommandsCfg = CommandsCfg()
    go2x_scenarios: list | None = None
    go2x_image_view: str = NAV["policy"]["image_view"]   # the baselines run with "world"

    def __post_init__(self):
        self.sim.dt = SIM_DT
        self.decimation = NAV_DEC
        self.sim.render_interval = NAV_DEC
        self.episode_length_s = NAV["timing"]["episode_s"]
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.contact_forces.update_period = self.sim.dt
        for b in ATRIUM["collision"]["robot_bodies"]:
            setattr(self.scene, f"obs_contact_{b}", ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/" + b, history_length=1, update_period=self.sim.dt,
                filter_prim_paths_expr=["{ENV_REGEX_NS}/Obs_" + p["name"] for p in PARTS]))
        self.seed = NAV["training"]["seed"]


def building_boxes():
    with open(CONFIG_DIR / "building_level1_near_square.json") as f:
        return json.load(f)["boxes"]


BUILDING_COLORS = {"wall": (0.85, 0.85, 0.82), "glass": (0.55, 0.70, 0.80), "rail": (0.35, 0.35, 0.38),
                   "low": (0.70, 0.68, 0.62), "stairs": (0.60, 0.58, 0.55), "barrier": (0.5, 0.5, 0.5),
                   "closed_door": (0.55, 0.40, 0.30)}


def add_render_parts(cfg: AtriumNavEnvCfg):
    """Rendered variant: the Level 1 building boxes near the square (static, visual only) and the fixed camera."""
    for i, b in enumerate(building_boxes()):
        q = quat_from_euler_xyz(torch.zeros(1), torch.zeros(1), torch.tensor([b["yaw"]]))[0].tolist()
        setattr(cfg.scene, f"bldg_{i}", AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Bldg_" + b["name"],
            spawn=sim_utils.CuboidCfg(size=(b["sx"], b["sy"], b["sz"]),
                                      visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=BUILDING_COLORS.get(b["kind"], (0.7, 0.7, 0.7)))),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(b["x"], b["y"], b["z"]), rot=tuple(q))))
    cam = Camera.from_config(CAM)
    cfg.scene.top_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/TopCamera", update_period=0.0, height=CAM["intrinsics"]["height"],
        width=CAM["intrinsics"]["width"], data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(intrinsic_matrix=cam.K.flatten().tolist(),
                                                               width=cam.width, height=cam.height, clipping_range=(0.05, 30.0)),
        offset=CameraCfg.OffsetCfg(pos=tuple(cam.C.tolist()), rot=tuple(cam.quat_wxyz_ros().tolist()), convention="ros"))
    return cfg


def nav_agent_dict(agent_cfg_dict):
    """Add the CNN policy settings that RslRlPpoActorCriticCfg has no fields for."""
    p = NAV["policy"]
    agent_cfg_dict["policy"].update(class_name="CNNActorCritic", image_shape=[2 * 4, TD["size"], TD["size"]],
                                    cnn_channels=p["cnn_channels"], cnn_out=p["cnn_out"], mlp=p["mlp"])
    agent_cfg_dict["obs_groups"] = {"policy": ["policy"], "critic": ["policy", "critic"]}
    return agent_cfg_dict


assert set(TERMS) == {k for k in RewardsCfg.__dataclass_fields__}, "every reward term is logged"
_ = math

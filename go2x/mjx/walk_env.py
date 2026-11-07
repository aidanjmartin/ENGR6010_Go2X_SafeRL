"""Go2 velocity-tracking task in MuJoCo MJX, for Brax PPO (scripts/mjx/train_walk.py).

The task matches the Isaac Lab walker's (config/walk.yaml, unitree_rl_lab Go2 velocity task) so the two trainers
can be compared on the same evaluation: same 45-value observation (go2x/sim2sim.py), same PD at every physics
step, same reward terms and weights, same terminations, same domain randomization ranges. Settings:
config/walk_mjx.yaml. The structure (reset, step, perturbations, privileged critic state) follows MuJoCo
Playground's Go1 joystick task (playground 0.2.0, locomotion/go1/joystick.py, Apache-2.0).

Differences from the Isaac task, all forced by the simulator or trainer:
- Brax PPO's policy is tanh-bounded, so the action is in [-1, 1] and scaled by 0.5 rad (Playground Go1) instead
  of unbounded and scaled by 0.25 rad; the exported policy rescales its output to the Isaac interface.
- Contacts are geometric on the flat floor (a foot touches when its sphere reaches the floor), not force
  sensors; the base contact termination checks the base box corners; a head touch is an undesired contact, as for
  Isaac's separate head link (go2x/mjx/go2_model.py).
- Episodes start with the feet on the floor (Isaac drops the robot from 0.40 m).
- No command curriculum (Playground samples the full range from the start).
"""
from typing import Any, Dict

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env

from go2x.config import load_config
from go2x.mjx import go2_model as G
from go2x.sim2sim import DEFAULT_POS, OBS_SCALES

CONTACT_TOL = 0.005    # m: a sphere this close to the floor touches it (contact margin 1 mm plus soft-contact play)


def default_config(cfg=None):
    cfg = cfg or load_config("walk_mjx")
    t = cfg["timing"]
    return config_dict.create(ctrl_dt=t["sim_dt"] * t["substeps"], sim_dt=t["sim_dt"], episode_length=t["episode_steps"],
                              impl="jax")


def _quat_rotate_inv(q, v):
    w, x, y, z = q[0], q[1], q[2], q[3]
    R = jp.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return R.T @ v


class Go2Walk(mjx_env.MjxEnv):
    def __init__(self, model_dir, cfg=None, randomize=None):
        self.cfg = cfg or load_config("walk_mjx")
        super().__init__(default_config(self.cfg))
        t = self.cfg["timing"]
        self._mj_model = G.training_model(model_dir, t["sim_dt"])
        self._mjx_model = mjx.put_model(self._mj_model, impl="jax")
        self._model_dir = model_dir
        R = self.cfg["randomization"]
        self.randomize = R["enabled"] if randomize is None else randomize
        ix = G.indices(self._mj_model)
        self._qadr, self._vadr, self._act = jp.array(ix["qadr"]), jp.array(ix["vadr"]), jp.array(ix["act"])
        self._limit = jp.array(ix["limit"])
        f = self.cfg["rewards"]["soft_limit_factor"]
        mid, half = (ix["upper"] + ix["lower"]) / 2, (ix["upper"] - ix["lower"]) / 2
        self._soft_lo, self._soft_hi = jp.array(mid - f * half), jp.array(mid + f * half)
        gid = G.geom_ids(self._mj_model)
        self._feet, self._knees = jp.array(gid["feet"]), jp.array(gid["knees"])
        self._base = gid["base_body"]
        corners, spheres = G.base_corners_local(model_dir)
        self._corners = jp.array(corners)
        self._spheres = jp.array([np.r_[c, r] for c, r in spheres])
        self._foot_r = float(self._mj_model.geom_size[gid["feet"][0], 0])
        self._knee_r = float(self._mj_model.geom_size[gid["knees"][0], 0])
        self._init_q = jp.array(G.home_qpos(self._mj_model, G.standing_height(self._mj_model)))
        self._default = jp.array(DEFAULT_POS)
        a = self.cfg["actuator"]
        self._kp, self._kd, self._ascale = a["kp"], a["kd"], a["action_scale"]
        self._iface = a["action_scale"] / a["interface_scale"]   # raw action -> Isaac-interface action
        c = self.cfg["commands"]
        self._cmd_lo = jp.array([c["vx"][0], c["vy"][0], c["wz"][0]])
        self._cmd_hi = jp.array([c["vx"][1], c["vy"][1], c["wz"][1]])
        self._resample = int(round(c["resample_s"] / self.dt))
        self._w = self.cfg["rewards"]["weights"]

    # ------------------------------------------------------------------------------------------- properties
    @property
    def xml_path(self):
        return self._model_dir

    @property
    def action_size(self):
        return 12

    @property
    def mj_model(self):
        return self._mj_model

    @property
    def mjx_model(self):
        return self._mjx_model

    # ------------------------------------------------------------------------------------------- helpers
    def _u(self, key, lo_hi, shape=()):
        lo, hi = lo_hi
        return jax.random.uniform(key, shape, minval=lo, maxval=hi)

    def _sample_command(self, key):
        k1, k2 = jax.random.split(key)
        cmd = jax.random.uniform(k1, (3,), minval=self._cmd_lo, maxval=self._cmd_hi)
        stand = jax.random.bernoulli(k2, self.cfg["commands"]["standing_fraction"])
        return jp.where(stand, jp.zeros(3), cmd)

    def _base_state(self, data):
        q = data.qpos[3:7]
        g = _quat_rotate_inv(q, jp.array([0.0, 0.0, -1.0]))
        v_b = _quat_rotate_inv(q, data.qvel[0:3])
        w_b = data.qvel[3:6]           # free-joint angular velocity is in the body frame
        return q, g, v_b, w_b

    def _base_touch(self, data):
        R = data.xmat[self._base]
        p = data.xpos[self._base]
        cz = (self._corners @ R.T + p)[:, 2]
        sz = (self._spheres[:, :3] @ R.T + p)[:, 2] - self._spheres[:, 3]
        return jp.min(cz) < 0.0, jp.min(sz) < 0.0

    def _obs(self, data, info, key):
        q, g, v_b, w_b = self._base_state(data)
        jq, jv = data.qpos[self._qadr], data.qvel[self._vadr]
        last = info["last_act"] * self._iface
        n = self.cfg["randomization"]["obs_noise"]
        lvl = 1.0 if self.randomize else 0.0
        ks = jax.random.split(key, 4)
        noise = lambda k, s, shape: lvl * s * (2 * jax.random.uniform(k, shape) - 1)  # noqa: E731
        state = jp.concatenate([
            (w_b + noise(ks[0], n["base_ang_vel"], (3,))) * OBS_SCALES["ang_vel"],
            g + noise(ks[1], n["projected_gravity"], (3,)),
            info["command"],
            jq + noise(ks[2], n["joint_pos"], (12,)) - self._default,
            (jv + noise(ks[3], n["joint_vel"], (12,))) * OBS_SCALES["joint_vel"],
            last])
        clean = jp.concatenate([w_b * OBS_SCALES["ang_vel"], g, info["command"], jq - self._default,
                                jv * OBS_SCALES["joint_vel"], last])
        R = self.cfg["randomization"]
        priv = jp.concatenate([
            clean, v_b, data.qpos[2:3], info["contact"].astype(jp.float32), jp.clip(info["air_time"], 0, 1),
            self.mjx_model.geom_friction[self._feet[0], 0:1],
            jp.stack([info["kp_s"], info["kd_s"], info["strength"], info["delay"] / max(R["action_delay_physics_steps"][1], 1),
                      info["mass_add"]]),
            (info["push_countdown"] <= 1).astype(jp.float32)[None]])
        return {"state": state, "privileged_state": priv}

    # ------------------------------------------------------------------------------------------- reset
    def reset(self, rng):
        R = self.cfg["randomization"]
        ks = jax.random.split(rng, 12)
        qpos = self._init_q
        yaw = jax.random.uniform(ks[0], (), minval=-jp.pi, maxval=jp.pi)
        qpos = qpos.at[3:7].set(jp.array([jp.cos(yaw / 2), 0.0, 0.0, jp.sin(yaw / 2)]))
        qpos = qpos.at[0:2].set(jax.random.uniform(ks[1], (2,), minval=-0.5, maxval=0.5))
        qvel = jp.zeros(self._mjx_model.nv)
        data = mjx_env.make_data(self._mj_model, qpos=qpos, qvel=qvel, impl="jax")
        data = mjx.forward(self._mjx_model, data)
        on = self.randomize
        info = {
            "rng": ks[2],
            "command": self._sample_command(ks[3]),
            "cmd_countdown": jp.array(self._resample, jp.int32),
            "last_act": jp.zeros(12), "hist": jp.zeros((3, 12)),
            "last_jv": jp.zeros(12),
            "contact": jp.ones(4, bool),
            "air_time": jp.zeros(4), "contact_time": jp.zeros(4),
            "last_air_time": jp.zeros(4), "last_contact_time": jp.zeros(4),
            "feet_prev": data.geom_xpos[self._feet],
            "kp_s": self._u(ks[4], R["kp_scale"]) if on else jp.array(1.0),
            "kd_s": self._u(ks[5], R["kd_scale"]) if on else jp.array(1.0),
            "strength": self._u(ks[6], R["motor_strength"]) if on else jp.array(1.0),
            "delay": jax.random.randint(ks[7], (), R["action_delay_physics_steps"][0], R["action_delay_physics_steps"][1] + 1)
            if on else jp.array(0, jp.int32),
            "mass_add": jp.array(0.0),    # set by the randomization function's model; kept for the critic
            "push_countdown": jp.round(self._u(ks[8], R["push_interval_s"]) / self.dt).astype(jp.int32),
        }
        if on:
            info["mass_add"] = self.mjx_model.body_mass[self._base] - self._mj_model.body_mass[self._base]
        metrics = {f"reward/{k}": jp.zeros(()) for k in self._w}
        metrics.update({"track/err_vx": jp.zeros(()), "track/err_wz": jp.zeros(())})
        obs = self._obs(data, info, ks[9])
        return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    # ------------------------------------------------------------------------------------------- step
    def _push(self, data, info, key):
        R = self.cfg["randomization"]
        k1, k2 = jax.random.split(key)
        do = (info["push_countdown"] <= 0) & self.randomize
        dv = jax.random.uniform(k1, (2,), minval=-R["push_velocity_xy"], maxval=R["push_velocity_xy"])
        qvel = data.qvel.at[0:2].add(jp.where(do, dv, 0.0))
        nxt = jp.round(self._u(k2, R["push_interval_s"]) / self.dt).astype(jp.int32)
        info["push_countdown"] = jp.where(do, nxt, info["push_countdown"] - 1)
        return data.replace(qvel=qvel)

    def step(self, state, action):
        info = dict(state.info)
        info["rng"], k_push, k_obs, k_cmd = jax.random.split(info["rng"], 4)
        data = self._push(state.data, info, k_push)
        hist = jp.concatenate([action[None], info["hist"][:2]], 0)      # actions at t, t-1, t-2
        kp, kd = self._kp * info["kp_s"], self._kd * info["kd_s"]
        n_sub = self.n_substeps

        def substep(d, k):
            lag = jp.clip(jp.ceil((info["delay"] - k) / n_sub), 0, 2).astype(jp.int32)
            target = self._default + self._ascale * hist[lag]
            tau = jp.clip(kp * (target - d.qpos[self._qadr]) - kd * d.qvel[self._vadr], -self._limit, self._limit)
            ctrl = jp.zeros(self._mjx_model.nu).at[self._act].set(info["strength"] * tau)
            d = mjx.step(self._mjx_model, d.replace(ctrl=ctrl))
            return d, (tau * info["strength"])

        data, taus = jax.lax.scan(substep, data, jp.arange(n_sub))
        tau = taus[-1]

        q, g, v_b, w_b = self._base_state(data)
        jq, jv = data.qpos[self._qadr], data.qvel[self._vadr]
        feet = data.geom_xpos[self._feet]
        contact = feet[:, 2] - self._foot_r < CONTACT_TOL
        knee_touch = data.geom_xpos[self._knees][:, 2] - self._knee_r < CONTACT_TOL
        dt = self.dt
        # Isaac Lab ContactSensor air-time bookkeeping
        first_contact = (info["air_time"] > 0) & contact
        first_air = (info["contact_time"] > 0) & ~contact
        last_air = jp.where(first_contact, info["air_time"] + dt, info["last_air_time"])
        last_con = jp.where(first_air, info["contact_time"] + dt, info["last_contact_time"])
        air = jp.where(contact, 0.0, info["air_time"] + dt)
        con = jp.where(contact, info["contact_time"] + dt, 0.0)

        cmd = info["command"]
        tilt = jp.arccos(jp.clip(-g[2], -1.0, 1.0))
        base_touch, head_touch = self._base_touch(data)
        done = (tilt > self.cfg["termination"]["tilt_rad"]) | base_touch | jp.any(jp.isnan(data.qpos))

        sig = self.cfg["rewards"]["tracking_sigma"]
        Rw = self.cfg["rewards"]
        feet_v = (feet - info["feet_prev"]) / dt
        jnorm = jp.linalg.norm(jq - self._default)
        moving = (jp.linalg.norm(cmd) > 0.0) | (jp.linalg.norm(v_b[:2]) > Rw["stand_velocity_threshold"])
        act_iface = action * self._iface
        last_iface = info["last_act"] * self._iface
        terms = {
            "track_lin_vel_xy": jp.exp(-jp.sum(jp.square(cmd[:2] - v_b[:2])) / sig),
            "track_ang_vel_z": jp.exp(-jp.square(cmd[2] - w_b[2]) / sig),
            "lin_vel_z": jp.square(v_b[2]),
            "ang_vel_xy": jp.sum(jp.square(w_b[:2])),
            "joint_vel": jp.sum(jp.square(jv)),
            "joint_acc": jp.sum(jp.square((jv - info["last_jv"]) / dt)),
            "joint_torques": jp.sum(jp.square(tau)),
            "action_rate": jp.sum(jp.square(act_iface - last_iface)),
            "dof_pos_limits": jp.sum(jp.clip(self._soft_lo - jq, 0.0) + jp.clip(jq - self._soft_hi, 0.0)),
            "energy": jp.sum(jp.abs(jv) * jp.abs(tau)),
            "flat_orientation": jp.sum(jp.square(g[:2])),
            "joint_pos": jp.where(moving, jnorm, Rw["stand_still_scale"] * jnorm),
            "feet_air_time": jp.sum((last_air - Rw["air_time_threshold"]) * first_contact) * (jp.linalg.norm(cmd[:2]) > 0.1),
            "air_time_variance": jp.var(jp.clip(last_air, max=0.5)) + jp.var(jp.clip(last_con, max=0.5)),
            "feet_slide": jp.sum(jp.linalg.norm(feet_v[:, :2], axis=-1) * contact),
            "undesired_contacts": jp.sum(knee_touch.astype(jp.float32)) + head_touch.astype(jp.float32),
        }
        rewards = {k: v * self._w[k] for k, v in terms.items()}
        reward = sum(rewards.values()) * dt

        info["cmd_countdown"] = info["cmd_countdown"] - 1
        resample = info["cmd_countdown"] <= 0
        info["command"] = jp.where(resample, self._sample_command(k_cmd), cmd)
        info["cmd_countdown"] = jp.where(resample, self._resample, info["cmd_countdown"])
        info.update(last_act=action, hist=hist, last_jv=jv, contact=contact, air_time=air, contact_time=con,
                    last_air_time=last_air, last_contact_time=last_con, feet_prev=feet)
        metrics = dict(state.metrics)
        for k, v in rewards.items():
            metrics[f"reward/{k}"] = v
        metrics["track/err_vx"] = jp.abs(cmd[0] - v_b[0])
        metrics["track/err_wz"] = jp.abs(cmd[2] - w_b[2])
        obs = self._obs(data, info, k_obs)
        reward = jp.where(jp.isnan(reward), 0.0, reward)
        return state.replace(data=data, obs=obs, reward=reward, done=done.astype(jp.float32), metrics=metrics, info=info)


def domain_randomize(model: mjx.Model, rng: jax.Array, mj_model: mujoco.MjModel, cfg=None):
    """Per-environment model randomization for the Playground training wrapper (vmapped over rng)."""
    R = (cfg or load_config("walk_mjx"))["randomization"]
    feet = np.array([mj_model.geom(f).id for f in G.FEET])
    base = mj_model.body("base_link").id
    leg_dofs = np.arange(6, mj_model.nv)

    @jax.vmap
    def rand(key):
        k = jax.random.split(key, 6)
        fr = jax.random.uniform(k[0], minval=R["foot_friction"][0], maxval=R["foot_friction"][1])
        geom_friction = model.geom_friction.at[feet, 0].set(fr)
        mass = model.body_mass * jax.random.uniform(k[1], (model.nbody,), minval=R["link_mass_scale"][0],
                                                    maxval=R["link_mass_scale"][1])
        mass = mass.at[base].add(jax.random.uniform(k[2], minval=R["base_mass_add_kg"][0], maxval=R["base_mass_add_kg"][1]))
        c = R["base_com_offset_m"]
        ipos = model.body_ipos.at[base].add(jax.random.uniform(k[3], (3,), minval=-c, maxval=c))
        fl = model.dof_frictionloss.at[leg_dofs].multiply(
            jax.random.uniform(k[4], (len(leg_dofs),), minval=R["joint_frictionloss_scale"][0], maxval=R["joint_frictionloss_scale"][1]))
        arm = model.dof_armature.at[leg_dofs].multiply(
            jax.random.uniform(k[5], (len(leg_dofs),), minval=R["joint_armature_scale"][0], maxval=R["joint_armature_scale"][1]))
        return geom_friction, mass, ipos, fl, arm

    geom_friction, mass, ipos, fl, arm = rand(rng)
    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace({"geom_friction": 0, "body_mass": 0, "body_ipos": 0, "dof_frictionloss": 0,
                                    "dof_armature": 0})
    model = model.tree_replace({"geom_friction": geom_friction, "body_mass": mass, "body_ipos": ipos,
                                "dof_frictionloss": fl, "dof_armature": arm})
    return model, in_axes


Info = Dict[str, Any]

"""The Go2 model for training in MuJoCo, built from Unitree's go2.xml (config/mujoco.yaml) with mujoco.MjSpec.

`training_model` keeps the bodies, joints, masses, inertias, joint damping, armature and friction loss of
go2.xml, and changes only contacts, for speed on the GPU (MJX):
- the four foot spheres collide with the floor (condim 3 instead of 6), with the pyramidal friction cone and
  4 solver iterations instead of the elliptic cone at impratio 100;
- a sphere at each knee (calf joint, radius 0.025 m) collides with the floor, so a gait that drags the knees
  meets the floor; its contact counts as an undesired contact;
- no other geom collides (the base box touching the floor ends the episode, the head touching it is an
  undesired contact: see `base_corners_local`), and the robot does not collide with itself.

`eval_model` is go2.xml unchanged on a flat floor (go2x/mj_scene.py): every collision geom, condim 6 feet. The
evaluation scripts use it, so every reported number comes from the full model, not the training model.

Index helpers give, in policy joint order (go2x/sim2sim.py POLICY_JOINTS), the qpos and qvel addresses and the
actuator ids.
"""
import itertools

import mujoco
import numpy as np

from go2x.mj_scene import go2_spec
from go2x.sim2sim import DEFAULT_POS, POLICY_JOINTS

FEET = ("FL", "FR", "RL", "RR")
KNEE_RADIUS = 0.025


def _is_collision(g):
    return g.contype != 0 or g.conaffinity != 0


def training_spec(model_dir, sim_dt=0.005):
    spec = go2_spec(model_dir, offscreen=(640, 480))
    spec.option.timestep = sim_dt
    for b in spec.bodies:
        for g in b.geoms:
            if g.name in FEET:
                g.condim = 3
                g.contype, g.conaffinity = 1, 0
            elif g.name == "floor":
                g.contype, g.conaffinity = 0, 1
            elif _is_collision(g):
                g.contype, g.conaffinity = 0, 0
    for leg in FEET:
        calf = spec.body(f"{leg}_calf")
        calf.add_geom(name=f"{leg}_knee", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[KNEE_RADIUS, 0, 0],
                      contype=1, conaffinity=0, condim=3, group=3, rgba=[0, 0, 0, 0])
    return spec


def training_model(model_dir, sim_dt=0.005, iterations=4, ls_iterations=8):
    spec = training_spec(model_dir, sim_dt)
    spec.option.iterations = iterations
    spec.option.ls_iterations = ls_iterations
    # go2.xml uses the elliptic cone with impratio 100, which needs many solver iterations: with 4 the feet
    # chatter (the robot drifted 0.8 m in 4 s holding its default pose; scripts/mjx/check_contacts.py).
    # The pyramidal cone at impratio 1 (MuJoCo's default, as Playground's MJX models) stands still at 4.
    spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    spec.option.impratio = 1.0
    return spec.compile()


def eval_model(model_dir, sim_dt=0.005):
    m = go2_spec(model_dir).compile()
    m.opt.timestep = sim_dt
    return m


def indices(model):
    """Policy-order qpos addresses, qvel addresses, actuator ids, and torque limits."""
    qadr = np.array([model.joint(n).qposadr[0] for n in POLICY_JOINTS])
    vadr = np.array([model.joint(n).dofadr[0] for n in POLICY_JOINTS])
    act = np.array([model.actuator(n.replace("_joint", "")).id for n in POLICY_JOINTS])
    limit = model.actuator_ctrlrange[act, 1].copy()
    lo, hi = model.jnt_range[[model.joint(n).id for n in POLICY_JOINTS]].T
    return dict(qadr=qadr, vadr=vadr, act=act, limit=limit, lower=lo.copy(), upper=hi.copy())


def geom_ids(model):
    return dict(feet=np.array([model.geom(f).id for f in FEET]),
                knees=np.array([model.geom(f"{f}_knee").id for f in FEET]) if _has(model, "FL_knee") else None,
                floor=model.geom("floor").id, base_body=model.body("base_link").id)


def _has(model, geom_name):
    try:
        model.geom(geom_name)
        return True
    except KeyError:
        return False


def base_corners_local(model_dir):
    """Points on the base and head collision geoms (base frame) whose height is checked for base contact:
    the 8 corners of the base box, and the lowest points of the head cylinder and sphere approximated by
    their centers minus their radii (returned separately as (center, radius) pairs)."""
    spec = go2_spec(model_dir)
    base = spec.body("base_link")
    corners, spheres = [], []
    for g in base.geoms:
        if not _is_collision(g):
            continue
        p = np.array(g.pos, float)
        if g.type == mujoco.mjtGeom.mjGEOM_BOX:
            # go2.xml's base box has no rotation
            for s in itertools.product((-1, 1), repeat=3):
                corners.append(p + np.array(s) * np.array(g.size))
        else:
            spheres.append((p, float(g.size[0])))
    return np.array(corners), spheres


def home_qpos(model, height=0.40):
    """Free joint at the given height, upright, legs at the default pose."""
    ix = indices(model)
    q = model.qpos0.copy()
    q[0:3] = [0.0, 0.0, height]
    q[3:7] = [1.0, 0.0, 0.0, 0.0]
    q[ix["qadr"]] = DEFAULT_POS
    return q


def standing_height(model, clearance=0.002):
    """Base height at which the feet, at the default pose, sit `clearance` above the floor."""
    d = mujoco.MjData(model)
    d.qpos[:] = home_qpos(model, 1.0)
    mujoco.mj_forward(model, d)
    feet = [model.geom(f).id for f in FEET]
    low = min(d.geom_xpos[g, 2] - model.geom_size[g, 0] for g in feet)
    return 1.0 - low + clearance

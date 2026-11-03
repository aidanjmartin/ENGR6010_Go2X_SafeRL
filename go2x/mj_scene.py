"""MuJoCo scenes built on Unitree's Go2 model (unitree_mujoco go2.xml), used where Isaac Sim's RTX renderer cannot
run on the Dell (jobs 040, 045 and 065 crashed in librtx.scenedb at start; see LOG.md) and for the Task 6 check.

`scene.xml` from unitree_mujoco is not used: it places steps and boxes from x = 1.2 m forward, which a robot
walking forward for 20 s reaches. These scenes have a flat floor only, plus whatever the caller adds.

- `go2_spec(model_dir)`: go2.xml + flat floor + light, offscreen buffer sized for rendering.
- `add_box`, `add_cylinder`: visual-only static geoms (no contacts), yaw-only rotation.
- `add_fixed_camera(spec, cam)`: a camera matching go2x.atrium.camera.Camera (pinhole, principal point at the
  image center; MuJoCo cameras look along -z with +y up, the ROS optical frame looks along +z with +y down).
- `set_robot(model, data, xyz, quat_wxyz, joint_pos, joint_names)`: kinematic pose for replay.
"""
import math
import os

import mujoco
import numpy as np

GO2_XML = "unitree_robots/go2/go2.xml"


def go2_spec(model_dir, floor_rgba=(0.62, 0.60, 0.56, 1.0), offscreen=(1280, 720), checker=False):
    spec = mujoco.MjSpec.from_file(os.path.join(model_dir, GO2_XML))
    spec.visual.global_.offwidth, spec.visual.global_.offheight = offscreen
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]
    floor = spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05],
                                    rgba=list(floor_rgba))
    if checker:   # 0.5 m squares, so motion is visible in videos
        spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                         rgb1=list(floor_rgba[:3]), rgb2=[c * 0.8 for c in floor_rgba[:3]], width=64, height=64)
        mat = spec.add_material(name="floor", texrepeat=[1, 1], texuniform=True)
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
        floor.material = "floor"
        floor.rgba = [1, 1, 1, 1]
    spec.worldbody.add_light(name="sun", pos=[0, 0, 6], dir=[0.2, 0.1, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                             diffuse=[0.6, 0.6, 0.6])
    return spec


def _yaw_quat(yaw):
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def add_box(spec, name, center, size_full, yaw, rgb):
    spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=list(map(float, center)),
                            size=[s / 2 for s in size_full], quat=_yaw_quat(float(yaw)), rgba=[*map(float, rgb), 1.0],
                            contype=0, conaffinity=0)


def add_cylinder(spec, name, center, radius, height, rgb):
    spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=list(map(float, center)),
                            size=[float(radius), float(height) / 2, 0], rgba=[*map(float, rgb), 1.0],
                            contype=0, conaffinity=0)


def mj_camera_frame(R_wc):
    """MuJoCo camera xyaxes from a world-from-camera rotation in the ROS optical convention."""
    R = np.asarray(R_wc, float)
    return np.concatenate([R[:, 0], -R[:, 1]])


def add_fixed_camera(spec, cam, name="top"):
    fovy = math.degrees(2 * math.atan(cam.height / 2 / cam.K[1, 1]))
    assert abs(cam.K[0, 2] - cam.width / 2) < 1e-6 and abs(cam.K[1, 2] - cam.height / 2) < 1e-6
    assert abs(cam.K[0, 0] - cam.K[1, 1]) < 1e-6, "MuJoCo cameras have square pixels"
    spec.worldbody.add_camera(name=name, pos=list(map(float, cam.C)), xyaxes=mj_camera_frame(cam.R_wc).tolist(),
                              fovy=fovy)


def joint_addresses(model, joint_names):
    return (np.array([model.joint(n).qposadr[0] for n in joint_names]),
            np.array([model.joint(n).dofadr[0] for n in joint_names]))


def set_robot(model, data, xyz, quat_wxyz, joint_pos, joint_names):
    qadr, _ = joint_addresses(model, joint_names)
    data.qpos[0:3] = xyz
    data.qpos[3:7] = quat_wxyz
    data.qpos[qadr] = joint_pos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def add_moving_part(spec, part):
    """An obstacle part from go2x.atrium.state.object_parts as a mocap body (pose set every frame, no contacts).
    Returns the body name; its mocap index follows the order of the calls."""
    body = spec.worldbody.add_body(name="mocap_" + part["name"], mocap=True, pos=[0, 0, -5])
    rgba = [*map(float, part["color"]), 1.0]
    if part["shape"] == "box":
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[s / 2 for s in part["size"]], rgba=rgba,
                      contype=0, conaffinity=0)
    else:
        r, h = part["size"]
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[r, h / 2, 0], rgba=rgba, contype=0, conaffinity=0)
    return body.name


def add_building(spec, boxes, colors):
    for b in boxes:
        add_box(spec, "bldg_" + b["name"], (b["x"], b["y"], b["z"]), (b["sx"], b["sy"], b["sz"]), b["yaw"],
                colors.get(b["kind"], (0.7, 0.7, 0.7)))


BUILDING_COLORS = {"wall": (0.85, 0.85, 0.82), "glass": (0.55, 0.70, 0.80), "rail": (0.35, 0.35, 0.38),
                   "low": (0.70, 0.68, 0.62), "stairs": (0.60, 0.58, 0.55), "barrier": (0.5, 0.5, 0.5),
                   "closed_door": (0.55, 0.40, 0.30)}   # as go2x/isaac/nav_env_cfg.py

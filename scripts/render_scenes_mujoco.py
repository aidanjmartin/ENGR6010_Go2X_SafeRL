"""Task 3 rendered check with MuJoCo's renderer (Isaac Sim 5.1's RTX renderer crashes on the Dell's driver;
LOG.md, job 041). Same scenes, camera model and output files as scripts/isaac/render_scenes.py.

For each scene: the layout comes from the scene sampler (seed config/camera.yaml rendering.seed); obstacles are
placed with the navigation environment's own part geometry (go2x.atrium.state.object_parts / part_poses); the
Level 1 building boxes near the square are added; Unitree's Go2 model stands at the start pose (PD hold of the
default joint pose for `settle_s`, base x, y and yaw held by the start pose, contacts with obstacles off). The
fixed camera from config/camera.yaml renders the RGB image, saved with the ground truth (object classes, colors,
footprints, robot pose, camera model) and the drawn top-down image of the true state (no camera noise).

    MUJOCO_GL=glfw python scripts/render_scenes_mujoco.py --model_dir <unitree_mujoco> --out $DATA_DIR/scenes
"""
import argparse
import json
import os

import imageio
import mujoco
import numpy as np
import torch

from go2x.atrium.camera import Camera
from go2x.atrium.scene import layout_to_dict, sample_layout
from go2x.atrium.state import AtriumBatch
from go2x.atrium.topdown import draw
from go2x.config import CONFIG_DIR, load_config
from go2x.mj_scene import BUILDING_COLORS, add_box, add_building, add_cylinder, add_fixed_camera, go2_spec, joint_addresses
from go2x.sim2sim import DEFAULT_POS, POLICY_JOINTS, pd_torque

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--settle_s", type=float, default=1.0)
parser.add_argument("--num_scenes", type=int, default=None)
args = parser.parse_args()

ATRIUM, CAM, TD, MJ, WALK = (load_config(n) for n in ("atrium", "camera", "topdown", "mujoco", "walk"))
rc = CAM["rendering"]
n_scenes = args.num_scenes or rc["num_scenes"]
rng = np.random.default_rng(rc["seed"])
layouts = [sample_layout(rng, ATRIUM)[0] for _ in range(n_scenes)]
batch = AtriumBatch(1, "cpu", seed=0, atrium_cfg=ATRIUM, scenarios=[layout_to_dict(L) for L in layouts])
cam = Camera.from_config(CAM)
with open(CONFIG_DIR / "building_level1_near_square.json") as f:
    boxes = json.load(f)["boxes"]
kp, kd = WALK["actuator"]["kp"], WALK["actuator"]["kd"]
os.makedirs(args.out, exist_ok=True)
index = []

for i in range(n_scenes):
    start = batch.reset([0])[0].numpy()             # x, y, yaw of the robot base
    poses = batch.part_poses()[0].numpy()
    spec = go2_spec(args.model_dir, offscreen=(cam.width, cam.height))
    add_building(spec, boxes, BUILDING_COLORS)
    for p, pose in zip(batch.parts, poses):
        if abs(pose[0]) > 3.0 or abs(pose[1]) > 3.0:
            continue                                 # parked slots stay out of the picture
        yaw = 2 * np.arctan2(pose[6], pose[3])
        if p["shape"] == "box":
            add_box(spec, p["name"], pose[:3], p["size"], yaw, p["color"])
        else:
            add_cylinder(spec, p["name"], pose[:3], p["size"][0], p["size"][1], p["color"])
    add_fixed_camera(spec, cam)
    model = spec.compile()
    model.opt.timestep = MJ["timestep"]
    data = mujoco.MjData(model)
    qadr, vadr = joint_addresses(model, POLICY_JOINTS)
    act = np.array([model.actuator(n.replace("_joint", "")).id for n in POLICY_JOINTS])
    limit = model.actuator_ctrlrange[act, 1]
    data.qpos[0:3] = [start[0], start[1], MJ["start_height"]]
    data.qpos[3:7] = [np.cos(start[2] / 2), 0, 0, np.sin(start[2] / 2)]
    data.qpos[qadr] = DEFAULT_POS
    mujoco.mj_forward(model, data)
    for _ in range(int(round(args.settle_s / MJ["timestep"]))):
        data.ctrl[act] = pd_torque(DEFAULT_POS, data.qpos[qadr], data.qvel[vadr], kp, kd, limit)
        mujoco.mj_step(model, data)
    renderer = mujoco.Renderer(model, cam.height, cam.width)
    renderer.update_scene(data, camera="top")
    rgb = renderer.render()
    renderer.close()

    q = data.qpos[3:7]
    yaw = float(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))
    pose_t = torch.tensor([[float(data.qpos[0]), float(data.qpos[1]), yaw]])
    st = batch.perception_state(pose_t)
    top = draw(pose_t, tuple(ATRIUM["robot"]["footprint"]), st["rects"], st["rect_mask"], st["circles"],
               st["circle_mask"], st["target_rect"], st["target_circle"], st["target_is_circle"], TD)[0].numpy()
    L = batch.layouts[0]
    objs = []
    for o in L.objects():
        d = {"id": o["id"], "class": o["kind"], "color": o["color"]}
        if "rect" in o:
            d["footprint_rect"] = [float(v) for v in o["rect"]]
        else:
            k = int(o["id"].split("_")[1])
            d["footprint_circle"] = [float(v) for v in batch.circles[0, k].tolist()]
            d["walking"] = o["walking"]
        objs.append(d)
    name = f"scene_{i:03d}"
    imageio.imwrite(os.path.join(args.out, name + ".png"), rgb)
    np.save(os.path.join(args.out, name + "_topdown.npy"), top.astype(np.float32))
    imageio.imwrite(os.path.join(args.out, name + "_topdown.png"),
                    (np.clip(top[[0, 1, 2]].transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8))
    gt = {"scene": i, "scenario_id": int(batch.scenario_id[0]), "objects": objs, "target": L.target_id,
          "robot_pose_xy_yaw": pose_t[0].tolist(), "robot_fell_or_ended": False,
          "renderer": f"MuJoCo {mujoco.__version__} (Isaac Sim RTX unavailable on the Dell driver)",
          "camera": {"K": cam.K.tolist(), "R_wc": cam.R_wc.tolist(), "C": cam.C.tolist(),
                     "H_floor_to_image": cam.floor_homography().tolist()},
          "frame": "square frame = aeb_level1 world frame (origin at the square center)"}
    with open(os.path.join(args.out, name + ".json"), "w") as f:
        json.dump(gt, f, indent=1)
    index.append({"scene": i, "image": name + ".png", "objects": len(objs), "ended_early": False})
    print(f"[render_scenes_mujoco] {name}: {len(objs)} objects, rgb mean {rgb.mean():.1f}", flush=True)
with open(os.path.join(args.out, "index.json"), "w") as f:
    json.dump(index, f, indent=1)

"""Navigation videos drawn with MuJoCo from the rollouts saved by scripts/isaac/record_nav_videos.py.

Each frame: the fixed camera's view (config/camera.yaml) of Unitree's Go2 model posed at the recorded Isaac state,
the obstacles at their recorded poses and the building boxes near the square; beside it, the drawn top-down
image the controller received (current frame, enlarged, turned 90 degrees to match the camera view; static
obstacles red, people green, robot blue, target yellow). One mp4 per rollout at the navigation rate (10 Hz).

    MUJOCO_GL=glfw python scripts/render_nav_videos.py --model_dir <unitree_mujoco> --rollouts DIR --out DIR
"""
import argparse
import glob
import json
import os

import imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from go2x.atrium.camera import Camera
from go2x.atrium.state import object_parts
from go2x.config import CONFIG_DIR, load_config
from go2x.mj_scene import BUILDING_COLORS, add_building, add_fixed_camera, add_moving_part, go2_spec, set_robot

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", required=True)
parser.add_argument("--rollouts", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

ATRIUM, CAM, NAV = load_config("atrium"), load_config("camera"), load_config("nav")
cam = Camera.from_config(CAM)
parts = object_parts(ATRIUM)
with open(CONFIG_DIR / "building_level1_near_square.json") as f:
    boxes = json.load(f)["boxes"]
spec = go2_spec(args.model_dir, offscreen=(cam.width, cam.height))
add_building(spec, boxes, BUILDING_COLORS)
for p in parts:
    add_moving_part(spec, p)
add_fixed_camera(spec, cam)
model = spec.compile()
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, cam.height, cam.width)
mocap_ids = [model.body("mocap_" + p["name"]).mocapid[0] for p in parts]
COLORS = np.array([[0.9, 0.2, 0.2], [0.2, 0.8, 0.2], [0.2, 0.4, 1.0], [1.0, 0.85, 0.0]])  # channel colors

os.makedirs(args.out, exist_ok=True)
made = []
for path in sorted(glob.glob(os.path.join(args.rollouts, "nav_*.npz"))):
    r = np.load(path)
    names = [str(n) for n in r["joint_names"]]
    frames = []
    for t in range(len(r["topdown"])):
        set_robot(model, data, r["root_pos"][t], r["root_quat_wxyz"][t], r["joint_pos"][t], names)
        for j, m in enumerate(mocap_ids):
            data.mocap_pos[m] = r["part_poses"][t, j, :3]
            data.mocap_quat[m] = r["part_poses"][t, j, 3:7]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="top")
        rgb = renderer.render()
        img = r["topdown"][t, :4].astype(np.float32).clip(0, 1)        # current frame, 4 channels
        top = np.ones((*img.shape[1:], 3))
        for c in range(4):
            top = top * (1 - img[c, :, :, None]) + img[c, :, :, None] * COLORS[c]
        robot_view = str(r["view"]) == "robot" if "view" in r else False
        if not robot_view:
            top = np.rot90(top, k=-1)   # +y right, -x up: the same orientation as the camera view (camera at +x)
        panel = np.kron((top * 255).astype(np.uint8), np.ones((7, 7, 1), np.uint8))
        pad = np.full((rgb.shape[0], panel.shape[1], 3), 255, np.uint8)
        pad[: panel.shape[0]] = panel[: rgb.shape[0]]
        frame = Image.fromarray(np.concatenate([rgb, pad], 1))
        d = ImageDraw.Draw(frame)
        d.rectangle((0, 0, 330, 24), fill=(255, 255, 255))
        d.text((8, 7), f"{r['controller']}  scenario {int(r['scenario'])}  t {t * float(r['dt']):4.1f} s", fill=(0, 0, 0))
        d.text((cam.width + 8, panel.shape[0] + 8), "controller input (current frame), robot view: robot at the center, facing up"
               if robot_view else "controller input (current frame), turned to match the camera view", fill=(0, 0, 0))
        if t == len(r["topdown"]) - 1:
            d.rectangle((0, 24, 330, 46), fill=(255, 255, 255))
            d.text((8, 30), "ended by: " + ", ".join(map(str, r["ended_by"])), fill=(0, 0, 0))
        frames.append(np.asarray(frame))
    frames += [frames[-1]] * int(NAV["timing"]["nav_hz"])                # hold the last frame for 1 s
    out = os.path.join(args.out, os.path.basename(path).replace(".npz", ".mp4"))
    imageio.mimwrite(out, frames, fps=int(NAV["timing"]["nav_hz"]), quality=7, macro_block_size=8)
    made.append({"file": os.path.basename(out), "frames": len(frames), "ended_by": [str(x) for x in r["ended_by"]]})
    print("[render_nav_videos]", made[-1], flush=True)
json.dump(made, open(os.path.join(args.out, "nav_videos_rendered.json"), "w"), indent=1)

"""Video of a recorded Isaac Lab walking trajectory, drawn with MuJoCo's renderer.

Isaac Sim 5.1's RTX renderer crashes on the Dell's driver (LOG.md, job 041), so job 040 records environment 0's
base pose and joint angles every policy step (`eval_walk.py --trajectory`) and this script poses Unitree's Go2
model (unitree_mujoco go2.xml) at each recorded state. No physics runs here: the motion is the Isaac motion.
A chase camera follows the base; the text shows time and the velocity target.

    MUJOCO_GL=glfw python scripts/render_walk_video.py --model_dir <unitree_mujoco> --trajectory traj.npz --out walk_20s.mp4
"""
import argparse

import imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from go2x.mj_scene import go2_spec, set_robot

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", required=True)
parser.add_argument("--trajectory", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--fps", type=int, default=25)
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=544)   # multiple of 16 for the encoder
args = parser.parse_args()

tr = np.load(args.trajectory)
names = [str(n) for n in tr["joint_names"]]
dt = float(tr["dt"])
every = max(1, int(round(1.0 / (dt * args.fps))))
spec = go2_spec(args.model_dir, offscreen=(args.width, args.height), checker=True)
model = spec.compile()
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, args.height, args.width)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.distance, cam.azimuth, cam.elevation = 2.2, 135.0, -18.0

frames, resets = [], 0
for t in range(0, len(tr["done"]), every):
    set_robot(model, data, tr["root_pos"][t], tr["root_quat_wxyz"][t], tr["joint_pos"][t], names)
    cam.lookat[:] = [tr["root_pos"][t][0], tr["root_pos"][t][1], 0.25]
    renderer.update_scene(data, camera=cam)
    img = Image.fromarray(renderer.render())
    resets = int(np.sum(tr["done"][:t]))
    vx, vy, wz = tr["command"][t]
    draw_ = ImageDraw.Draw(img)
    draw_.rectangle((0, 0, args.width, 30), fill=(255, 255, 255))
    draw_.text((12, 10), f"t {t * dt:5.2f} s   target vx {vx:+.2f} m/s  vy {vy:+.2f} m/s  wz {wz:+.2f} rad/s"
                             + (f"   episode resets so far: {resets}" if resets else ""), fill=(0, 0, 0))
    frames.append(np.asarray(img))
imageio.mimwrite(args.out, frames, fps=args.fps, quality=7)
print(f"video {args.out}: {len(frames)} frames at {args.fps} fps, {len(frames) / args.fps:.1f} s, "
      f"first episode fell: {bool(tr['first_episode_fell'])}", flush=True)

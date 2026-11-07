"""Videos of walking policies in CPU MuJoCo on the full go2.xml (physics, not replay), chase camera.

--progress: one clip per policy file, in order (training snapshots), each labeled with its file name, so the
  video shows the policy learning to walk. Each clip runs the same targets (forward 0.6 m/s, then a turn).
--demo: one long run of one policy through a scripted course: forward at 0.3, 0.6 and 1.0 m/s, turning,
  sideways, backward, standing, and base velocity kicks of 0.8 m/s from the side.

    MUJOCO_GL=glfw python scripts/mjx/record_walk.py --model_dir <unitree_mujoco> --policy p1.pt [--policy p2.pt ...]
        --out walk.mp4 (--progress | --demo)
"""
import argparse
import math
import os
import sys
from pathlib import Path

import imageio
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from go2x.config import load_config  # noqa: E402
from go2x.mj_scene import go2_spec  # noqa: E402
from go2x.mjx.go2_model import indices  # noqa: E402
from go2x.sim2sim import joint_targets, observation, quat_rotate_inverse, tilt  # noqa: E402

PROGRESS = [(0.0, (0.6, 0.0, 0.0)), (4.0, (0.4, 0.0, 0.8))]          # (start time s, target)
DEMO = [(0.0, (0.0, 0.0, 0.0)), (1.5, (0.3, 0.0, 0.0)), (4.5, (0.6, 0.0, 0.0)), (7.5, (1.0, 0.0, 0.0)),
        (10.5, (0.5, 0.0, 0.8)), (14.0, (0.5, 0.0, -0.8)), (17.5, (0.0, 0.4, 0.0)), (20.0, (0.0, -0.4, 0.0)),
        (22.5, (-0.4, 0.0, 0.0)), (25.0, (0.6, 0.0, 0.0)), (31.0, (0.0, 0.0, 0.0))]
DEMO_PUSHES = [(27.0, 0.8), (29.0, -0.8)]                              # (time s, lateral kick m/s in the body frame)


def run(model, policy, schedule, duration, pushes, renderer, cam, label, fps, writer, size):
    MJ, WALK = load_config("mujoco"), load_config("walk")
    d = mujoco.MjData(model)
    ix = indices(model)
    qadr, vadr, act, limit = ix["qadr"], ix["vadr"], ix["act"], ix["limit"]
    kp, kd, scale = WALK["actuator"]["kp"], WALK["actuator"]["kd"], WALK["actuator"]["action_scale"]
    gyro_adr, quat_adr = model.sensor("imu_gyro").adr[0], model.sensor("imu_quat").adr[0]
    d.qpos[0:3] = [0, 0, MJ["start_height"]]
    d.qpos[3:7] = [1, 0, 0, 0]
    d.qpos[qadr] = joint_targets(np.zeros(12), scale)
    mujoco.mj_forward(model, d)
    hold = joint_targets(np.zeros(12), scale)
    for _ in range(int(MJ["settle_s"] / MJ["timestep"])):
        d.ctrl[act] = np.clip(kp * (hold - d.qpos[qadr]) - kd * d.qvel[vadr], -limit, limit)
        mujoco.mj_step(model, d)
    dt = MJ["timestep"] * MJ["pd_substeps"]
    last = np.zeros(12, np.float32)
    every = max(1, int(round(1 / (fps * dt))))
    pushes = list(pushes)
    fell = None
    for k in range(int(duration / dt)):
        t = k * dt
        cmd = np.array([c for s, c in schedule if s <= t][-1], float)
        if pushes and t >= pushes[0][0]:
            _, v = pushes.pop(0)
            q = d.qpos[3:7]
            yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
            d.qvel[0:2] += v * np.array([-math.sin(yaw), math.cos(yaw)])
        obs = observation(d.sensordata[gyro_adr:gyro_adr + 3], d.sensordata[quat_adr:quat_adr + 4], cmd,
                          d.qpos[qadr], d.qvel[vadr], last)
        with torch.no_grad():
            last = policy(torch.from_numpy(obs)[None])[0].numpy().astype(np.float32)
        tgt = joint_targets(last, scale)
        for _ in range(MJ["pd_substeps"]):
            d.ctrl[act] = np.clip(kp * (tgt - d.qpos[qadr]) - kd * d.qvel[vadr], -limit, limit)
            mujoco.mj_step(model, d)
        if fell is None and (tilt(d.qpos[3:7]) > MJ["fall"]["tilt_rad"] or d.qpos[2] < MJ["fall"]["base_height_m"]):
            fell = t
        if k % every == 0:
            cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.25]
            renderer.update_scene(d, cam)
            img = Image.fromarray(renderer.render())
            dr = ImageDraw.Draw(img)
            font = ImageFont.load_default(size=18)
            v_b = quat_rotate_inverse(d.qpos[3:7], d.qvel[0:3])
            dr.rectangle([0, 0, size[0], 46], fill=(255, 255, 255))
            dr.text((8, 2), label, fill=(0, 0, 0), font=font)
            dr.text((8, 22), f"t {t:5.1f} s   target vx {cmd[0]:+.2f} vy {cmd[1]:+.2f} m/s wz {cmd[2]:+.2f} rad/s   "
                    f"actual vx {v_b[0]:+.2f} vy {v_b[1]:+.2f} wz {d.qvel[5]:+.2f}"
                    + (f"   FELL at {fell:.1f} s" if fell is not None else ""), fill=(160, 0, 0) if fell is not None else (0, 0, 0), font=font)
            writer.append_data(np.asarray(img))
    return fell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--policy", action="append", required=True)
    ap.add_argument("--label", action="append", default=None, help="one per policy (default: file name)")
    ap.add_argument("--out", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--progress", action="store_true")
    g.add_argument("--demo", action="store_true")
    ap.add_argument("--clip_s", type=float, default=8.0)
    ap.add_argument("--fps", type=int, default=25)
    args = ap.parse_args()
    size = (960, 544)
    spec = go2_spec(args.model_dir, offscreen=size, checker=True)
    model = spec.compile()
    model.opt.timestep = load_config("mujoco")["timestep"]
    renderer = mujoco.Renderer(model, size[1], size[0])
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation, cam.azimuth = 2.2, -20, 135
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    labels = args.label or [Path(p).stem for p in args.policy]
    with imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=7) as w:
        for path, label in zip(args.policy, labels):
            pol = torch.jit.load(path, map_location="cpu").eval()
            if args.progress:
                fell = run(model, pol, PROGRESS, args.clip_s, [], renderer, cam, label, args.fps, w, size)
            else:
                fell = run(model, pol, DEMO, 33.0, DEMO_PUSHES, renderer, cam, label, args.fps, w, size)
            print(f"[record_walk] {label}: {'fell at %.1f s' % fell if fell is not None else 'no fall'}", flush=True)
    print(f"[record_walk] wrote {args.out}")


if __name__ == "__main__":
    main()

"""Digital-twin video of a run folder written by scripts/robot/run_nav_policy.py (robot or MuJoCo).

Only the log is used, so the same script shows a real-robot run: every sport-mode state message (20 Hz) gives one
frame. The base pose is the odometry and IMU yaw mapped into the square frame (the run's own mapping, run.json),
roll and pitch from the IMU, height from body_height; the 12 leg joints come from the nearest low-state message
(the default standing pose if there is none). Obstacles and people are the scenario's, people moved along their
scripts by the time since the policy started. Panels: the fixed camera (config/camera.yaml), a chase camera, and
the policy's input at that time; the header shows time, odometry speed, the command sent and the outcome.
For a camera run (run.json pose_source "camera") the base position and yaw are the poses the policy used (camera
tag, odometry between fixes), interpolated between policy steps, and the obstacles are the last perceived layout.

    python scripts/robot/render_twin.py data/<run folder> --model_dir <unitree_mujoco>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from go2x.atrium.camera import Camera  # noqa: E402
from go2x.atrium.state import object_parts  # noqa: E402
from go2x.config import CONFIG_DIR, load_config  # noqa: E402
from go2x.deploy.mujoco_robot import frame_to_rgb  # noqa: E402
from go2x.deploy.nav_runtime import OdomToSquare, SquareScene  # noqa: E402
from go2x.mj_scene import BUILDING_COLORS, add_building, add_fixed_camera, add_moving_part, go2_spec  # noqa: E402
from go2x.sim2sim import DEFAULT_POS, POLICY_JOINTS, SDK_JOINTS  # noqa: E402


def read_jsonl(path):
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def quat_from_rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r / 2), np.sin(r / 2), np.cos(p / 2), np.sin(p / 2), np.cos(y / 2), np.sin(y / 2)
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--model_dir", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--size", type=int, nargs=2, default=(640, 480))
    args = p.parse_args()
    run_dir = Path(args.run)
    run = json.loads((run_dir / "run.json").read_text())
    if not run.get("mapping"):
        raise SystemExit("this run never reached the policy start, so there is no odometry mapping to draw")
    mapper = OdomToSquare(run["mapping"]["start_pose_square"], run["mapping"]["first_odom"])
    scene = SquareScene(run["scenario"])
    states = read_jsonl(run_dir / "state.jsonl")
    lows = read_jsonl(run_dir / "lowstate.jsonl")
    events = read_jsonl(run_dir / "events.jsonl")
    cmds = read_jsonl(run_dir / "commands.jsonl")
    pol = np.load(run_dir / "policy.npz") if (run_dir / "policy.npz").exists() else None
    t_start = next(e["t_mono_ns"] for e in events if e["event"] == "policy_start")
    low_t = np.array([m["t_mono_ns"] for m in lows]) if lows else None
    cmd_t = np.array([c["t_mono_ns"] for c in cmds]) if cmds else None

    cam_poses = None
    if run.get("pose_source") == "camera" and pol is not None and len(pol["t"]):
        cam_poses = np.array(pol["pose"], float)
        cam_poses[:, 2] = np.unwrap(cam_poses[:, 2])
        t_start = run["t0_mono_ns"]
    W, H = args.size
    cam = Camera.from_config(load_config("camera"))
    spec = go2_spec(args.model_dir, offscreen=(W, H), checker=True)
    with open(CONFIG_DIR / "building_level1_near_square.json") as f:
        add_building(spec, json.load(f)["boxes"], BUILDING_COLORS)
    parts = object_parts(load_config("atrium"))
    for part in parts:
        add_moving_part(spec, part)
    add_fixed_camera(spec, cam)
    model = spec.compile()
    data = mujoco.MjData(model)
    mocap = [model.body("mocap_" + part["name"]).mocapid[0] for part in parts]
    sdk_qadr = np.array([model.joint(n).qposadr[0] for n in SDK_JOINTS])
    pol_qadr = np.array([model.joint(n).qposadr[0] for n in POLICY_JOINTS])
    renderer = mujoco.Renderer(model, H, W)
    chase = mujoco.MjvCamera()
    chase.type = mujoco.mjtCamera.mjCAMERA_FREE
    chase.distance, chase.azimuth, chase.elevation = 2.4, 135.0, -22.0

    import imageio
    out = Path(args.out) if args.out else run_dir / "twin.mp4"
    writer = imageio.get_writer(out, fps=20, quality=7, macro_block_size=8)
    blank = np.full((320, 320, 3), 255, np.uint8)
    outcome = run.get("final", run.get("outcome", ""))
    n = 0
    for k, s in enumerate(states):
        d = s["data"]
        rpy = (d.get("imu_state") or {}).get("rpy") or [0.0, 0.0, 0.0]
        pos = d.get("position") or [0.0, 0.0, 0.0]
        t_rel = (s["t_mono_ns"] - t_start) / 1e9
        if cam_poses is not None:      # camera run: the pose the policy used (camera, odometry between fixes)
            x, y, yaw = (np.interp(t_rel, pol["t"], cam_poses[:, i]) for i in range(3))
        else:
            x, y, yaw = mapper(np.array([pos[0], pos[1], rpy[2]]))
        scene.set_time(max(t_rel, 0.0))
        poses = scene.part_poses()
        for j, mid in enumerate(mocap):
            data.mocap_pos[mid] = poses[j, :3]
            data.mocap_quat[mid] = poses[j, 3:7]
        data.qpos[0:3] = [x, y, float(d.get("body_height") or 0.32)]
        data.qpos[3:7] = quat_from_rpy(rpy[0], rpy[1], yaw)
        if lows:
            i = int(np.clip(np.searchsorted(low_t, s["t_mono_ns"]), 0, len(lows) - 1))
            q = [m["q"] for m in lows[i]["data"]["motor_state"][:12]]
            data.qpos[sdk_qadr] = q
        else:
            data.qpos[pol_qadr] = DEFAULT_POS
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="top")
        a = renderer.render().copy()
        chase.lookat[:] = [x, y, 0.25]
        renderer.update_scene(data, camera=chase)
        b = renderer.render().copy()
        panel = blank
        if pol is not None and len(pol["t"]) and t_rel >= 0:
            j = int(np.clip(np.searchsorted(pol["t"], t_rel, side="right") - 1, 0, len(pol["t"]) - 1))
            panel = frame_to_rgb(pol["frame"][j].astype(np.float32))
        pad = np.full((H, 320, 3), 255, np.uint8)
        pad[: min(H, panel.shape[0]), : panel.shape[1]] = panel[:H, :320]
        img = Image.fromarray(np.concatenate([a, b, pad], 1))
        dr = ImageDraw.Draw(img)
        sent = [0.0, 0.0, 0.0]
        if cmds:
            ci = int(np.searchsorted(cmd_t, s["t_mono_ns"], side="right")) - 1
            if ci >= 0:
                sent = cmds[ci]["sent"]
        v = d.get("velocity") or [0.0, 0.0, 0.0]
        dr.rectangle((0, 0, 2 * W, 40), fill=(255, 255, 255))
        dr.text((8, 5), f"digital twin ({run['backend']} run, {run['source']})   t {t_rel:+6.2f} s from policy start", fill=(0, 0, 0))
        dr.text((8, 22), f"odometry speed vx {v[0]:+.2f} vy {v[1]:+.2f} m/s, yaw rate {float(d.get('yaw_speed') or 0):+.2f} rad/s"
                         f"   command sent vx {sent[0]:+.2f} vy {sent[1]:+.2f} wz {sent[2]:+.2f}", fill=(0, 0, 0))
        dr.text((2 * W + 6, 326), "policy input (robot view)", fill=(0, 0, 0))
        if k == len(states) - 1:
            dr.rectangle((0, 40, 2 * W, 60), fill=(255, 255, 255))
            dr.text((8, 45), f"outcome: {outcome}", fill=(0, 0, 0))
        writer.append_data(np.asarray(img))
        n += 1
    writer.close()
    print(f"twin video {out}: {n} frames at 20 fps ({n / 20:.1f} s), joints from "
          f"{'low state' if lows else 'the default pose (no low state logged)'}")


if __name__ == "__main__":
    main()

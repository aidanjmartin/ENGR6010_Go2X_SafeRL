"""Evaluate the walking policy (Task 1) and log transitions for the learned dynamics model.

One environment per episode (config/walk.yaml `evaluation`), fixed seed, 20 s, a new random velocity target
from the training range every `command_hold_s`. An episode counts until its first termination; a termination
other than the time limit is a fall. Every step of every environment also goes to the transition logger.

    python scripts/isaac/eval_walk.py --policy $DATA_ROOT/030-walk-full/walk/exported/policy.pt \
        --out $RESULTS_DIR --transitions $DATA_DIR/transitions_walk_eval.npz --video $RESULTS_DIR/walk_20s.mp4
"""
import argparse
import json
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--policy", required=True, help="TorchScript policy exported by train_walk.py")
parser.add_argument("--out", required=True)
parser.add_argument("--transitions", required=True)
parser.add_argument("--video", default=None, help="mp4 path; records environment 0")
parser.add_argument("--video_fps", type=int, default=25)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
if args.video:
    args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils.math import euler_xyz_from_quat  # noqa: E402

from go2x.isaac.walk_env_cfg import WALK, WalkEvalEnvCfg  # noqa: E402
from go2x.transitions import TransitionLogger  # noqa: E402

ev = WALK["evaluation"]
cfg = WalkEvalEnvCfg()
cfg.seed = int(ev["seeds"][0])
if args.video:
    cfg.viewer.origin_type = "asset_root"
    cfg.viewer.asset_name = "robot"
    cfg.viewer.env_index = 0
    cfg.viewer.eye = (1.8, 1.8, 1.0)
    cfg.viewer.lookat = (0.0, 0.0, 0.25)
    cfg.viewer.resolution = (960, 540)
env = ManagerBasedRLEnv(cfg=cfg, render_mode="rgb_array" if args.video else None)
policy = torch.jit.load(args.policy, map_location=env.device).eval()

robot = env.scene["robot"]
sensor = env.scene["contact_forces"]
foot_ids, foot_names = sensor.find_bodies(".*_foot")
B, dt = env.num_envs, env.step_dt
steps = int(round(ev["episode_s"] / dt))
settle = int(round(ev["settle_s"] / dt))
limits = robot.data.joint_pos_limits  # (B, 12, 2), hard limits
margin = ev["joint_limit_margin_rad"]
logger = TransitionLogger(dt, WALK["transitions"]["horizons_s"], B)


def base_state():
    pos = robot.data.root_pos_w - env.scene.env_origins
    r, p, y = euler_xyz_from_quat(robot.data.root_quat_w)
    wrap = lambda a: torch.atan2(torch.sin(a), torch.cos(a))  # noqa: E731
    return torch.cat([pos, torch.stack([wrap(r), wrap(p), wrap(y)], 1), robot.data.root_lin_vel_b,
                      robot.data.root_ang_vel_b], 1)


obs, _ = env.reset()
alive = torch.ones(B, dtype=torch.bool, device=env.device)
fell = torch.zeros(B, dtype=torch.bool, device=env.device)
alive_steps = torch.zeros(B, device=env.device)
err_sum = torch.zeros(B, 3, device=env.device)
err_n = torch.zeros(B, device=env.device)
torque_sum = torch.zeros(B, device=env.device)
limit_hits = torch.zeros(B, device=env.device)      # steps with any joint at a limit
contact_sum = torch.zeros(B, len(foot_ids), device=env.device)
since_change = torch.zeros(B, device=env.device)
prev_cmd = env.command_manager.get_command("base_velocity").clone()
new_episode = torch.zeros(B, dtype=torch.bool)
frames, render_every = [], max(1, int(round(1.0 / (dt * args.video_fps))))

with torch.inference_mode():
    for t in range(steps):
        cmd = env.command_manager.get_command("base_velocity").clone()
        since_change = torch.where(torch.any(cmd != prev_cmd, 1), torch.zeros_like(since_change), since_change + 1)
        prev_cmd = cmd
        logger.record(base_state().cpu().numpy(), cmd.cpu().numpy(), new_episode.numpy())

        obs, _, terminated, truncated, _ = env.step(policy(obs["policy"]))
        done = terminated | truncated
        new_episode = done.cpu()

        a = alive & ~done  # still in its first episode after this step (the state below is not a reset state)
        vel = torch.stack([robot.data.root_lin_vel_b[:, 0], robot.data.root_lin_vel_b[:, 1],
                           robot.data.root_ang_vel_b[:, 2]], 1)
        track = a & (since_change >= settle)
        err_sum += torch.where(track[:, None], (vel - cmd).abs(), torch.zeros_like(vel))
        err_n += track.float()
        torque_sum += torch.where(a, robot.data.applied_torque.abs().mean(1), torch.zeros(B, device=env.device))
        q = robot.data.joint_pos
        at_limit = ((q - limits[..., 0]) < margin) | ((limits[..., 1] - q) < margin)
        limit_hits += (a & at_limit.any(1)).float()
        f = sensor.data.net_forces_w[:, foot_ids].norm(dim=-1)
        contact_sum += torch.where(a[:, None], (f > ev["contact_force_n"]).float(), torch.zeros_like(f))
        alive_steps += a.float()
        fell |= alive & terminated & ~truncated
        alive &= ~done
        if args.video and t % render_every == 0:
            frames.append(env.render())

T_alive = alive_steps.clamp(min=1)
per_ep = {
    "abs_err_vx": (err_sum[:, 0] / err_n.clamp(min=1)),
    "abs_err_vy": (err_sum[:, 1] / err_n.clamp(min=1)),
    "abs_err_wz": (err_sum[:, 2] / err_n.clamp(min=1)),
    "mean_abs_torque_Nm": torque_sum / T_alive,
    "joint_limit_step_fraction": limit_hits / T_alive,
}
for i, name in enumerate(foot_names):
    per_ep[f"contact_fraction_{name}"] = contact_sum[:, i] / T_alive
per_ep = {k: v.cpu().numpy() for k, v in per_ep.items()}
minutes = float(alive_steps.sum()) * dt / 60.0
falls = int(fell.sum())
summary = {k: {"mean": float(v.mean()), "se": float(v.std(ddof=1) / math.sqrt(len(v)))} for k, v in per_ep.items()}
summary["falls"] = falls
summary["episodes"] = B
summary["robot_minutes"] = minutes
summary["falls_per_minute"] = {"mean": falls / minutes, "se": math.sqrt(max(falls, 1)) / minutes,
                               "se_note": "Poisson, sqrt(max(falls, 1)) / minutes"}
summary["policy"] = args.policy
summary["seed"] = cfg.seed
summary["settings"] = {k: ev[k] for k in ("episode_s", "command_hold_s", "settle_s", "joint_limit_margin_rad",
                                          "contact_force_n")}
summary["command_range"] = WALK["commands"]["train_range"]
shapes = logger.save(args.transitions, policy=args.policy, seed=cfg.seed)
summary["transitions"] = {"path": args.transitions, "shapes": {k: list(v) for k, v in shapes.items()}}

os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, "walk_eval.json"), "w") as f:
    json.dump(summary, f, indent=1)
rows = [("velocity tracking error, vx (m/s)", "abs_err_vx"), ("velocity tracking error, vy (m/s)", "abs_err_vy"),
        ("velocity tracking error, wz (rad/s)", "abs_err_wz"), ("mean joint torque magnitude (N m)", "mean_abs_torque_Nm"),
        ("fraction of steps with a joint within 0.01 rad of a limit", "joint_limit_step_fraction")]
rows += [(f"contact fraction, {n}", f"contact_fraction_{n}") for n in foot_names]
lines = [f"Walking policy evaluation: {B} episodes of {ev['episode_s']:g} s, seed {cfg.seed}, a new velocity target "
         f"every {ev['command_hold_s']:g} s from vx {WALK['commands']['train_range']['vx']}, vy "
         f"{WALK['commands']['train_range']['vy']}, wz {WALK['commands']['train_range']['wz']}. Tracking error is the mean "
         f"absolute error per episode, skipping {ev['settle_s']:g} s after each new target. Mean and standard error over "
         f"episodes.", "", "| metric | mean | standard error |", "| :--- | ---: | ---: |"]
lines += [f"| {label} | {summary[k]['mean']:.4f} | {summary[k]['se']:.4f} |" for label, k in rows]
lines += [f"| falls per minute ({falls} falls in {minutes:.1f} robot-minutes) | {summary['falls_per_minute']['mean']:.4f} | "
          f"{summary['falls_per_minute']['se']:.4f} |"]
with open(os.path.join(args.out, "walk_eval.md"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines), flush=True)

if args.video and frames:
    import imageio

    imageio.mimwrite(args.video, [np.asarray(fr)[..., :3] for fr in frames], fps=args.video_fps, quality=6)
    print("video", args.video, len(frames), "frames", flush=True)
env.close()
app.close()

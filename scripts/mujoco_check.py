"""Task 6: run the walking policy in MuJoCo on the same velocity targets as the Isaac evaluation (job 040).

Model: unitree_mujoco Go2 (config/mujoco.yaml), torque motors in SDK order. Conventions: docs/LOWLEVEL_INTERFACE.md
via go2x/sim2sim.py. PD at every physics step (kp 25, kd 0.5, torque clipped to the model's ctrlrange).

    python scripts/mujoco_check.py --model_dir <unitree_mujoco> --policy policy.pt \
        --commands walk_eval_commands.npz --isaac_eval walk_eval.json --out DIR
"""
import argparse
import json
import math
import os

import mujoco
import numpy as np
import torch

from go2x.config import load_config
from go2x.sim2sim import JOINT_IDS_MAP, POLICY_JOINTS, SDK_JOINTS, joint_targets, observation, pd_torque, quat_rotate_inverse, tilt

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", required=True)
parser.add_argument("--policy", required=True)
parser.add_argument("--commands", required=True)
parser.add_argument("--isaac_eval", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

MJ, WALK = load_config("mujoco"), load_config("walk")
model = mujoco.MjModel.from_xml_path(os.path.join(args.model_dir, MJ["model"]["scene"]))
model.opt.timestep = MJ["timestep"]
data = mujoco.MjData(model)
policy = torch.jit.load(args.policy, map_location="cpu").eval()
cmd_npz = np.load(args.commands)
commands = cmd_npz["commands"]                     # (steps, episodes, 3)
dt_policy = float(cmd_npz["dt"])
assert abs(dt_policy - MJ["timestep"] * MJ["pd_substeps"]) < 1e-9
steps, episodes, _ = commands.shape
ev = WALK["evaluation"]
settle = int(round(ev["settle_s"] / dt_policy))
kp, kd, scale = WALK["actuator"]["kp"], WALK["actuator"]["kd"], WALK["actuator"]["action_scale"]

qadr = np.array([model.joint(n).qposadr[0] for n in POLICY_JOINTS])
vadr = np.array([model.joint(n).dofadr[0] for n in POLICY_JOINTS])
act_id = np.array([model.actuator(n.replace("_joint", "")).id for n in POLICY_JOINTS])
limit = model.actuator_ctrlrange[act_id, 1]
gyro_adr = model.sensor("imu_gyro").adr[0]
quat_adr = model.sensor("imu_quat").adr[0]


def reset():
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0.0, 0.0, MJ["start_height"]]
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qpos[qadr] = joint_targets(np.zeros(12), scale)
    mujoco.mj_forward(model, data)
    target = joint_targets(np.zeros(12), scale)
    for _ in range(int(round(MJ["settle_s"] / MJ["timestep"]))):
        data.ctrl[act_id] = pd_torque(target, data.qpos[qadr], data.qvel[vadr], kp, kd, limit)
        mujoco.mj_step(model, data)


per_ep = {"abs_err_vx": [], "abs_err_vy": [], "abs_err_wz": [], "mean_abs_torque_Nm": []}
falls, alive_steps_total = 0, 0
fell_list = []
for e in range(episodes):
    reset()
    last = np.zeros(12, np.float32)
    err, n_err, torque_sum, since, prev_cmd = np.zeros(3), 0, 0.0, 0, None
    fell, t_alive = False, 0
    for t in range(steps):
        cmd = commands[t, e]
        since = 0 if prev_cmd is None or np.any(cmd != prev_cmd) else since + 1
        prev_cmd = cmd
        quat = data.sensordata[quat_adr:quat_adr + 4].copy()
        obs = observation(data.sensordata[gyro_adr:gyro_adr + 3], quat, cmd, data.qpos[qadr], data.qvel[vadr], last)
        with torch.no_grad():
            last = policy(torch.from_numpy(obs)[None])[0].numpy().astype(np.float32)
        target = joint_targets(last, scale)
        tq = 0.0
        for _ in range(MJ["pd_substeps"]):
            tau = pd_torque(target, data.qpos[qadr], data.qvel[vadr], kp, kd, limit)
            data.ctrl[act_id] = tau
            mujoco.mj_step(model, data)
            tq += np.abs(tau).mean()
        torque_sum += tq / MJ["pd_substeps"]
        t_alive += 1
        q = data.qpos[3:7]
        v_b = quat_rotate_inverse(q, data.qvel[0:3])
        vel = np.array([v_b[0], v_b[1], data.qvel[5]])     # free-joint angular velocity is in the body frame
        if since >= settle:
            err += np.abs(vel - cmd)
            n_err += 1
        if tilt(q) > MJ["fall"]["tilt_rad"] or data.qpos[2] < MJ["fall"]["base_height_m"]:
            fell = True
            break
    for i, k in enumerate(("abs_err_vx", "abs_err_vy", "abs_err_wz")):
        per_ep[k].append(err[i] / max(n_err, 1))
    per_ep["mean_abs_torque_Nm"].append(torque_sum / max(t_alive, 1))
    falls += fell
    fell_list.append(bool(fell))
    alive_steps_total += t_alive
    print(f"[mujoco] episode {e}: {'fell' if fell else 'ok'} after {t_alive * dt_policy:.1f} s, "
          f"err vx {per_ep['abs_err_vx'][-1]:.3f} vy {per_ep['abs_err_vy'][-1]:.3f} wz {per_ep['abs_err_wz'][-1]:.3f}", flush=True)

minutes = alive_steps_total * dt_policy / 60
st = lambda v: {"mean": float(np.mean(v)), "se": float(np.std(v, ddof=1) / math.sqrt(len(v)))}  # noqa: E731
summary = {k: st(v) for k, v in per_ep.items()}
summary.update(episodes=episodes, falls=int(falls), robot_minutes=minutes,
               falls_per_minute={"mean": falls / minutes, "se": math.sqrt(max(falls, 1)) / minutes},
               model=MJ["model"], mujoco=mujoco.__version__, timestep=MJ["timestep"], policy=args.policy)
isaac = json.load(open(args.isaac_eval))
isaac_fell = cmd_npz["fell"].tolist() if "fell" in cmd_npz else None
rows = [("velocity tracking error, vx (m/s)", "abs_err_vx"), ("velocity tracking error, vy (m/s)", "abs_err_vy"),
        ("velocity tracking error, wz (rad/s)", "abs_err_wz"), ("mean joint torque magnitude (N m)", "mean_abs_torque_Nm"),
        ("falls per minute", "falls_per_minute")]
lines = [f"Walking policy, Isaac Lab vs. MuJoCo {mujoco.__version__} (unitree_mujoco Go2), same {episodes} episodes and "
         f"velocity targets. Mean and standard error over episodes; falls per minute with a Poisson standard error.", "",
         "| metric | Isaac Lab | MuJoCo |", "| :--- | ---: | ---: |"]
for label, k in rows:
    lines.append(f"| {label} | {isaac[k]['mean']:.4f} ± {isaac[k]['se']:.4f} | {summary[k]['mean']:.4f} ± {summary[k]['se']:.4f} |")
lines.append(f"| falls | {isaac['falls']} | {falls} |")
os.makedirs(args.out, exist_ok=True)
json.dump({"summary": summary, "per_episode": {**{k: list(map(float, v)) for k, v in per_ep.items()}, "fell": fell_list},
           "isaac_fell": isaac_fell}, open(os.path.join(args.out, "mujoco_check.json"), "w"), indent=1)
open(os.path.join(args.out, "mujoco_check.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
_ = (SDK_JOINTS, JOINT_IDS_MAP)

"""Walking policies in CPU MuJoCo on Unitree's full go2.xml (every collision geom, elliptic cone, 100 solver
iterations), under a set of test conditions. Same conventions as scripts/mujoco_check.py and the MuJoCo robot
backend (go2x/deploy/mujoco_robot.py): 45-value observation, PD torque at every 5 ms physics step, no delay.

Conditions (every policy sees the same episodes and velocity targets, from job 040, 100 episodes of 20 s):
  nominal                go2.xml as is (foot friction 0.4)
  friction_0.2/0.8       foot friction
  payload_3kg/5kg        mass added to the base
  weak_motors_0.8        every joint torque x 0.8
  kp_0.8                 PD stiffness x 0.8
  latency_20ms/40ms      joint targets applied 4 / 8 physics steps late
  obs_noise              Isaac's observation noise ranges (config/walk.yaml obs_noise)
  push_1.0/1.5           base velocity kick of 1.0 / 1.5 m/s in a random horizontal direction every 4 s
  stairs_step_5cm        random 5 cm high boxes on the floor (the robot walks into them; not seen in training)

Metrics per episode: tracking error |v - v_cmd| for vx, vy (m/s) and wz (rad/s) after a 0.5 s settle following
each new target; fell (tilt above 0.8 rad or base below 0.15 m); mean |torque| (N m); mean mechanical power
sum |tau * dq| (W). Mean and standard error over episodes; falls with a binomial standard error.

    python scripts/mjx/eval_walk.py --model_dir <unitree_mujoco> --commands walk_eval_commands.npz \
        --policy isaac=<policy.pt> --policy mjx_s0=<policy.pt> ... --out DIR [--episodes 100] [--workers 20]
"""
import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CONDITIONS = {
    "nominal": {},
    "friction_0.2": {"friction": 0.2},
    "friction_0.8": {"friction": 0.8},
    "payload_3kg": {"payload": 3.0},
    "payload_5kg": {"payload": 5.0},
    "weak_motors_0.8": {"strength": 0.8},
    "kp_0.8": {"kp_scale": 0.8},
    "latency_20ms": {"delay": 4},
    "latency_40ms": {"delay": 8},
    "obs_noise": {"noise": True},
    "push_1.0": {"push": 1.0},
    "push_1.5": {"push": 1.5},
    "steps_5cm": {"steps": 0.05},
}


def build_model(model_dir, cond, seed):
    import mujoco

    from go2x.mj_scene import go2_spec
    spec = go2_spec(model_dir)
    if "steps" in cond:
        rng = np.random.default_rng(seed)
        for i in range(60):
            x, y = rng.uniform(-6, 6, 2)
            if math.hypot(x, y) < 0.6:
                continue
            spec.worldbody.add_geom(name=f"step{i}", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[x, y, cond["steps"] / 2],
                                    size=[rng.uniform(0.2, 0.6), rng.uniform(0.2, 0.6), cond["steps"] / 2],
                                    quat=[math.cos(t := rng.uniform(0, math.pi)), 0, 0, math.sin(t)],
                                    rgba=[0.5, 0.45, 0.4, 1])
    m = spec.compile()
    m.opt.timestep = 0.005
    if "friction" in cond:
        for f in ("FL", "FR", "RL", "RR"):
            m.geom_friction[m.geom(f).id, 0] = cond["friction"]
    if "payload" in cond:
        m.body_mass[m.body("base_link").id] += cond["payload"]
    return m


def run_episodes(job):
    import mujoco
    import torch

    from go2x.config import load_config
    from go2x.sim2sim import joint_targets, observation, quat_rotate_inverse, tilt
    from go2x.mjx.go2_model import indices
    torch.set_num_threads(1)
    model_dir, policy_path, cond_name, commands, dt_policy, ep_ids = job
    cond = CONDITIONS[cond_name]
    MJ, WALK = load_config("mujoco"), load_config("walk")
    m = build_model(model_dir, cond, seed=1234)
    d = mujoco.MjData(m)
    ix = indices(m)
    qadr, vadr, act, limit = ix["qadr"], ix["vadr"], ix["act"], ix["limit"]
    policy = torch.jit.load(policy_path, map_location="cpu").eval()
    kp = WALK["actuator"]["kp"] * cond.get("kp_scale", 1.0)
    kd, scale = WALK["actuator"]["kd"], WALK["actuator"]["action_scale"]
    strength, delay = cond.get("strength", 1.0), cond.get("delay", 0)
    noise = WALK["randomization"]["obs_noise"]
    settle = int(round(WALK["evaluation"]["settle_s"] / dt_policy))
    gyro_adr, quat_adr = m.sensor("imu_gyro").adr[0], m.sensor("imu_quat").adr[0]
    sub = MJ["pd_substeps"]
    out = []
    for e in ep_ids:
        rng = np.random.default_rng(10_000 + e)
        mujoco.mj_resetData(m, d)
        d.qpos[0:3] = [0.0, 0.0, MJ["start_height"]]
        d.qpos[3:7] = [1, 0, 0, 0]
        d.qpos[qadr] = joint_targets(np.zeros(12), scale)
        mujoco.mj_forward(m, d)
        target = joint_targets(np.zeros(12), scale)
        for _ in range(int(round(MJ["settle_s"] / MJ["timestep"]))):
            d.ctrl[act] = np.clip(kp * (target - d.qpos[qadr]) - kd * d.qvel[vadr], -limit, limit)
            mujoco.mj_step(m, d)
        last = np.zeros(12, np.float32)
        queue = [target.copy() for _ in range(delay + 1)]
        err, n_err, tq, pw, since, prev_cmd, fell, alive = np.zeros(3), 0, 0.0, 0.0, 0, None, False, 0
        push_every = int(round(4.0 / dt_policy))
        for t in range(commands.shape[0]):
            cmd = commands[t, e]
            since = 0 if prev_cmd is None or np.any(cmd != prev_cmd) else since + 1
            prev_cmd = cmd
            if "push" in cond and t > 0 and t % push_every == 0:
                a = rng.uniform(0, 2 * math.pi)
                d.qvel[0:2] += cond["push"] * np.array([math.cos(a), math.sin(a)])
            gyro = d.sensordata[gyro_adr:gyro_adr + 3].copy()
            quat = d.sensordata[quat_adr:quat_adr + 4].copy()
            q, dq = d.qpos[qadr].copy(), d.qvel[vadr].copy()
            if cond.get("noise"):
                gyro += rng.uniform(-1, 1, 3) * noise["base_ang_vel"]
                q += rng.uniform(-1, 1, 12) * noise["joint_pos"]
                dq += rng.uniform(-1, 1, 12) * noise["joint_vel"]
            obs = observation(gyro, quat, cmd, q, dq, last)
            if cond.get("noise"):
                obs[3:6] += rng.uniform(-1, 1, 3).astype(np.float32) * noise["projected_gravity"]
            with torch.no_grad():
                last = policy(torch.from_numpy(obs)[None])[0].numpy().astype(np.float32)
            new_target = joint_targets(last, scale)
            for _ in range(sub):
                queue.append(new_target)
                tgt = queue.pop(0)
                tau = strength * np.clip(kp * (tgt - d.qpos[qadr]) - kd * d.qvel[vadr], -limit, limit)
                d.ctrl[act] = tau
                mujoco.mj_step(m, d)
            tq += np.abs(tau).mean()
            pw += np.abs(tau * d.qvel[vadr]).sum()
            alive += 1
            v_b = quat_rotate_inverse(d.qpos[3:7], d.qvel[0:3])
            vel = np.array([v_b[0], v_b[1], d.qvel[5]])
            if since >= settle:
                err += np.abs(vel - cmd)
                n_err += 1
            if tilt(d.qpos[3:7]) > MJ["fall"]["tilt_rad"] or d.qpos[2] < MJ["fall"]["base_height_m"]:
                fell = True
                break
        out.append({"episode": int(e), "fell": fell, "alive_s": alive * dt_policy,
                    "abs_err_vx": err[0] / max(n_err, 1), "abs_err_vy": err[1] / max(n_err, 1),
                    "abs_err_wz": err[2] / max(n_err, 1), "torque": tq / max(alive, 1), "power": pw / max(alive, 1)})
    return policy_path, cond_name, out


def summarize(rows):
    st = lambda v: {"mean": float(np.mean(v)), "se": float(np.std(v, ddof=1) / math.sqrt(len(v)))}  # noqa: E731
    n = len(rows)
    falls = sum(r["fell"] for r in rows)
    p = falls / n
    s = {k: st([r[k] for r in rows]) for k in ("abs_err_vx", "abs_err_vy", "abs_err_wz", "torque", "power")}
    s.update(episodes=n, falls=falls, fall_rate={"mean": p, "se": math.sqrt(max(p * (1 - p), 1 / n ** 2) / n)})
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--commands", required=True)
    ap.add_argument("--policy", action="append", required=True, help="name=path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()
    cmd = np.load(args.commands)
    commands, dt_policy = cmd["commands"], float(cmd["dt"])
    pols = dict(p.split("=", 1) for p in args.policy)
    conds = args.conditions.split(",")
    eps = np.arange(min(args.episodes, commands.shape[1]))
    chunks = np.array_split(eps, max(1, len(eps) // 10))
    jobs = [(args.model_dir, path, c, commands, dt_policy, ch) for path in pols.values() for c in conds for ch in chunks]
    res = {n: {c: [] for c in conds} for n in pols}
    name_of = {v: k for k, v in pols.items()}
    with ProcessPoolExecutor(args.workers) as ex:
        for i, (path, c, rows) in enumerate(ex.map(run_episodes, jobs)):
            res[name_of[path]][c] += rows
            if i % 20 == 0:
                print(f"[eval_walk] {i + 1}/{len(jobs)} chunks", flush=True)
    summary = {n: {c: summarize(res[n][c]) for c in conds} for n in pols}
    os.makedirs(args.out, exist_ok=True)
    json.dump({"policies": pols, "summary": summary, "per_episode": res}, open(os.path.join(args.out, "eval_walk.json"), "w"), indent=1)
    lines = [f"Walking policies in CPU MuJoCo on the full go2.xml, {len(eps)} episodes of 20 s per condition (job 040's "
             "velocity targets). Fall rate with binomial standard error; tracking errors mean ± standard error over episodes.", ""]
    lines += ["| condition | " + " | ".join(pols) + " |", "| :--- | " + " | ".join("---:" for _ in pols) + " |"]
    for c in conds:
        lines.append(f"| {c}: falls | " + " | ".join(f"{summary[n][c]['falls']}/{len(eps)}" for n in pols) + " |")
    for metric, label in (("abs_err_vx", "vx error (m/s)"), ("abs_err_wz", "wz error (rad/s)"), ("power", "power (W)")):
        lines += ["", f"**{label}**", "", "| condition | " + " | ".join(pols) + " |",
                  "| :--- | " + " | ".join("---:" for _ in pols) + " |"]
        for c in conds:
            lines.append(f"| {c} | " + " | ".join(f"{summary[n][c][metric]['mean']:.3f} ± {summary[n][c][metric]['se']:.3f}"
                                                  for n in pols) + " |")
    open(os.path.join(args.out, "eval_walk.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

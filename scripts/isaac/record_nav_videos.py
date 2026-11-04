"""Rollouts for the navigation videos: one success and one failure per controller (Task 4).

Reads the evaluation results (nav_eval_<controller>.json) and replays the chosen scenarios. Isaac Sim 5.1's RTX
renderer crashes on the Dell's driver (LOG.md, job 041), so this script runs without a camera and saves, for
every navigation step, the robot base pose and joint angles, every obstacle part pose, and the drawn top-down
input. scripts/render_nav_videos.py draws the frames from the fixed camera with MuJoCo.

    python scripts/isaac/record_nav_videos.py --walk_policy policy.pt --checkpoint model_final.pt \
        --evals nav_eval_*.json --out DIR
"""
import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--walk_policy", required=True)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--evals", nargs="+", required=True, help="nav_eval_<controller>.json files")
parser.add_argument("--out", required=True)
parser.add_argument("--controllers", nargs="+", default=None, help="only these controllers (one image view per run)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
os.environ["GO2X_WALK_POLICY"] = os.path.abspath(args.walk_policy)
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

from go2x.atrium.baselines import AStarController, StraightController  # noqa: E402
from go2x.atrium.rewards import normalize_action  # noqa: E402
from go2x.config import CONFIG_DIR  # noqa: E402
from go2x.isaac.nav_env_cfg import NAV, TD, AtriumNavEnvCfg, get_runtime, nav_agent_dict  # noqa: E402
from go2x.isaac.shutdown import finish  # noqa: E402

scenarios = json.load(open(CONFIG_DIR / "nav_eval_scenarios.json"))["scenarios"]
jobs = []
for path in args.evals:
    d = json.load(open(path))
    ctl = d["summary"]["controller"]
    if args.controllers and ctl not in args.controllers:
        continue
    eps = d["episodes"]
    ok = [e["scenario"] for e in eps if e["success"]]
    bad = [e["scenario"] for e in eps if not e["success"]]
    for label, ids in (("success", ok), ("failure", bad)):
        for sid in ids[:5]:            # a replay may end differently (camera noise, randomization): try up to 5
            jobs.append((ctl, label, sid))
print("[videos]", jobs, flush=True)

cfg = AtriumNavEnvCfg()
cfg.scene.num_envs = 1
cfg.go2x_scenarios = [scenarios[s] for _, _, s in jobs]
kinds = {ctl == "policy" for ctl, _, _ in jobs}
assert len(kinds) == 1, "run the policy and the baselines separately: they use different image views"
if not kinds.pop():
    cfg.go2x_image_view = "world"
env = ManagerBasedRLEnv(cfg=cfg)
wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
S = TD["size"]
policy = None
if args.checkpoint:
    from rsl_rl.runners import OnPolicyRunner

    from go2x.nav_policy import register_with_rsl_rl

    register_with_rsl_rl()
    agent = json.load(open(os.path.join(os.path.dirname(args.checkpoint), "agent.json")))
    runner = OnPolicyRunner(wrapped, nav_agent_dict(agent), log_dir=None, device=str(env.device))
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.device)

os.makedirs(args.out, exist_ok=True)
obs = wrapped.get_observations()   # the wrapper's reset loaded jobs[0]
summary, done_labels = [], set()
for n, (ctl, label, sid) in enumerate(jobs):
    if (ctl, label) in done_labels:
        continue
    rt = get_runtime(env)
    # an episode end auto-resets into the next scenario; reset further only until scenario n is loaded
    # inside inference mode: the rollout below creates the state tensors there, and reset writes to them
    with torch.inference_mode():
        while int(rt.batch.scenario_id[0]) < n:
            env.reset()
            obs = wrapped.get_observations()
    assert int(rt.batch.scenario_id[0]) == n, "scenario order"
    c = {"straight": StraightController, "astar": AStarController}.get(ctl)
    c = c() if c else None
    rec = {k: [] for k in ("root_pos", "root_quat_wxyz", "joint_pos", "part_poses", "topdown")}
    with torch.inference_mode():
        for t in range(int(env.max_episode_length)):
            img = obs["policy"][:, : 4 * S * S].reshape(1, 4, S, S)
            if ctl == "policy":
                if policy is None:
                    break
                act = policy(obs)
            else:
                act = normalize_action(torch.tensor(c.act(img[0].cpu().numpy()), dtype=torch.float32,
                                                    device=env.device)[None], NAV["action"])
            robot = rt.robot
            rec["root_pos"].append((robot.data.root_pos_w[0] - env.scene.env_origins[0]).cpu().numpy())
            rec["root_quat_wxyz"].append(robot.data.root_quat_w[0].cpu().numpy())
            rec["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy())
            rec["part_poses"].append(rt.batch.part_poses()[0].cpu().numpy())
            rec["topdown"].append(img[0].cpu().numpy().astype(np.float16))
            obs, _, dones, _ = wrapped.step(act)
            if bool(dones[0]):
                break
    outcome = "success" if bool(env.termination_manager.get_term("success")[0]) else "failure"
    reasons = [k for k in ("success", "collision_footprint", "collision_force", "fall", "out_of_square", "time_out")
               if bool(env.termination_manager.get_term(k)[0])]
    if outcome != label or not rec["topdown"]:
        print(f"[videos] {ctl} scenario {sid}: replay ended in {reasons}, wanted {label}; trying the next", flush=True)
        continue
    path = os.path.join(args.out, f"nav_{ctl}_{label}_scenario{sid:03d}.npz")
    np.savez_compressed(path, **{k: np.stack(v) for k, v in rec.items()}, joint_names=np.array(rt.robot.data.joint_names),
                        controller=ctl, label=label, scenario=sid, ended_by=np.array(reasons), view=cfg.go2x_image_view,
                        dt=float(env.step_dt))
    done_labels.add((ctl, label))
    summary.append({"controller": ctl, "label": label, "scenario": sid, "frames": len(rec["topdown"]), "ended_by": reasons,
                    "file": os.path.basename(path)})
    print("[videos]", summary[-1], flush=True)
tag = "_".join(sorted({c for c, _, _ in jobs}))
json.dump(summary, open(os.path.join(args.out, f"nav_videos_{tag}.json"), "w"), indent=1)
finish(0)

"""Evaluate a navigation controller on the fixed scenarios (config/nav_eval_scenarios.json).

Controllers: `policy` (RSL-RL checkpoint), `straight` and `astar` (go2x/atrium/baselines.py). All three send
velocity targets to the same frozen walking policy and see the same drawn image (with camera realism).

    python scripts/isaac/eval_nav.py --controller policy --checkpoint model_final.pt --walk_policy policy.pt --out DIR
"""
import argparse
import json
import math
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--controller", choices=["policy", "straight", "astar"], required=True)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--walk_policy", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N scenarios")
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

with open(CONFIG_DIR / "nav_eval_scenarios.json") as f:
    scenarios = json.load(f)["scenarios"]
if args.limit:
    scenarios = scenarios[: args.limit]
N = len(scenarios)
ev = NAV["evaluation"]
cfg = AtriumNavEnvCfg()
cfg.scene.num_envs = min(ev["num_envs"], N)
cfg.seed = ev["seed"]
cfg.go2x_scenarios = scenarios
env = ManagerBasedRLEnv(cfg=cfg)
wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
B, S = env.num_envs, TD["size"]

if args.controller == "policy":
    from rsl_rl.runners import OnPolicyRunner

    from go2x.nav_policy import register_with_rsl_rl

    register_with_rsl_rl()
    with open(os.path.join(os.path.dirname(args.checkpoint), "agent.json")) as f:
        agent = json.load(f)
    runner = OnPolicyRunner(wrapped, nav_agent_dict(agent), log_dir=None, device=str(env.device))
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.device)
else:
    ctls = [(StraightController if args.controller == "straight" else AStarController)() for _ in range(B)]

obs = wrapped.get_observations()
rt = get_runtime(env)
results = {}
running_sid = rt.batch.scenario_id.clone()
for c in (ctls if args.controller != "policy" else []):
    c.reset()
t0 = time.time()
step = 0
max_steps = int(math.ceil(N / B) + 2) * int(env.max_episode_length) + 10
with torch.inference_mode():
    while len(results) < N and step < max_steps:
        if args.controller == "policy":
            act = policy(obs)
        else:
            img = obs["policy"][:, : 4 * S * S].reshape(B, 4, S, S).cpu().numpy()
            cmd = torch.tensor(np.stack([c.act(img[i]) for i, c in enumerate(ctls)]), dtype=torch.float32, device=env.device)
            act = normalize_action(cmd, NAV["action"])
        sid_before = running_sid.clone()
        ep_len = env.episode_length_buf.clone()
        gap, fp_steps, f_steps = rt.ep_min_person_gap.clone(), rt.ep_fp_steps.clone(), rt.ep_force_steps.clone()
        obs, _, dones, extras = wrapped.step(act)
        step += 1
        m = rt.m
        tm = env.termination_manager
        # metrics() ran before the reset, so its per-episode sums still describe the finished episodes
        for e in torch.nonzero(dones).flatten().tolist():
            sid = int(sid_before[e])
            if sid < 0 or sid in results:
                continue
            res = {
                "scenario": sid,
                "target": scenarios[sid]["target"][0],
                "success": bool(tm.get_term("success")[e]),
                "collision_footprint": bool(tm.get_term("collision_footprint")[e]),
                "collision_force": bool(tm.get_term("collision_force")[e]),
                "fall": bool(tm.get_term("fall")[e]),
                "out_of_square": bool(tm.get_term("out_of_square")[e]),
                "timeout": bool(tm.get_term("time_out")[e]),
                "time_s": float((ep_len[e] + 1) * env.step_dt),
                "min_person_gap_m": float(min(gap[e].item(), m["person_gap"][e].item())) if scenarios[sid]["people"] else None,
                "footprint_collision_steps": float(fp_steps[e] + m["collision_footprint"][e].float()),
                "force_collision_steps": float(f_steps[e] + m["collision_force"][e].float()),
            }
            res["collision"] = res["collision_footprint"] or res["collision_force"]
            results[sid] = res
            if args.controller != "policy":
                ctls[e].reset()
        running_sid = rt.batch.scenario_id.clone()
        if step % 200 == 0:
            print(f"[eval_nav] {args.controller}: {len(results)}/{N} scenarios, {time.time() - t0:.0f} s", flush=True)

rows = [results[k] for k in sorted(results)]


def stat(vals):
    v = np.asarray([x for x in vals if x is not None], float)
    if len(v) == 0:
        return {"mean": None, "se": None, "n": 0}
    return {"mean": float(v.mean()), "se": float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else None, "n": int(len(v))}


summary = {
    "controller": args.controller, "checkpoint": args.checkpoint, "walk_policy": args.walk_policy,
    "scenarios_evaluated": len(rows), "scenarios_total": N,
    "success_rate": stat([r["success"] for r in rows]),
    "collision_rate": stat([r["collision"] for r in rows]),
    "collision_footprint_rate": stat([r["collision_footprint"] for r in rows]),
    "collision_force_rate": stat([r["collision_force"] for r in rows]),
    "collision_check_disagreement_rate": stat([r["collision_footprint"] != r["collision_force"] for r in rows]),
    "fall_rate": stat([r["fall"] for r in rows]),
    "out_of_square_rate": stat([r["out_of_square"] for r in rows]),
    "timeout_rate": stat([r["timeout"] for r in rows]),
    "min_person_gap_m": stat([r["min_person_gap_m"] for r in rows]),
    "time_to_goal_s": stat([r["time_s"] for r in rows if r["success"]]),
    "by_target": {k: stat([r["success"] for r in rows if r["target"] == k]) for k in ("chair", "table", "person")},
}
if args.controller == "astar":
    summary["astar_failed_plans"] = int(sum(c.failed_plans for c in ctls))
    summary["astar_plans"] = int(sum(c.plans for c in ctls))
os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, f"nav_eval_{args.controller}.json"), "w") as f:
    json.dump({"summary": summary, "episodes": rows}, f, indent=1)
print(json.dumps(summary, indent=1), flush=True)
finish(0)

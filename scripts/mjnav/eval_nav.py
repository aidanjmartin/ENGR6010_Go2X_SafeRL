"""Evaluate navigation controllers in CPU MuJoCo (go2x/mjnav/env.py) on the 200 fixed scenarios
(config/nav_eval_scenarios.json), with the metrics of scripts/isaac/eval_nav.py.

Controllers: `actor:<nav_actor.pt>` (an exported navigation actor, deterministic: the mean action), `straight` and
`astar` (go2x/atrium/baselines.py, world view). All send velocity targets to the same walking policy. The actor
sees the view stored in its file.

    python scripts/mjnav/eval_nav.py --model_dir <unitree_mujoco> --walk_policy policy.pt \
        --controller name=actor:<nav_actor.pt> --controller straight=straight --controller astar=astar --out DIR
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from go2x.atrium.baselines import AStarController, StraightController  # noqa: E402
from go2x.atrium.rewards import normalize_action  # noqa: E402
from go2x.config import CONFIG_DIR, load_config  # noqa: E402
from go2x.mjnav.env import MujocoNavEnv  # noqa: E402
from go2x.nav_net import ImageMLP  # noqa: E402


def stat(vals):
    v = np.asarray([x for x in vals if x is not None], float)
    if len(v) == 0:
        return {"mean": None, "se": None, "n": 0}
    return {"mean": float(v.mean()), "se": float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else None, "n": int(len(v))}


def load_actor(path, device):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    m = blob["meta"]
    net = ImageMLP(m["image_shape"], m["extra_dim"], 3, m["cnn_channels"], m["cnn_out"], m["mlp"])
    net.load_state_dict(blob["state_dict"])
    return net.to(device).eval(), m["image_view"]


def evaluate(kind, arg, args, scenarios, device):
    NAV = load_config("nav")
    N = len(scenarios)
    B = min(args.num_envs, N)
    if kind == "actor":
        net, view = load_actor(arg, device)
    else:
        net, view = None, "world"
        ctls = [(StraightController if kind == "straight" else AStarController)() for _ in range(B)]
    env = MujocoNavEnv(B, args.model_dir, args.walk_policy, seed=NAV["evaluation"]["seed"], scenarios=scenarios, view=view,
                       device=device, threads=args.threads)
    obs = env.get_observations()
    results, step, t0 = {}, 0, time.time()
    sid = env.batch.scenario_id.clone()
    max_steps = int(math.ceil(N / B) + 2) * env.max_episode_length + 10
    with torch.inference_mode():
        while len(results) < N and step < max_steps:
            if net is not None:
                act = net(obs["policy"]).clamp(-1, 1)
            else:
                img = env.image().cpu().numpy()
                cmd = torch.tensor(np.stack([c.act(img[i]) for i, c in enumerate(ctls)]), dtype=torch.float32, device=device)
                act = normalize_action(cmd, NAV["action"])
            obs, _, dones, _ = env.step(act)
            step += 1
            lm = env.last_metrics
            for e in torch.nonzero(dones).flatten().tolist():
                s = int(sid[e])
                if s < 0 or s in results:
                    continue
                results[s] = {
                    "scenario": s, "target": scenarios[s]["target"][0],
                    **{k: bool(lm[k][e]) for k in ("success", "collision", "fall", "out_of_square", "time_out")},
                    "time_s": float(lm["episode_steps"][e]) * env.step_dt,
                    "min_person_gap_m": float(lm["min_person_gap"][e]) if scenarios[s]["people"] else None,
                }
                if net is None:
                    ctls[e].reset()
            sid = env.batch.scenario_id.clone()
            if step % 200 == 0:
                print(f"[eval_nav_mujoco] {kind}: {len(results)}/{N} scenarios, {time.time() - t0:.0f} s", flush=True)
    env.close()
    rows = [results[k] for k in sorted(results)]
    summary = {
        "controller": kind, "source": arg, "image_view": view, "walk_policy": args.walk_policy, "simulator": "mujoco",
        "scenarios_evaluated": len(rows), "scenarios_total": N, "seconds": time.time() - t0,
        "success_rate": stat([r["success"] for r in rows]),
        "collision_rate": stat([r["collision"] for r in rows]),
        "fall_rate": stat([r["fall"] for r in rows]),
        "out_of_square_rate": stat([r["out_of_square"] for r in rows]),
        "timeout_rate": stat([r["time_out"] for r in rows]),
        "min_person_gap_m": stat([r["min_person_gap_m"] for r in rows]),
        "time_to_goal_s": stat([r["time_s"] for r in rows if r["success"]]),
        "by_target": {k: stat([r["success"] for r in rows if r["target"] == k]) for k in ("chair", "table", "person")},
    }
    return summary, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--walk_policy", required=True)
    ap.add_argument("--controller", action="append", required=True, help="name=actor:<path> | name=straight | name=astar")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--num_envs", type=int, default=50)
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    scenarios = json.load(open(CONFIG_DIR / "nav_eval_scenarios.json"))["scenarios"]
    if args.limit:
        scenarios = scenarios[: args.limit]
    os.makedirs(args.out, exist_ok=True)
    table = []
    for spec in args.controller:
        name, what = spec.split("=", 1)
        kind, _, arg = what.partition(":")
        summary, rows = evaluate(kind, arg or None, args, scenarios, args.device)
        summary["name"] = name
        json.dump({"summary": summary, "episodes": rows}, open(os.path.join(args.out, f"nav_eval_{name}.json"), "w"), indent=1)
        table.append(summary)
        print(json.dumps({k: summary[k] for k in ("name", "success_rate", "collision_rate", "fall_rate")}), flush=True)
    f = lambda s: f"{s['mean']:.3f} ± {s['se']:.3f}" if s["mean"] is not None and s["se"] is not None else "n/a"  # noqa: E731
    lines = [f"Navigation in CPU MuJoCo (full go2.xml), {len(scenarios)} scenarios, walker {args.walk_policy}.", "",
             "| controller | success | collision | fall | out of square | timeout | time to goal (s) | chair | table | person |",
             "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for s in table:
        lines.append(f"| {s['name']} | {f(s['success_rate'])} | {f(s['collision_rate'])} | {f(s['fall_rate'])} | "
                     f"{f(s['out_of_square_rate'])} | {f(s['timeout_rate'])} | {f(s['time_to_goal_s'])} | "
                     + " | ".join(f(s["by_target"][k]) for k in ("chair", "table", "person")) + " |")
    open(os.path.join(args.out, "nav_eval_mujoco.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

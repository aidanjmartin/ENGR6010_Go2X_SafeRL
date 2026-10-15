"""Evaluate a trained PPO policy and two baselines over the same seeded episodes.

Simulation only, with unfitted dynamics. These numbers say nothing about the Go2 X.

Usage:
    python scripts/rl/evaluate.py --model models/full.zip
Writes results/eval.json and results/eval.md. Episode seeds are 10000 + i, which training
(seed 0 + worker index, then the environment's own RNG) does not reuse by construction.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from go2x.sim.baselines import GoToGoalPolicy, RandomPolicy
from go2x.sim.nav_env import Go2NavEnv

ROOT = Path(__file__).resolve().parents[2]
SEED0 = 10_000


def run(policy, n, env):
    out = []
    for i in range(n):
        obs, _ = env.reset(seed=SEED0 + i)
        while True:
            obs, _, term, trunc, info = env.step(policy(obs, env))
            if term or trunc:
                break
        out.append({"success": bool(info["is_success"]), "collision": bool(info["collision"]),
                    "timeout": bool(trunc), "time_s": info["time_s"]})
    return out


def summarize(eps):
    succ = [e for e in eps if e["success"]]
    return {"episodes": len(eps),
            "success_rate": np.mean([e["success"] for e in eps]),
            "collision_rate": np.mean([e["collision"] for e in eps]),
            "timeout_rate": np.mean([e["timeout"] for e in eps]),
            "mean_time_to_goal_s": float(np.mean([e["time_s"] for e in succ])) if succ else None}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default=str(ROOT / "models" / "full.zip"))
    p.add_argument("--episodes", type=int, default=100)
    args = p.parse_args()

    model = PPO.load(args.model, device="cpu")
    policies = {
        "PPO": lambda obs, env: model.predict(obs, deterministic=True)[0],
        "random": RandomPolicy(seed=0),
        "go-to-goal (P, no obstacle awareness)": GoToGoalPolicy(),
    }
    env = Go2NavEnv()
    results = {name: summarize(run(pol, args.episodes, env)) for name, pol in policies.items()}
    results = {k: {kk: (float(vv) if isinstance(vv, np.floating) else vv) for kk, vv in v.items()}
               for k, v in results.items()}

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "eval.json").write_text(json.dumps({"model": Path(args.model).name, "seeds": f"{SEED0}..{SEED0 + args.episodes - 1}",
                                               "results": results}, indent=2))
    lines = [f"Simulation only, unfitted dynamics, {args.episodes} seeded episodes (seeds {SEED0} to {SEED0 + args.episodes - 1}).", "",
             "| policy | success | collision | timeout | mean time to goal (s) |",
             "| :--- | ---: | ---: | ---: | ---: |"]
    for name, r in results.items():
        t = f"{r['mean_time_to_goal_s']:.1f}" if r["mean_time_to_goal_s"] is not None else "n/a"
        lines.append(f"| {name} | {r['success_rate']:.2f} | {r['collision_rate']:.2f} | {r['timeout_rate']:.2f} | {t} |")
    (out / "eval.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

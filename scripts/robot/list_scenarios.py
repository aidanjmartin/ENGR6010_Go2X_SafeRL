"""List evaluation scenarios that can be set up on the real floor (no walking people), with the policy's result.

    python scripts/robot/list_scenarios.py [--max-objects 2] [--target chair|table|person]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

p = argparse.ArgumentParser()
p.add_argument("--max-objects", type=int, default=99, help="chairs + tables + people at most")
p.add_argument("--target", choices=["chair", "table", "person"], default=None)
p.add_argument("--eval", default=str(ROOT / "results/phase2/nav_eval_attempt4/nav_eval_policy.json"))
args = p.parse_args()

sc = json.load(open(ROOT / "config/nav_eval_scenarios.json"))["scenarios"]
ev = {e["scenario"]: e for e in json.load(open(args.eval))["episodes"]} if Path(args.eval).exists() else {}
print("id  target     chairs table people  policy in Isaac (attempt 4)")
for i, s in enumerate(sc):
    if any(p["walking"] for p in s["people"]):
        continue
    n = len(s["chairs"]) + len(s["table"]) + len(s["people"])
    if n > args.max_objects or (args.target and s["target"][0] != args.target):
        continue
    e = ev.get(i)
    res = "" if e is None else ("success in %.1f s" % e["time_s"] if e["success"] else
                                "failed: " + ", ".join(k for k in ("collision_footprint", "collision_force", "fall",
                                                                  "out_of_square", "timeout") if e[k]))
    print(f"{i:3d} {s['target'][0]:<10} {len(s['chairs']):6d} {len(s['table']):5d} {len(s['people']):6d}  {res}")

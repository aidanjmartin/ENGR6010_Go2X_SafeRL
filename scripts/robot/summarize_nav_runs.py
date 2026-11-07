"""Table of navigation-policy robot runs (data/*_nav_policy_webrtc, gitignored) as markdown, for results/robot/.

One row per run folder: target phrase, outcome, policy steps, start and end distance to the target, path length,
largest forward command, pose source counts. Runs that never reached the policy are listed with their reason.

    python scripts/robot/summarize_nav_runs.py --out results/robot/nav_camera_runs_webrtc.md
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def summarize(d):
    ev = [json.loads(line) for line in open(d / "events.jsonl")]
    names = [e["event"] for e in ev]
    header = json.loads((d / "header.json").read_text())
    desc = header.get("description") or header.get("extra", {}).get("description") or ""
    target = desc.split("target '")[1].split("'")[0] if "target '" in desc else "?"
    if not (d / "run.json").exists():
        return {"run": d.name[:16], "target": target, "class": "no connection",
                "final": "never connected (only: " + ", ".join(names) + ")"}
    run = json.loads((d / "run.json").read_text())
    st = [e for e in ev if e["event"] == "policy_step"]
    row = {"run": d.name[:16], "target": target, "final": run.get("final", "?"), "steps": len(st)}
    if not st:
        row["class"] = "started at target" if run["final"].startswith("success") else "no policy steps"
        return row
    P = np.array([e["pose"] for e in st])
    src = [e["pose_source"] for e in st]
    row.update(d0=st[0]["distance"], d1=st[-1]["distance"],
               path=float(np.sum(np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1))),
               vx=max(e["cmd_policy"][0] for e in st), cam=src.count("camera"), odo=src.count("odometry"),
               dur=(st[-1]["t_mono_ns"] - st[0]["t_mono_ns"]) / 1e9)
    f = run["final"]
    row["class"] = ("success" if f.startswith("success") else "virtual bumper" if "bumper" in f
                    else "camera lost" if "camera" in f else "other stop")
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(ROOT / "data"))
    p.add_argument("--out", default=None)
    args = p.parse_args()
    rows = [summarize(d) for d in sorted(Path(args.data).glob("*_nav_policy_webrtc"))]
    lines = ["| run (UTC) | target | outcome | steps | target distance start → end (m) | path (m) | max vx asked (m/s) "
             "| pose camera / odometry |", "| :--- | :--- | :--- | ---: | :--- | ---: | ---: | :--- |"]
    for r in rows:
        if "d0" in r:
            lines.append(f"| {r['run']} | {r['target']} | {r['final']} | {r['steps']} | {r['d0']:.2f} → {r['d1']:.2f} "
                         f"| {r['path']:.2f} | {r['vx']:.2f} | {r['cam']} / {r['odo']} |")
        else:
            lines.append(f"| {r['run']} | {r['target']} | {r['final']} | {r.get('steps', 0)} | | | | |")
    real = [r for r in rows if "d0" in r]
    counts = {}
    for r in real:
        counts[r["class"]] = counts.get(r["class"], 0) + 1
    other = {}
    for r in rows:
        if "d0" not in r:
            other[r["class"]] = other.get(r["class"], 0) + 1
    near = [r for r in real if r["d0"] < 1.9]
    far = [r for r in real if r["d0"] >= 1.9]
    summary = [
        f"{len(rows)} run folders; {len(real)} with policy steps: " + ", ".join(f"{v} {k}" for k, v in counts.items())
        + f"; success {counts.get('success', 0)}/{len(real)}.",
        "Without policy steps: " + ", ".join(f"{v} {k}" for k, v in other.items()) + ".",
        f"Start under 1.9 m from the target: {sum(r['class'] == 'success' for r in near)}/{len(near)} success; "
        f"1.9 m or more: {sum(r['class'] == 'success' for r in far)}/{len(far)}.",
        f"Pose from the camera on {sum(r['cam'] for r in real)} of {sum(r['cam'] + r['odo'] for r in real)} policy steps.",
    ]
    text = "\n".join(lines) + "\n\n" + "\n".join(f"- {s}" for s in summary) + "\n"
    if args.out:
        Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()

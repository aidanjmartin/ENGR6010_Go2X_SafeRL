"""Collect 07_l_path trials into one table (stdout and results/robot/l_path_trials.md).

Usage:
    python scripts/motion/summarize_trials.py --backend webrtc [--since 20261017T2100]
--since keeps sessions whose folder name (UTC time stamp) sorts at or after the given prefix.
If one trial number was run more than once, the latest run is used.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load_trial(folder):
    ev = [json.loads(line) for line in open(folder / "events.jsonl")]
    header = json.loads((folder / "header.json").read_text())
    seg = {e["segment"]: e for e in ev if e["event"] == "segment_end"}
    end = next((e for e in ev if e["event"] == "trial_end"), None)
    stop = next((e.get("reason") for e in ev if e["event"] == "stop"), None)
    if stop is None and not any(e["event"] == "connected" for e in ev):
        stop = "connection failed; nothing sent"
    return {"trial": header["trial"], "folder": folder.name, "args": header["args"], "seg": seg, "end": end, "stop": stop}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--backend", default="webrtc")
    p.add_argument("--since", default="")
    args = p.parse_args()

    folders = sorted(f for f in (ROOT / "data").glob(f"*_07_l_path_t*_{args.backend}") if f.name >= args.since)
    trials = {}
    for f in folders:  # later runs of the same trial number replace earlier ones
        t = load_trial(f)
        trials[t["trial"]] = t
    if not trials:
        raise SystemExit("no 07_l_path sessions found")

    a = next(iter(trials.values()))["args"]
    D, A = a["distance"], a["angle"]
    fmt = lambda e, k, u: "n/a" if e is None else f"{e[k]:.3f}{'' if e['reached'] else ' (timeout)'}"
    lines = [f"L-path trials on the Go2 X, {args.backend} backend: forward {D} m at {a['vx']} m/s, turn left {A} deg "
             f"at {a['wz']} rad/s, forward {D} m; closed loop on the robot's odometry and IMU. Values are the robot's own "
             "odometry and IMU, after a 1 s settle; they are not external ground truth.", "",
             "| trial | leg 1 (m) | turn (deg) | leg 2 (m) | end forward (m) | end left (m) | heading change (deg) | stop reason |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |"]
    rows = []
    for n in sorted(trials):
        t = trials[n]
        s, e = t["seg"], t["end"]
        l1, tu, l2 = s.get("leg1"), s.get("turn"), s.get("leg2")
        row = [l1["final_m"] if l1 else np.nan, tu["final_deg"] if tu else np.nan, l2["final_m"] if l2 else np.nan,
               *(e["end_rel"] if e else [np.nan, np.nan]), e["dyaw_deg"] if e else np.nan]
        rows.append(row)
        turn_txt = "n/a" if tu is None else f"{tu['final_deg']:.1f}" + ("" if tu["reached"] else " (timeout)")
        lines.append(f"| {n} | {fmt(l1, 'final_m', 'm')} | {turn_txt} | "
                     f"{fmt(l2, 'final_m', 'm')} | {row[3]:+.3f} | {row[4]:+.3f} | {row[5]:+.1f} | {t['stop']} |")
    r = np.array(rows, float)
    if len(r) > 1:
        m, sd = np.nanmean(r, 0), np.nanstd(r, 0, ddof=1)
        lines.append(f"| mean ± sd | {m[0]:.3f} ± {sd[0]:.3f} | {m[1]:.1f} ± {sd[1]:.1f} | {m[2]:.3f} ± {sd[2]:.3f} | "
                     f"{m[3]:+.3f} ± {sd[3]:.3f} | {m[4]:+.3f} ± {sd[4]:.3f} | {m[5]:+.1f} ± {sd[5]:.1f} | |")
    lines.append("")
    lines.append(f"Ideal: legs {D} m, turn {A} deg, end +{D} m forward and +{D} m left of the start, heading change +{A} deg. "
                 "Sessions: " + ", ".join(trials[n]["folder"] for n in sorted(trials)) + " (data/, not committed).")
    text = "\n".join(lines) + "\n"
    print(text)
    out = ROOT / "results" / "robot"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"l_path_trials_{args.backend}.md").write_text(text)
    print(f"wrote {out / f'l_path_trials_{args.backend}.md'}")


if __name__ == "__main__":
    main()

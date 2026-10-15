"""Plot the training curve from a run's monitor.csv (one row per episode).

Usage:
    python scripts/rl/plot_training.py --run runs/full
Writes results/training_curve.png: rolling mean episode return, and rolling success and
collision rates, against environment steps. Simulation only, unfitted dynamics.
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SERIES_1, SERIES_2 = "#2a78d6", "#eb6834"
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def rolling(x, w):
    return np.convolve(x, np.ones(w) / w, mode="valid")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", default=str(ROOT / "runs" / "full"))
    p.add_argument("--window", type=int, default=500, help="episodes per rolling mean")
    p.add_argument("--out", default=str(ROOT / "results" / "training_curve.png"))
    args = p.parse_args()

    with open(Path(args.run) / "monitor.csv") as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    length = np.array([int(r["l"]) for r in rows])
    ret = np.array([float(r["r"]) for r in rows])
    succ = np.array([r["is_success"] == "True" for r in rows], float)
    coll = np.array([r["collision"] == "True" for r in rows], float)
    # VecMonitor rows come in completion order; cumulative episode length approximates env steps
    steps = np.cumsum(length)
    w = min(args.window, max(len(rows) // 10, 1))
    x = steps[w - 1:] / 1e6

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6), facecolor=SURFACE)
    for ax in (a1, a2):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("environment steps (millions)")
    a1.plot(x, rolling(ret, w), color=SERIES_1, lw=2)
    a1.set_title("Episode return", loc="left", fontsize=10)
    a2.plot(x, rolling(succ, w), color=SERIES_1, lw=2, label="success")
    a2.plot(x, rolling(coll, w), color=SERIES_2, lw=2, label="collision")
    a2.set_ylim(0, 1)
    a2.set_title("Episode outcome rate", loc="left", fontsize=10)
    a2.legend(frameon=False, loc="center right")
    for y, name, c in ((rolling(succ, w)[-1], "success", SERIES_1), (rolling(coll, w)[-1], "collision", SERIES_2)):
        a2.annotate(f"{name} {y:.2f}", (x[-1], y), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8, color=MUTED)
    fig.suptitle(f"PPO on Go2NavEnv, rolling mean over {w} episodes (simulation, unfitted dynamics)",
                 x=0.01, ha="left", fontsize=10, color=TEXT)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"wrote {args.out}: {len(rows)} episodes, {steps[-1]} steps")


if __name__ == "__main__":
    main()

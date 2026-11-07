"""Training curves of the MJX walking runs: evaluation return and episode length against environment steps.

    python scripts/mjx/plot_walk_curves.py results/phase2/mujoco_walk
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUNS = ["mjx_s0", "mjx_s1", "mjx_s2", "mjx_nodr", "mjx_sym", "mjx_push1"]


def main():
    root = Path(sys.argv[1])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for r in RUNS:
        f = root / r / "progress.json"
        if not f.exists():
            continue
        p = json.load(open(f))
        x = [e["step"] / 1e6 for e in p]
        style = "-" if r.startswith("mjx_s") else "--"
        ax[0].plot(x, [e["eval/episode_reward"] for e in p], style, label=r)
        ax[1].plot(x, [e["eval/avg_episode_length"] * 0.02 for e in p], style, label=r)
    ax[0].set(xlabel="environment steps (millions)", ylabel="evaluation return", title="Return (each run's own randomization)")
    ax[1].set(xlabel="environment steps (millions)", ylabel="episode length (s, 20 s max)", title="Time before a fall")
    for a in ax:
        a.grid(alpha=0.3)
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(root / "training_curves.png", dpi=120)
    print(f"wrote {root / 'training_curves.png'}")


if __name__ == "__main__":
    main()

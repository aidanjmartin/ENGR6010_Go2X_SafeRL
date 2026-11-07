"""Training curves of the MuJoCo navigation runs: success, collision and the policy's action noise per iteration
(25-iteration moving average).

    python scripts/mjnav/plot_nav_curves.py results/phase2/mujoco_nav
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RUNS = ["nav_mj_scratch", "nav_mj_finetune", "nav_mj_ent003"]
KEYS = [("/episode/success", "success rate"), ("/episode/collision", "collision rate"),
        ("Policy/mean_noise_std", "action noise (std)")]


def smooth(v, k=25):
    v = np.asarray(v, float)
    if len(v) < k:
        return v
    return np.convolve(v, np.ones(k) / k, mode="valid")


def main():
    root = Path(sys.argv[1])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for r in RUNS:
        f = root / r / "scalars.json"
        if not f.exists():
            continue
        s = json.load(open(f))
        for a, (key, name) in zip(ax, KEYS):
            it, val = zip(*s[key])
            y = smooth(val)
            a.plot(np.asarray(it)[len(it) - len(y):], y, label=r)
            a.set(xlabel="iteration (this run)", ylabel=name, title=name)
    for a in ax:
        a.grid(alpha=0.3)
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(root / "training_curves.png", dpi=120)
    print(f"wrote {root / 'training_curves.png'}")


if __name__ == "__main__":
    main()

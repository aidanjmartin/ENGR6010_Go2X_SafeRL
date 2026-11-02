"""Plot an RSL-RL training run from its TensorBoard event files.

    python scripts/isaac/plot_training.py --run $DATA_DIR/walk --out curve.png --json scalars.json

Panels: mean reward, mean episode length, and every `Episode_Reward/*` and `Episode_Termination/*` term.
"""
import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--run", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--json", default=None)
parser.add_argument("--title", default="Walking policy training")
args = parser.parse_args()

series = {}
for path in sorted(glob.glob(os.path.join(args.run, "**", "events.out.tfevents.*"), recursive=True)):
    acc = EventAccumulator(path, size_guidance={"scalars": 0})
    acc.Reload()
    for tag in acc.Tags()["scalars"]:
        series.setdefault(tag, {}).update({e.step: e.value for e in acc.Scalars(tag)})
series = {k: sorted(v.items()) for k, v in series.items()}
if not series:
    raise SystemExit(f"no TensorBoard scalars under {args.run}")

groups = [
    ("Mean reward", [k for k in series if k == "Train/mean_reward"]),
    ("Mean episode length (policy steps)", [k for k in series if k == "Train/mean_episode_length"]),
    ("Reward terms (per episode, per second)", [k for k in series if k.startswith("Episode_Reward/")]),
    ("Termination fractions", [k for k in series if k.startswith("Episode_Termination/")]),
]
groups = [g for g in groups if g[1]]
fig, axes = plt.subplots(len(groups), 1, figsize=(8, 2.6 * len(groups)), sharex=True)
axes = [axes] if len(groups) == 1 else axes
for ax, (title, tags) in zip(axes, groups):
    for tag in sorted(tags):
        x, y = zip(*series[tag])
        ax.plot(x, y, lw=1.2, label=tag.split("/", 1)[-1])
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    if len(tags) > 1:
        ax.legend(fontsize=6, ncol=3, loc="best")
axes[-1].set_xlabel("PPO iteration")
fig.suptitle(args.title)
fig.tight_layout()
fig.savefig(args.out, dpi=120)
print("wrote", args.out, "tags:", len(series))
if args.json:
    with open(args.json, "w") as f:
        json.dump(series, f)

"""Report Python, PyTorch, CUDA, Isaac Sim and Isaac Lab versions, then start Isaac Sim headless.

Runs in the Isaac venv on the Dell (job 010). Writes a JSON report.
"""
import argparse
import importlib.metadata as md
import json
import platform
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
args = parser.parse_args()

import torch  # noqa: E402

report = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "torch": torch.__version__,
    "torch_cuda_build": torch.version.cuda,
    "torch.cuda.is_available()": torch.cuda.is_available(),
    "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}
for pkg in ("isaacsim", "isaaclab", "isaaclab_tasks", "isaaclab_rl", "rsl-rl-lib", "unitree_rl_lab"):
    try:
        report[pkg] = md.version(pkg)
    except md.PackageNotFoundError:
        report[pkg] = None
print(json.dumps(report, indent=2), flush=True)

t0 = time.time()
from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True).app
report["isaac_sim_headless_start_s"] = round(time.time() - t0, 1)
try:
    from isaacsim.core.version import get_version

    report["isaac_sim_build"] = list(get_version())
except Exception as exc:  # the version helper moved between releases; keep the pip version above
    report["isaac_sim_build"] = f"unavailable: {exc}"

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: E402,F401
import unitree_rl_lab.tasks  # noqa: E402,F401

report["go2_tasks"] = sorted(k for k in gym.registry if "Go2" in k)
report["navigation_tasks"] = sorted(k for k in gym.registry if "Navigation" in k)
print(json.dumps(report, indent=2), flush=True)
with open(args.out, "w") as f:
    json.dump(report, f, indent=2)
from go2x.isaac.shutdown import finish  # noqa: E402

finish(0)

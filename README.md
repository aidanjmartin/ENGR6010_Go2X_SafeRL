# Learned-Model Safety Filtering for Vision-Guided RL Navigation on the Unitree Go2

Aidan J. Martin, ENGR 6010: AI in Robotics, Middle Tennessee State University, Fall 2026.

An RL policy drives a Unitree Go2 X to a target object while a safety filter built on a learned dynamics model blocks commands predicted to cause a collision. The final study compares RL alone, RL with a physics-model filter, and RL with a learned-ensemble filter.

This repository holds **Phase 1** (due 2026-10-16):

1. **Motion control:** a laptop connects to the Go2 X over a USB-C Ethernet tether and sends body velocity commands, with logs.
2. **RL simulation:** a velocity-level navigation environment and a PPO policy trained in it.

The safety filter, the learned ensemble, segmentation, dynamics data collection and overhead camera tracking are later phases and are not here.

It builds on [ENGR6010_Exam1](https://github.com/aidanjmartin/ENGR6010_Exam1) (commit `90150da`), which showed that this Go2 X answers both DDS and WebRTC over the wired network, read-only. Code copied from it says so at the top of the file. [`LOG.md`](LOG.md) records every step and every difference from Exam 1.

## Status

| part | state |
| :--- | :--- |
| mock robot backend | in progress |
| DDS backend (`unitree_sdk2_python`) | **untested on hardware** |
| WebRTC backend (`unitree_webrtc_connect`) | **untested on hardware** |
| motion test scripts | in progress |
| RL environment and PPO | in progress |

## Layout

```
README.md  LOG.md  .gitignore  .env.example  pyproject.toml
env/
  environment-mac.yml          simulation and tests (conda-forge)
  environment-ubuntu.yml       simulation plus robot libraries (conda-forge)
config/
  safety_limits.yaml           Phase 1 test limits, rate limit, watchdog, duration cap
  sim.yaml                     simulation defaults
go2x/                          Python package
  robot/                       interface, mock / DDS / WebRTC backends, safety layer, logger
  sim/                         dynamics models, scene and LiDAR, Gymnasium env, baselines
scripts/
  motion/                      Phase 1 motion tests and log summary
  rl/                          training, evaluation, video, training curve
tests/                         pytest, all against the mock or the simulator
results/                       evaluation table, training curve, rollout video
docs/LAPTOP_SESSION.md         brief for the robot session on the Ubuntu laptop
data/  models/  runs/          untracked: robot logs, checkpoints, training runs
```

## Setup

On the Mac:

```bash
conda env create -p .venv -f env/environment-mac.yml
.venv/bin/pip install -e . --no-deps
.venv/bin/pytest
```

On the Ubuntu laptop, follow [`docs/LAPTOP_SESSION.md`](docs/LAPTOP_SESSION.md). Robot addresses and keys go in an untracked `.env` file. [`.env.example`](.env.example) names the variables.

## Results

Filled in after the robot session and the full training run.

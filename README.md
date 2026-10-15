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
| mock robot backend, safety layer, logger | done; 9 tests against the mock pass |
| DDS backend (`unitree_sdk2_python` `814556d`) | written from the library source; **untested on hardware** |
| WebRTC backend (`unitree_webrtc_connect` `e0abc57`) | written from the library source; **untested on hardware** |
| motion test scripts | done; all five run against the mock |
| RL environment | done; `check_env` and 5 other tests pass |
| PPO training | 10M-step run on the Ubuntu laptop (44.5 min, 24 envs); evaluation success 0.79, no better than the 200k smoke run |

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

## Robot interface (Task 2)

`go2x.robot` gives one interface for velocity commands and sport-mode state, with three backends:

| backend | link | state topic | velocity command |
| :--- | :--- | :--- | :--- |
| `mock` | none | 20 Hz from a first-order lag plus dead time (tau 0.25 s, delay 0.1 s; assumptions) | held until replaced |
| `dds` | Ethernet, CycloneDDS | `rt/lf/sportmodestate` | `SportClient.Move` |
| `webrtc` | Ethernet, WebRTC data channel | `rt/lf/sportmodestate` | sport request `Move` (api 1008), no reply |

**The DDS and WebRTC backends are untested on hardware.** Every library call in them cites the source file at the pinned commit. Motion scripts never call a backend directly. They go through `SafeRobot` ([`go2x/robot/safety.py`](go2x/robot/safety.py)), which does the following, with limits from [`config/safety_limits.yaml`](config/safety_limits.yaml) only:
- clamps commands to 0.2 m/s linear and 0.5 rad/s yaw;
- limits their rate of change;
- sends zero velocity when state stops for 0.5 s (in the command loop and in a separate watchdog thread);
- stops on Ctrl+C, any key, any exception, and a per-script time cap.

Stopping sends zero velocity, then `StopMove`, never `Damp`. The logger writes the Exam 1 data-collection format: JSON Lines stamped with `time.monotonic_ns()` and `time.time_ns()` on arrival, the robot's own time stamp kept, the serial number dropped, and a session header.

## Motion tests (Task 3)

```bash
.venv/bin/python scripts/motion/00_read_state.py --backend mock   # read-only: state rate, sends nothing
.venv/bin/python scripts/motion/01_stand.py   --backend mock      # stand up and hold 10 s
.venv/bin/python scripts/motion/02_forward.py --backend mock      # 0.5 m forward at 0.2 m/s
.venv/bin/python scripts/motion/03_rotate.py  --backend mock      # 90 deg at 0.3 rad/s
.venv/bin/python scripts/motion/04_lateral.py --backend mock      # 0.3 m left at 0.15 m/s
.venv/bin/python scripts/motion/05_square.py  --backend mock      # 1 m square at 0.2 m/s
```

`--backend dds` or `--backend webrtc` asks for the safety checklist first. Each run writes `data/<time>_<script>_<backend>/` and prints commanded vs. measured velocity, a rough command-to-motion delay, and commanded travel vs. odometry. The robot session follows [`docs/LAPTOP_SESSION.md`](docs/LAPTOP_SESSION.md).

## RL simulation (Task 4)

[`go2x/sim/nav_env.py`](go2x/sim/nav_env.py) is a Gymnasium environment for velocity-level navigation:
- **Action:** normalized (vx, vy, wz), with limits of 0.5 m/s, 0.3 m/s and 0.8 rad/s.
- **Dynamics:** integrated at 20 Hz by a pluggable `DynamicsModel`.
- **Scene:** a random room with boxes and cylinders, and a goal that is checked to be reachable.
- **Observation:** goal distance and bearing, body velocity, previous action, and 32 simulated LiDAR beams.
- **Reward:** progress, success bonus, collision penalty and a time penalty.
- **Filter hook:** `action_filter` is the single point between policy output and dynamics where a later phase inserts the safety filter. Phase 1 leaves it empty.

**The default dynamics are assumptions, not measurements.** `LagDelayDynamics` is a per-axis first-order lag with dead time, and its time constant (0.1 to 0.4 s) and delay (0 to 0.15 s) are drawn at random each episode. Later phases replace it with fitted and learned models through the same class.

```bash
.venv/bin/python scripts/rl/train_ppo.py --timesteps 10000000 --name full
.venv/bin/python scripts/rl/evaluate.py --model models/full.zip      # 100 seeded episodes vs. baselines
.venv/bin/python scripts/rl/record_video.py --model models/full.zip  # results/rollout.mp4
.venv/bin/python scripts/rl/plot_training.py --run runs/full         # results/training_curve.png
```

## Results

**Simulation results come from unfitted dynamics. They say nothing about Go2 X performance.** Robot results are filled in after the robot session.

Robot motion tests: pending (see [`docs/LAPTOP_SESSION.md`](docs/LAPTOP_SESSION.md)).

| backend | script | measured velocity | rough delay (10 % / 50 %) | displacement | stop reason |
| :--- | :--- | :--- | :--- | :--- | :--- |
| | | | | | |

RL evaluation, 10M-step PPO run (`models/full.zip`, trained on the Ubuntu laptop on 2026-10-15), 100 seeded episodes (seeds 10000 to 10099):

| policy | success | collision | timeout | mean time to goal (s) |
| :--- | ---: | ---: | ---: | ---: |
| PPO | 0.79 | 0.21 | 0.00 | 5.8 |
| random | 0.00 | 0.17 | 0.83 | n/a |
| go-to-goal (P, no obstacle awareness) | 0.69 | 0.31 | 0.00 | 6.2 |

PPO beats the obstacle-unaware go-to-goal controller by 10 points of success. The curve ([`results/training_curve.png`](results/training_curve.png)) plateaus by about 2M steps, and the 200k smoke model scored about the same (0.81). See "Problems found" in [`LOG.md`](LOG.md). Rollout video: [`results/rollout.mp4`](results/rollout.mp4).

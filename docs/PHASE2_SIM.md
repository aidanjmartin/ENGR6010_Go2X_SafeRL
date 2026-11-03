# Phase 2 simulation: walking policy and atrium navigation

Branch `phase2-rl-sim`. All Isaac Sim work runs on the Dell T2 tower through the Dell runner (jobs on `dell-jobs`, results on `dell-results`). [`LOG.md`](../LOG.md) records every step, every failure and every version source.

**Status rule:** a number appears here only after a job measured it. Anything else says "Plan, not yet built" or "not yet run".

## What was built

```
top-down image (10 Hz) ──► navigation policy ──► velocity target (vx, vy, wz)
                                                        │
proprioception (50 Hz) ───────────────────────────► walking policy ──► 12 joint position targets ──► PD (200 Hz in simulation)
```

| part | files | tests (CPU) |
| :--- | :--- | :--- |
| versions and install | `jobs/pins.sh`, `jobs/constraints.txt`, `jobs/common.sh`, `jobs/010_setup_isaac.sh` | |
| low-level contract | [`docs/LOWLEVEL_INTERFACE.md`](LOWLEVEL_INTERFACE.md), `go2x/sim2sim.py` | `tests/test_sim2sim.py` |
| Task 1 walking policy | `config/walk.yaml`, `go2x/isaac/walk_env_cfg.py`, `go2x/isaac/actuators.py`, `scripts/isaac/train_walk.py`, `scripts/isaac/eval_walk.py` | |
| Task 1 transition logger | `go2x/transitions.py` | `tests/test_transitions.py` |
| Task 2 atrium scene | `config/atrium.yaml`, `go2x/atrium/building.py`, `scripts/atrium/convert_building.py`, `go2x/atrium/scene.py`, `go2x/atrium/geometry.py`, `go2x/atrium/batch.py`, `go2x/atrium/state.py` | `tests/test_atrium_scene.py`, `tests/test_atrium_geometry.py`, `tests/test_atrium_state.py` |
| Task 3 camera and drawn image | `config/camera.yaml`, `config/topdown.yaml`, `go2x/atrium/camera.py`, `go2x/atrium/topdown.py`, `scripts/isaac/render_scenes.py` | `tests/test_camera_topdown.py` |
| Task 4 navigation | `config/nav.yaml`, `go2x/isaac/nav_env_cfg.py`, `go2x/nav_policy.py`, `go2x/atrium/rewards.py`, `go2x/atrium/baselines.py`, `scripts/atrium/make_scenarios.py`, `scripts/isaac/train_nav.py`, `scripts/isaac/eval_nav.py`, `scripts/isaac/record_nav_videos.py` | `tests/test_rewards_baselines.py`, `tests/test_nav_policy.py` |
| Task 5 language target | `config/grounding.yaml`, `go2x/grounding.py`, `scripts/ground_prompts.py` | `tests/test_grounding.py` |
| Task 6 MuJoCo check | `config/mujoco.yaml`, `scripts/mujoco_check.py` | `tests/test_sim2sim.py` |

Phase 1 code (`go2x/robot`, `go2x/sim`, `scripts/motion`, `scripts/rl`) is unchanged.

### Task 1: walking policy

Base: `unitreerobotics/unitree_rl_lab` `4960b84`, task `Unitree-Go2-Velocity`, on Isaac Lab v2.3.2 and Isaac Sim 5.1.0 (choice and sources in `LOG.md`). PPO (RSL-RL 3.1.2, `unitree_rl_lab` settings) from a random initialization; no checkpoint is ever loaded. Changes from the base task: flat floor, the Go2 model from the `unitree_ros` URDF, the Phase 2 velocity target ranges, and added randomization (motor strength, PD gain error). Every range is in `config/walk.yaml`.

Evaluation (`scripts/isaac/eval_walk.py`):
- 100 episodes of 20 s, one per environment, fixed seed;
- a new random velocity target from the training range every 5 s, no pushes;
- an episode counts until its first termination, and any termination other than the time limit is a fall.

The same pass logs (base state, velocity target, base state at +0.1 s and +0.5 s) for every step of every environment to `transitions_walk_eval.npz` in the job's `$DATA_DIR`. Windows that cross a reset are dropped.

### Task 2: atrium scene

- **Square:** 3.048 m (10 ft) on a side, centered on the `aeb_level1.sdf` world origin (Central Lobby, room 100).
- **Building check:** `scripts/atrium/convert_building.py` reads the world from `tourbot` at `2757cf1`. Result in `results/phase2/square_check.json`: no box inside the square; nearest is `glass_3485`, 2.81 m from the center and 0.99 m outside the edge; 97 boxes within 3 m of the edge, exported to `config/building_level1_near_square.json`.
- **Building in the renderer:** those boxes appear only in the rendered variant. The drawn image covers the square plus 0.3 m, and no building box reaches into that area.
- **Obstacles:** chairs, a table, and people (cylinders like `tourbot_rl` `visitor.sdf`) are kinematic rigid bodies. People walk a straight path across the square and back, following a speed script (0.5 to 1.2 m/s with pauses, adapted from `tourbot_rl` `pedestrian.py`), moved every walking-policy step. Unused obstacles wait behind the camera, outside the square.
- **Layout checks:** layouts are rejected when the start pose overlaps anything (0.15 m clearance) or the target is unreachable on a 5 cm free-space grid.
- **Collision:** checked two ways, logged separately: footprint overlap (robot 0.70 m by 0.31 m rectangle against obstacle footprints), and contact force above 1 N between any robot body and any obstacle part (one filtered contact sensor per robot body).

### Task 3: camera and drawn image

- **Camera:** 640 by 480 pixels, 60 degree horizontal field of view, at (3.0, 0.0, 4.0) above the lobby railing side, aimed at the square center.
- **Coverage check:** `coverage_check()` verifies that the square plus 0.3 m margin and the robot top (0.4 m) anywhere in the square project at least 2 pixels inside the image; every check point is at least 33 pixels inside.
- **Lens choice:** 90 degrees also passed but left the square about 300 by 200 pixels.
- **Floor homography:** computed from the same model (`Camera.floor_homography`).
- **Drawn image:** 64 by 64 pixels over 3.648 m (5.7 cm per pixel). Four channels: static obstacles, people, robot footprint with heading marker, target. The current and previous frames are stacked.
- **Camera realism:** frames captured at 15 Hz with pose, position and yaw noise and missed detections, delivered after a 0.1 to 0.2 s delay drawn per episode (`config/topdown.yaml`).

### Task 4: navigation policy

- **Structure:** follows Isaac Lab's `Isaac-Navigation-Flat-Anymal-C-v0`. A `ManagerBasedRLEnv` whose action term (a subclass of `PreTrainedPolicyAction`) runs the frozen TorchScript walking policy at 50 Hz under 200 Hz physics. The navigation policy acts at 10 Hz.
- **Action:** a velocity target scaled to vx 0 to 0.8 m/s, vy -0.3 to 0.3 m/s, wz -1.0 to 1.0 rad/s.
- **Observation:** the stacked image plus the previous action, read by a small convolutional encoder (`go2x/nav_policy.py`, because RSL-RL 3.1.2 has only MLP policies).
- **Rewards:** seven terms, each logged separately.
- **Episode ends:** success, footprint collision, force collision, fall, leaving the square, or 20 s.
- **Baselines:** both read only the drawn image the policy sees and send velocity targets to the same walking policy.
- **Evaluation:** 200 scenarios, generated once with seed 20261108 and stored in `config/nav_eval_scenarios.json`, shared by all three controllers.

### Task 5: language target

`go2x/grounding.py` makes prompts ("go to the {color} chair", "go to the chair", "go to the table", "go to the person") with ground-truth target IDs and skips prompts that match more than one object. `scripts/ground_prompts.py`:
1. runs OWLv2 on the rendered images;
2. projects the best box's bottom center to the floor with the homography;
3. draws a disk-shaped target mask at that point;
4. compares obstacle masks from detections with the drawn image of the same scene.

### Task 6: MuJoCo check

`scripts/mujoco_check.py` runs the exported walking policy in MuJoCo 3.16.0 on Unitree's `unitree_mujoco` Go2 model (`1eb6642`). It uses the conventions in `docs/LOWLEVEL_INTERFACE.md` and the exact velocity targets of the Isaac evaluation, and reports tracking error and falls side by side.

## How to run each job

Every job is a file in `jobs/` on this branch plus a TOML file on `dell-jobs`. Each job reads earlier jobs' files from `$DATA_ROOT/<job-id>/`.

| job | command | depends on | GPU |
| :--- | :--- | :--- | :--- |
| 010-isaac-setup | `bash jobs/010_setup_isaac.sh` | | yes |
| 020-walk-short | `bash jobs/020_walk_short.sh` | 010 | yes |
| 025-nav-smoke | `bash jobs/025_nav_smoke.sh` | 020 | yes |
| 030-walk-full | `bash jobs/030_walk_full.sh` | 020 | yes |
| 040-walk-eval | `bash jobs/040_walk_eval.sh` | 030 | yes |
| 045-render-scenes | `bash jobs/045_render_scenes.sh` | 030 | yes |
| 048-nav-baselines | `bash jobs/048_nav_baselines.sh` | 030 | yes |
| 050-nav-train | `bash jobs/050_nav_train.sh` | 030, 025 | yes |
| 060-nav-eval | `bash jobs/060_nav_eval.sh` | 050 | yes |
| 065-nav-videos | `bash jobs/065_nav_videos.sh` | 060, 048 | yes |
| 070-grounding | `bash jobs/070_grounding.sh` | 045 | yes |
| 080-mujoco-check | `bash jobs/080_mujoco_check.sh` | 040 | no |

CPU tests, in any Python 3.11 environment with numpy, scipy, pyyaml, torch, `rsl-rl-lib==3.1.2`, `mujoco==3.16.0` and this package installed:

```bash
pytest -q tests/
```

## Results

Values are mean ± standard error. Rates over scenarios use the binomial standard error; `n` is the number of episodes behind each number.

### Task 1: walking policy

- **Training (job 030):** 3000 PPO iterations from a random initialization, 4096 environments, 53 minutes on the RTX 4000 Ada. Over the last 50 iterations, 0.996 of training episodes reached the 20 s limit. Curve: `results/phase2/walk_train/training_curve.png`.
- **Evaluation (job 040):** 100 episodes of 20 s, seed 1000, a new random velocity target every 5 s from the training ranges, no pushes. Table: `results/phase2/walk_eval/walk_eval.md`.

| metric | value |
| :--- | ---: |
| tracking error, vx (m/s) | 0.060 ± 0.009 |
| tracking error, vy (m/s) | 0.054 ± 0.002 |
| tracking error, wz (rad/s) | 0.101 ± 0.006 |
| mean joint torque magnitude (N m) | 3.56 ± 0.07 |
| fraction of steps with a joint within 0.01 rad of a hard limit | 0.0094 ± 0.0060 |
| foot contact fraction, FL / FR / RL / RR | 0.77 / 0.79 / 0.70 / 0.71 (standard errors 0.006 to 0.009) |
| falls per minute | 0.061 ± 0.043 (2 falls in 32.7 robot-minutes; Poisson standard error) |

- **Transitions:** 97,451 windows of (state, target, state 0.1 s later, state 0.5 s later), 12 state values each, in `transitions_walk_eval.npz` (13 MB, compressed) in job 040's `$DATA_DIR` on the Dell (`~/dell-runner/data/ENGR6010_Go2X_SafeRL/040-walk-eval/`).
- **Video:** `results/phase2/walk_eval/walk_20s.mp4`, environment 0 for 20 s. Isaac Lab simulated the motion; MuJoCo drew it (next item).
- **Rendering:** Isaac Sim 5.1's RTX renderer crashes at start on the Dell's driver 595.99.02. NVIDIA's maintainers say 5.1.0 is validated on driver 580.65.06 (`LOG.md` step 7). Physics and training are not affected. Every image and video in this phase is drawn with MuJoCo 3.16.0 using Unitree's Go2 model, posed at states recorded in Isaac Lab.

### Task 2: atrium scene

The square and building check results are above (`results/phase2/square_check.json`). The 200 evaluation scenarios have chair targets 110, table 24 and person 66. They contain 340 chairs, 94 tables and 207 people, of whom 50 walk. Both collision checks are logged for every episode; their disagreement rate is in the Task 4 table.

### Task 3: camera and drawn image

- **Coverage:** `results/phase2/camera_coverage.json`.
- **Rendered scenes (job 045):** 60 scenes (seed 5000), each with the camera image, ground truth (classes, colors, footprints, robot pose, K, R, C, floor homography) and the drawn top-down image of the true state. All 60 are in job 045's `$DATA_DIR`. Four examples and a contact sheet are in `results/phase2/scenes/`.
- **Renderer:** the images come from MuJoCo, not Isaac Sim (Task 1, "Rendering"). The MuJoCo camera reproduces the configured pinhole model: `tests/test_mj_scene.py` checks that it projects world and floor points to the same pixels as `Camera.project` and the homography, within 1e-6 pixel.

### Task 4: navigation (200 shared scenarios)

| metric | straight to target | A* with pure pursuit | learned policy, attempt 1 |
| :--- | ---: | ---: | ---: |
| success rate | 0.580 ± 0.035 (n = 200) | 0.695 ± 0.033 (n = 200) | 0.125 ± 0.023 (n = 200) |
| collision rate (either check) | 0.410 ± 0.035 (n = 200) | 0.290 ± 0.032 (n = 200) | 0.240 ± 0.030 (n = 200) |
| collision, footprint check | 0.380 ± 0.034 (n = 200) | 0.255 ± 0.031 (n = 200) | 0.195 ± 0.028 (n = 200) |
| collision, contact-force check | 0.065 ± 0.017 (n = 200) | 0.055 ± 0.016 (n = 200) | 0.065 ± 0.017 (n = 200) |
| episodes where the two checks disagree | 0.375 ± 0.034 (n = 200) | 0.270 ± 0.031 (n = 200) | 0.220 ± 0.029 (n = 200) |
| fall rate | 0.010 ± 0.007 (n = 200) | 0.000 ± 0.000 (n = 200) | 0.000 ± 0.000 (n = 200) |
| left the square | 0.000 ± 0.000 (n = 200) | 0.015 ± 0.009 (n = 200) | 0.365 ± 0.034 (n = 200) |
| reached the 20 s limit | 0.000 ± 0.000 (n = 200) | 0.000 ± 0.000 (n = 200) | 0.270 ± 0.031 (n = 200) |
| minimum distance to a person (m), episodes with people | 0.443 ± 0.041 (n = 137) | 0.438 ± 0.041 (n = 137) | 0.641 ± 0.044 (n = 137) |
| time to goal (s), successful episodes | 3.184 ± 0.074 (n = 116) | 3.959 ± 0.090 (n = 139) | 5.144 ± 0.436 (n = 25) |

- **Attempt 1:** the learned policy (job 050 attempt 1) did worse than both baselines. Leaving the square ended an episode with no penalty, and the policy learned to walk out (`LOG.md` step 8).
- **Attempt 2:** job 050 is retraining with a leaving-the-square penalty and critic-only target information. Plan, not yet built: its evaluation and the videos.
- **A* planner:** 0 of 281 A* plans failed.
- **Collision checks:** they disagree in 0.22 to 0.38 of episodes, and the footprint check fires 3 to 5 times as often as the contact-force check. A likely reason, not yet checked: the 0.70 by 0.31 m footprint rectangle is larger than the robot's legs and body near the floor, so it can overlap an obstacle's footprint before any robot body touches the obstacle.

### Task 5 and Task 6

Plan, not yet built: jobs 070 (grounding) and 080 (MuJoCo check) run after the navigation jobs.

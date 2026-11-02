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

Plan, not yet built: this section is filled from the job outputs as they arrive.

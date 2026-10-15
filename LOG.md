# Phase 1 log

Machine: MacBook, Apple M5 (10 cores), macOS (Darwin 25.6.0), conda 26.1.1, Homebrew 7.0.7.
Date: 2026-10-14.

Reference: [ENGR6010_Exam1](https://github.com/aidanjmartin/ENGR6010_Exam1) at commit `90150da`, read-only, cloned outside this repository.

## Task 1: repository and environments

| step | command | outcome |
| :--- | :--- | :--- |
| clone Exam 1, read-only | `git clone https://github.com/aidanjmartin/ENGR6010_Exam1 <scratchpad>` | success: commit `90150da` |
| GitHub CLI | `brew install gh` | success: gh 2.102.0 |
| read pinned upstream sources | `git clone <url>` then `git checkout <commit>`, outside this repository | `unitree_sdk2_python` `814556d`: `python_requires='>=3.8'`, `cyclonedds==0.10.2` (`setup.py`). `unitree_webrtc_connect` `e0abc57`: `requires-python = ">=3.8"`, version 2.2.0 (`pyproject.toml`) |
| resolve Mac environment | `conda create --dry-run --override-channels -c conda-forge python=3.11 ...` (osx-arm64) | success: versions pinned in `env/environment-mac.yml` |
| resolve Ubuntu environment | the same with `--platform linux-64` and `pytorch=*=cuda*`, `CONDA_OVERRIDE_CUDA=12.8` | success: pytorch 2.14.1 CUDA 12.9 build; versions pinned in `env/environment-ubuntu.yml` |

| Mac environment | `conda env create -p .venv -f env/environment-mac.yml && .venv/bin/pip install -e . --no-deps` (run by Aidan) | success: torch 2.14.1, gymnasium 1.3.0, stable-baselines3 2.9.0, numpy 2.4.6 |
| first commit and GitHub repo | `gh repo create aidanjmartin/ENGR6010_Go2X_SafeRL --public --source . --remote origin --push` (run by Aidan) | success: commit `f14f9ef` |

## Task 4: RL simulation

| step | command | outcome |
| :--- | :--- | :--- |
| tests | `.venv/bin/pytest -q` | 6 passed: `check_env`, lag-plus-delay step response against the analytic curve, raycast and collision on a hand-built scene, seeded determinism, filter hook, go-to-goal in an empty room |
| single-env speed | 3 s of random actions | 9400 steps per second; reset 0.4 ms including the reachability check |
| smoke training | `.venv/bin/python scripts/rl/train_ppo.py --timesteps 200000 --name smoke` | 28.8 s, 6900 steps per second with 16 `SubprocVecEnv` workers. First quarter of episodes: success 0.58, collision 0.34. Last quarter: success 0.79, collision 0.21. It learns. |
| decision: where to run full training | (2026-10-15) | moved to the Ubuntu laptop, which Aidan has a day early, to keep the Mac free for Tasks 2, 3 and 5. The run is CPU-bound either way (SB3 warns that PPO with an MLP policy is meant for the CPU, `stable_baselines3/common/on_policy_algorithm.py:152`). |
| evaluation, video and curve scripts on the smoke model | `scripts/rl/evaluate.py`, `record_video.py`, `plot_training.py` | all run. Smoke model over 100 seeded episodes (seeds 10000 to 10099): PPO success 0.81, collision 0.19; random 0.00 and 0.17 (0.83 timeouts); proportional go-to-goal 0.69 and 0.31. Evaluation takes 12 s. |

## Differences from Exam 1

| item | Exam 1 | Phase 1 | why |
| :--- | :--- | :--- | :--- |
| robot computer | lab desktop, Ubuntu 26.04 | laptop, Ubuntu 24.04, RTX 4000 Ada Laptop GPU | the laptop goes with the robot; setup is repeated there |
| environments | one per part: Q2 Python 3.11 (DDS), Q3 Python 3.10 (WebRTC) | one per machine, Python 3.11, both robot libraries together on Ubuntu | one robot interface imports both backends; both libraries declare Python >= 3.8, and CycloneDDS 0.10.2 was proven on 3.11 in Exam 1. The two libraries were never installed together in Exam 1. |
| environment files | `requirements.txt` per part | conda `environment-*.yml` per machine | project brief |
| PyTorch | pip wheel, `cu128` index | conda-forge `pytorch=2.14.1` CUDA 12.9 build | conda-forge only; Exam 1 already used conda-forge-only environments |
| `.gitignore` | Python template plus a custom section | the custom section only, plus SB3 `.zip` checkpoints, `models/`, `runs/` | SB3 saves checkpoints as `.zip` |
| `.env.example` | four variables | same four; the AES key is marked not needed on this firmware | Exam 1 found `data2 = 2` |
| upstream clones | `~/go2_deps/`, commits in Exam 1 `q2_sdk/LOG.md` | same location and commits for CycloneDDS and `unitree_sdk2_python`; `unitree_webrtc_connect` installed by pip at `e0abc57` | Exam 1 `q3_apps/requirements.txt` pins it the same way |

## Problems found

**The scenes may be too easy to separate the policies.** A proportional go-to-goal controller that ignores obstacles reaches the goal in 69 % of evaluation episodes, because obstacles are placed uniformly and often lie off the start-to-goal line. The headroom for RL, and later for a safety filter, is the remaining 31 %. Not changed yet; noted for discussion.

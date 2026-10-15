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

| push | `git push origin main` | success: commit `16b89b0`. Obstacle layout left unchanged; full training not started (Aidan to decide where it runs). |

## Task 2: robot interface (2026-10-15)

| step | command | outcome |
| :--- | :--- | :--- |
| read pinned sources | `unitree_sdk2_python` `814556d`, `unitree_webrtc_connect` `e0abc57`, outside this repository | DDS: `SportClient` `Move` / `StopMove` / `StandUp` / `BalanceStand` (`go2/sport/sport_client.py`); `Move` is a no-reply call that returns 0 when sent (`rpc/client_base.py`). The only sport-state topic in this SDK commit is `rt/lf/sportmodestate` (`example/a2/sport/a2_sport_state.py:8`). WebRTC: `Move` as a no-reply request (`examples/go2/data_channel/sportmode_mcf/sportmode_mcf.py:109-126`); the same file says to send `BalanceStand` after `StandUp` before any `Move` (lines 301-303). |
| velocity frame | web search for Unitree's `SportModeState` field documentation | not found for the Go2. Unitree's Go1 docs describe a world frame. Left open; the motion summary reports both readings (see Problems). |
| write interface, mock, DDS, WebRTC, safety layer, logger | `go2x/robot/` | every library call cites its source file in the module docstring; `read_env()` adapted from Exam 1 `q3_apps/webrtc_read_state.py` |
| tests | `.venv/bin/pytest -q tests/test_robot.py` | first run: 8 of 9 passed. The rate-limit test counted the stop's immediate zero as a rate violation; the stop is immediate by design, so the test now excludes the stop commands. Then 9 passed: clamp, rate limit, zero-then-StopMove, state watchdog (zero within 0.5 s plus margin of state stopping), exception, Ctrl+C, key press, time cap, log format with `sn` dropped, mock forward distance 0.5 m within 3 cm |
| import check without robot libraries | `python -c "import go2x.robot.dds, go2x.robot.webrtc"` | success on the Mac; the libraries are imported only on `connect()` |

## Task 3: motion scripts (2026-10-15)

| step | command | outcome |
| :--- | :--- | :--- |
| scripts 01 to 05 on the mock | `.venv/bin/python scripts/motion/0N_*.py --backend mock` | all five end with "end of script". Forward: 0.501 m commanded, 0.501 m odometry; rotate: +90.3 deg commanded, +90.2 deg IMU; lateral: 0.300 m; square: closes within 0.012 m and 1.0 deg. Mock state 20.0 to 20.1 per second, largest gap 63 ms. The mock's time to 10 % is about 250 ms (0.1 s assumed delay plus the rate-limit ramp). |
| summary fix | `summarize_log.py` | the first square summary averaged "measured" velocity over turns and pauses (0.092 m/s for a 0.2 m/s command). It now averages within each plateau and skips the first second of each: +0.200 m/s, +0.300 rad/s |
| time cap | `config/safety_limits.yaml` `max_script_s` 60 to 90 s; square capped at 75 s | the square takes 54 s on the mock, and a real stand-up adds about 2 s, which would have put the 60 s cap within a few seconds of the planned run |
| read-only state check | `scripts/motion/00_read_state.py --backend mock --seconds 3` | 20.3 per second. Added because Exam 1 never read `rt/lf/sportmodestate` over DDS, and the watchdog depends on it. It sends no commands and does not use `SafeRobot`. |

## Task 5: laptop session brief (2026-10-15)

[`docs/LAPTOP_SESSION.md`](docs/LAPTOP_SESSION.md) covers, in order: clone, environment, `go2-eth` and finding the robot, the read-only tests (Exam 1's plus `00_read_state` on both links), the motion scripts one at a time with Aidan's confirmation (DDS first, then WebRTC), recording results, optional training, and what to do if neither backend moves the robot. Aidan runs the motion scripts in his own terminal, because the `yes` prompt and the key-press stop need a real terminal.

## Ubuntu laptop setup and training (2026-10-15)

Machine: laptop, Intel Core i9-13950HX (24 cores: 8 P-cores with Hyper-Threading plus 16 E-cores, 32 threads), 62 GB RAM, NVIDIA RTX 4000 Ada Generation Laptop GPU (12 GB), driver 580.173.02 (CUDA 13.0), Ubuntu 24.04.5 LTS, kernel 7.0.0-31.

| step | command | outcome |
| :--- | :--- | :--- |
| read-only inspection | `lsb_release -a; nvidia-smi; nproc; lscpu; free -h; ip -brief link` | machine line above. conda and git missing. Interfaces: `enp0s31f6` (built-in Ethernet, `e1000e`, no carrier) and `wlp0s20f3` (Wi-Fi, default route). No USB-C Ethernet adapter attached. |
| git | installed by Aidan | git 2.43.0 |
| clone | `git clone <url> ~/ENGR6010_Go2X_SafeRL`; `git clone <url> ~/ENGR6010_Exam1_ref && git -C ~/ENGR6010_Exam1_ref checkout 90150da` | success: project at `9a207c5`, Exam 1 at `90150da` (read-only) |
| branch | `git switch -c laptop` | success; all laptop work goes on `laptop` |
| system packages | `sudo apt install build-essential cmake portaudio19-dev` (run by Aidan) | success: build-essential 12.10ubuntu1, cmake 3.28.3, portaudio19-dev 19.6.0 |
| Miniconda | `bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3` (run by Aidan) | success: conda 26.9.3 |
| build CycloneDDS | `git clone -b releases/0.10.x <url> ~/go2_deps/cyclonedds && git checkout 5041f35`, then `cmake .. -DCMAKE_INSTALL_PREFIX=../install -DBUILD_EXAMPLES=OFF && cmake --build . --target install` | success: `libddsc.so.0.10.5` in `~/go2_deps/cyclonedds/install` |
| clone the SDK | `git clone <url> ~/go2_deps/unitree_sdk2_python && git checkout 814556d` | success; `setup.py` pins `cyclonedds==0.10.2` and `python_requires='>=3.8'` |
| conda env | `conda env create -p .venv -f env/environment-ubuntu.yml` | success: Python 3.11.17, pytorch 2.14.1 (CUDA build), unitree_webrtc_connect 2.2.0 (pyaudio built against portaudio19-dev) |
| SDK install | `CYCLONEDDS_HOME=~/go2_deps/cyclonedds/install .venv/bin/pip install -e ~/go2_deps/unitree_sdk2_python` | success: unitree_sdk2py 1.0.1, cyclonedds 0.10.2 |
| project install | `.venv/bin/pip install -e . --no-deps` | success: go2x 0.1.0 |
| import check | `.venv/bin/python -c "import unitree_sdk2py, cyclonedds, unitree_webrtc_connect, torch; ..."` | all four import in one Python 3.11 environment (Exam 1 needed 3.11 for DDS and 3.10 for WebRTC); `torch.cuda.is_available()` is `True` on the RTX 4000 Ada |
| dependency check | `.venv/bin/pip check` | "No broken requirements found." |
| tests | `.venv/bin/pytest -q` | 15 passed in 12.2 s |
| network, state and motion tests | `LAPTOP_SESSION.md` steps 3 to 6 | not run: no robot or USB-C Ethernet adapter on 2026-10-15; they wait for the robot session |
| PPO throughput | `train_ppo.py --timesteps 200000 --n-envs {16,24}` | 16 envs: 3146 steps/s; 24 envs: 3795 steps/s. Full run uses 24 (one per physical core). Smoke runs deleted. |
| PPO full run | `train_ppo.py --timesteps 10000000 --n-envs 24 --name full` | success: 2671 s (44.5 min), 3744 steps/s, 86460 episodes. `models/full.zip` and `runs/full/` stay local (gitignored). |
| evaluation | `evaluate.py --model models/full.zip` | 100 seeded episodes (seeds 10000 to 10099): PPO success 0.79, collision 0.21, timeout 0.00, 5.8 s to goal; random 0.00 / 0.17 / 0.83; go-to-goal 0.69 / 0.31 / 0.00, 6.2 s. Written to `results/eval.md` and `results/eval.json`. |
| GPU use | (none: reasoning from the code) | Training, evaluation and video all run on the CPU: `train_ppo.py` defaults to `--device cpu`, and `evaluate.py:52` and `record_video.py:25` load the model with `device="cpu"`. The GPU sat essentially idle. Usage was not measured during the run. The GPU is only used by the `torch.cuda.is_available()` check. |
| git identity | `git commit` (run by Aidan) | failed: `user.name` and `user.email` are not set on the laptop. The files were staged; `git push -u origin laptop` pushed only the existing `9a207c5`. |
| video | `record_video.py --model models/full.zip` | `results/rollout.mp4`; both episodes reach the goal (3.9 s, 6.1 s) |
| training curve | `plot_training.py --run runs/full` | `results/training_curve.png`. Success rises from about 0.70 to 0.80 in the first 2M steps, then stays flat at 0.78 to 0.84; the last 500 episodes read success 0.76, collision 0.24. |

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

**The 10M-step run is no better than the 200k smoke run.** Full model: success 0.79 on the evaluation seeds; smoke model on the Mac: 0.81. The curve plateaus by about 2M steps. Likely causes, not yet tested: the easy-scenes problem below caps what the policy can gain, and the fixed hyperparameters (`ent_coef=0.0`, constant learning rate) may stop exploration early. Not changed; noted for discussion.

**The laptop trains at about half the Mac's speed.** 3146 steps/s with 16 envs and 3795 with 24 on the i9-13950HX, against 6900 with 16 on the Mac. The cause was not investigated. With 8 P-cores and 16 E-cores, part of each rollout may wait on the slower E-cores.

**The scenes may be too easy to separate the policies.** A proportional go-to-goal controller that ignores obstacles reaches the goal in 69 % of evaluation episodes, because obstacles are placed uniformly and often lie off the start-to-goal line. The headroom for RL, and later for a safety filter, is the remaining 31 %. Not changed yet; noted for discussion.

**The frame of `SportModeState.velocity` is not documented in the SDK source.** `_SportModeState_.py` gives only the field types. Unitree's Go1 documentation describes a world frame for its high-level state. The motion summary therefore reports linear velocity as reported and rotated world→body by the IMU yaw. The rotate and square runs on the robot will show which is right.

**Sport-mode state may read zero on some Go2 firmware.** A forum reply ([mybotshop forum](https://forum.mybotshop.de/t/unitree-go2-cannot-echo-sportmodestate/1810)) reports all-zero sport-mode state on a Go2 and calls it a legacy feature. This is unofficial, and Exam 1's WebRTC run contradicts it on this robot: body height 0.321 m, 20 messages per second. The watchdog depends only on arrival, not on content, but velocity measurements would be lost if it happens over DDS. `00_read_state` checks it first.

**`unitree_webrtc_connect` has two silent-failure paths.** `pub_sub.publish_without_callback` builds an exception for a closed channel but never raises it, so the message is silently dropped. `publish_request_new` awaits its reply with no timeout. The WebRTC backend checks `channel.readyState` before every `Move` and wraps each request in `asyncio.wait_for` (5 s).

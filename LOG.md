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
| git identity | `git commit` (run by Aidan) | failed at first: `user.name` and `user.email` were not set on the laptop, and `git push -u origin laptop` pushed only the existing `9a207c5`. After the identity was set, the work landed as `87a711a` on `laptop`, which was fast-forwarded into `main` on the Mac on 2026-10-17. |
| video | `record_video.py --model models/full.zip` | `results/rollout.mp4`; both episodes reach the goal (3.9 s, 6.1 s) |
| training curve | `plot_training.py --run runs/full` | `results/training_curve.png`. Success rises from about 0.70 to 0.80 in the first 2M steps, then stays flat at 0.78 to 0.84; the last 500 episodes read success 0.76, collision 0.24. |

## Mac robot session (2026-10-17)

Machine: MacBook, Apple M5 (10 cores), macOS (Darwin 25.6.0); USB-C Ethernet adapter AX88179B (`en9`) to the robot's wired port; internet on Wi-Fi (`en0`). The Ubuntu laptop was no longer available, so this Mac replaced it for the robot session. Only the WebRTC backend was tested; **the DDS backend is still untested on hardware.** Robot: a Go2 X, possibly not the unit from Exam 1. Its handshake is the same (`data2 = 2`).

| step | command | outcome |
| :--- | :--- | :--- |
| install | `brew install portaudio` (19.7.0); `CFLAGS=-I/opt/homebrew/include LDFLAGS=-L/opt/homebrew/lib .venv/bin/pip install "unitree_webrtc_connect @ git+...@e0abc57"` | success: unitree_webrtc_connect 2.2.0, aiortc 1.15.0 (same as Exam 1), PyAudio 0.2.14 built from source; numpy 2.4.6 and torch 2.14.1 unchanged; `pip check` clean |
| retrain PPO (the laptop's `models/full.zip` was not transferable) | `train_ppo.py --timesteps 3000000 --n-envs 8 --name full_mac` | 520 s, 5769 steps per second. Evaluation seeds 10000 to 10099: success 0.75, collision 0.25, 6.0 s to goal (laptop 10M model: 0.79 / 0.21). Used only for the robot demo; the README's evaluation table stays the laptop's. |
| network | `sudo ifconfig en9 alias 192.168.123.99 netmask 255.255.255.0` (run by Aidan) | a temporary second address next to the adapter's saved `192.168.185.5/16`, which another project uses; saved settings untouched. Ping to `192.168.123.161`: 0.2 to 0.6 ms |
| handshake probe | Exam 1 `q3_apps/probe_signaling.py 192.168.123.161` | port 9991 open, `data2 = 2` (firmware 1.1.14 or earlier, no AES key) |
| read-only state | `scripts/motion/00_read_state.py --backend webrtc --seconds 10` | 200 messages in 9.9 s, 20.1 per second, largest gap 62 ms; motion switcher reports `mcf`; body height 0.318 m; real, nonzero data |
| 01_stand, first try | `01_stand.py --backend webrtc` | failed at connect, nothing sent: `LocalSignalingPortError`. `en9` had lost every address after an Ethernet link drop (see Problems). Fixed by re-adding the alias; from then on a pre-flight `nc -z 192.168.123.161 9991` ran before each script. |
| 01_stand | `01_stand.py --backend webrtc` | StandUp 0, BalanceStand 0, StopMove 0. Stopped by Ctrl+C at 16.0 s (about 1 s before the planned end). 20.1 per second, largest gap 59 ms; drift 1.1 cm, 5.1 deg; body height 0.318 to 0.336 m; `mode` stayed 0 |
| 02_forward, 0.2 m/s for 0.5 m | `02_forward.py --backend webrtc --skip-stand` | **first motion.** End of script. Reported vx +0.160 m/s mean (peak 0.215); odometry 0.327 m of 0.500 m commanded; 10 % at 138 ms, 50 % at 191 ms |
| velocity frame | the same run | **body frame.** IMU yaw was 2.98 rad, so forward motion moved odometry x from -2.043 to -2.369 m while reported vx stayed positive |
| 03_rotate, 0.3 rad/s | `03_rotate.py --backend webrtc --skip-stand` | **13 deg of 90.** Yaw rate rose to 0.14 rad/s by 0.5 s, then faded to about 0 while 0.3 rad/s was still commanded. Aidan saw the robot trying to turn without turning. |
| `--wz` option | `scripts/motion/03_rotate.py` | added; default 0.3, refused above `max_wz` |
| 03_rotate, 0.5 rad/s | `03_rotate.py --backend webrtc --skip-stand --wz 0.5` | **42 deg of 90**, no fade: 0.17 to 0.35 rad/s throughout (mean 0.29); stops within 0.4 s; counterclockwise for positive wz, as in the simulator |
| 04_lateral, 0.15 m/s | `04_lateral.py --backend webrtc --skip-stand` | no motion: 0.9 cm |
| `--vy` option | `scripts/motion/04_lateral.py` | added; default 0.15, refused above `max_vy` |
| 04_lateral, 0.2 m/s | `04_lateral.py --backend webrtc --skip-stand --vy 0.2` | 4 cm, reported vy +0.03 m/s; direction left |
| raise `max_vy` | `config/safety_limits.yaml`: 0.2 to 0.3 m/s (Aidan's decision) | the robot did not move sideways at 0.15 or 0.2 m/s; 0.3 m/s is the policy's training limit. `max_vx` 0.2 and `max_wz` 0.5 unchanged |
| summary fix | `summarize_log.py` | a plateau shorter than 2 s gave an empty averaging window (`nan`); it now skips half the plateau instead of 1 s |
| 04_lateral, 0.3 m/s | `04_lateral.py --backend webrtc --skip-stand --vy 0.3` | 0.17 m of 0.30 m, reported vy +0.173 m/s; **left, as in the simulator** |
| RL policy | `06_rl_goal.py --backend webrtc --velocity-frame body --skip-stand` (new script, tested on the mock first: goal reached, geofence fires) | **goal reached**: distance 1.00 to 0.30 m in 3.14 s (57 steps at 20 Hz); end of script, StopMove 0. Largest commands sent 0.2 / 0.3 / 0.5, all at the limits. Measured vx +0.201 of 0.2 m/s, vy +0.225 of 0.3, wz -0.453 of -0.5 rad/s; odometry 0.83 m, yaw -41 deg. The policy turned right while stepping left, then turned back toward the goal, as it does in simulation. Pose from odometry and IMU yaw; LiDAR inputs held at max range. |
| L-path test script | new `scripts/motion/07_l_path.py` and `summarize_trials.py` | forward 0.5 m at 0.2 m/s, turn left 90 deg at 0.5 rad/s, forward 0.5 m. Closed loop on odometry and IMU, with each segment stopping early by a predicted coast (measured speed x 0.2 s, plus the ramp-down) and timing out at 8 s per leg and 10 s for the turn. First mock run overshot to 0.61 m and 108 deg; with the early stop the mock gave 0.53 m and 94 deg. |
| link drop, again | first `07_l_path.py --trial 1` | failed at connect, nothing sent: `en9` had lost its addresses again (second time, about 20 min after the first). Alias re-added; trial 1 rerun. |
| L-path trials 1 to 4 | `07_l_path.py --backend webrtc --skip-stand --trial N` | all ended "end of script". Legs (m) / turn (deg): T1 0.412 / 82.5 / 0.450; T2 0.428 / 80.5 / 0.445; T3 0.341 (leg timeout) / 80.5 / 0.456; T4 0.294 (timeout) / 85.8 / 0.062 (timeout). Mean measured forward speed during the legs, at a 0.2 m/s command: T1 0.105 / 0.115, T2 0.110 / 0.110, T3 0.043 / 0.071, T4 0.037 / 0.012 m/s. Turn rate steady at 0.24 to 0.27 rad/s. Table: `results/robot/l_path_trials_webrtc.md` |
| L-path trial 5 | `07_l_path.py --backend webrtc --skip-stand --trial 5` | **WebRTC connection failed, nothing sent**: `DataChannelTimeoutError` after 15 s (peer stuck at `connecting`, ICE at `checking`). Kept in the record as it happened. |

## Differences from Exam 1

| item | Exam 1 | Phase 1 | why |
| :--- | :--- | :--- | :--- |
| robot computer | lab desktop, Ubuntu 26.04 | laptop, Ubuntu 24.04, RTX 4000 Ada Laptop GPU | the laptop goes with the robot; setup is repeated there |
| environments | one per part: Q2 Python 3.11 (DDS), Q3 Python 3.10 (WebRTC) | one per machine, Python 3.11, both robot libraries together on Ubuntu | one robot interface imports both backends; both libraries declare Python >= 3.8, and CycloneDDS 0.10.2 was proven on 3.11 in Exam 1. The two libraries were never installed together in Exam 1. |
| environment files | `requirements.txt` per part | conda `environment-*.yml` per machine | project brief |
| PyTorch | pip wheel, `cu128` index | conda-forge `pytorch=2.14.1` CUDA 12.9 build | conda-forge only; Exam 1 already used conda-forge-only environments |
| `.gitignore` | Python template plus a custom section | the custom section only, plus SB3 `.zip` checkpoints, `models/`, `runs/` | SB3 saves checkpoints as `.zip` |
| `.env.example` | four variables | same four; the AES key is marked not needed on this firmware | Exam 1 found `data2 = 2` |
| robot computer, final | lab desktop, Ubuntu 26.04 | MacBook (macOS) for the robot session on 2026-10-17 | the Ubuntu laptop was no longer available. Neither library README covers macOS (both give only `apt` steps); WebRTC worked with Homebrew PortAudio. DDS was not set up on the Mac. |
| robot network | NetworkManager `go2-eth`, static `192.168.123.99/24` | temporary alias `192.168.123.99/24` on the Mac's USB adapter (`ifconfig en9 alias`) | keeps the adapter's saved configuration for another project; the alias is lost on any link drop |
| upstream clones | `~/go2_deps/`, commits in Exam 1 `q2_sdk/LOG.md` | same location and commits for CycloneDDS and `unitree_sdk2_python`; `unitree_webrtc_connect` installed by pip at `e0abc57` | Exam 1 `q3_apps/requirements.txt` pins it the same way |

## Problems found

**Small commands barely move this robot.** Yaw at 0.3 rad/s faded to nothing (13 deg of 90); at 0.5 rad/s it turned at about 0.29 rad/s (42 deg). Lateral at 0.15 and 0.2 m/s did not move it; at 0.3 m/s it moved 0.17 m of 0.30. Forward at 0.2 m/s gave 0.33 m of 0.50. During the RL run, with larger commands on several axes at once, it tracked much better (vx 0.20 of 0.20, wz 0.45 of 0.50). The mechanism is not known: a minimum speed in the gait controller fits these runs. This is the first real dynamics data for the fitted and learned models in later phases. The simulator's lag-plus-delay model has no minimum speed.

**Forward tracking fell off over the L-path trials.** At the same 0.2 m/s command, mean forward speed was about 0.11 m/s in trials 1 and 2 and 0.01 to 0.07 m/s in trials 3 and 4, which made three legs time out. The turn rate stayed at 0.24 to 0.27 rad/s throughout. The cause is not known. Aidan's observation: nothing solid was ahead of the robot in trials 3 and 4, only a table to its right; the Ethernet tether trailing from the robot may have been detected as an obstacle (unconfirmed). Other candidates: the robot's own obstacle avoidance reacting to the table, battery level (not logged), or floor surface.

**The L-path early stop was too cautious on the robot.** Legs that reached target ended at 0.41 to 0.45 m of 0.5, and turns at 80 to 86 deg of 90. The robot coasts less than the 0.2 s response time assumed from the mock; `--t-resp` should be lowered for future runs.

**WebRTC connection can time out.** L-path trial 5 failed in the WebRTC handshake (`DataChannelTimeoutError`; the library's message suggests another client holding the slot, a flaky network, or a stale peer on the robot). It was the only handshake failure in the session. Nothing was sent.

**The adapter loses its addresses on a link drop.** It happened twice, about 20 minutes apart. Between the read-only check (15:11) and the first stand attempt (15:25), `en9` lost both the alias and its saved `192.168.185.5`, though the saved settings were unchanged. The connection failed before any command was sent. The cause of the link drop was not identified.

**Duplicate FFmpeg classes on macOS.** `opencv-python` and `av` (PyAV, used by aiortc) each bundle `libavdevice`, and the Objective-C runtime warns `Class AVFFrameReceiver is implemented in both ...`. Harmless here because no video is decoded, but video on this Mac may be unreliable.

**`SportModeState.mode` stayed 0** through StandUp and BalanceStand, so it does not show posture on this firmware over `rt/lf/sportmodestate`.

**The 10M-step run is no better than the 200k smoke run.** Full model: success 0.79 on the evaluation seeds; smoke model on the Mac: 0.81. The curve plateaus by about 2M steps. Likely causes, not yet tested: the easy-scenes problem below caps what the policy can gain, and the fixed hyperparameters (`ent_coef=0.0`, constant learning rate) may stop exploration early. Not changed; noted for discussion.

**The laptop trains at about half the Mac's speed.** 3146 steps/s with 16 envs and 3795 with 24 on the i9-13950HX, against 6900 with 16 on the Mac. The cause was not investigated. With 8 P-cores and 16 E-cores, part of each rollout may wait on the slower E-cores.

**The scenes may be too easy to separate the policies.** A proportional go-to-goal controller that ignores obstacles reaches the goal in 69 % of evaluation episodes, because obstacles are placed uniformly and often lie off the start-to-goal line. The headroom for RL, and later for a safety filter, is the remaining 31 %. Not changed yet; noted for discussion.

**The frame of `SportModeState.velocity` is not documented in the SDK source.** Resolved on 2026-10-17: body frame (see the Mac robot session). `_SportModeState_.py` gives only the field types. Unitree's Go1 documentation describes a world frame for its high-level state. The motion summary therefore reports linear velocity as reported and rotated world→body by the IMU yaw. The rotate and square runs on the robot will show which is right.

**Sport-mode state may read zero on some Go2 firmware.** A forum reply ([mybotshop forum](https://forum.mybotshop.de/t/unitree-go2-cannot-echo-sportmodestate/1810)) reports all-zero sport-mode state on a Go2 and calls it a legacy feature. This is unofficial, and Exam 1's WebRTC run contradicts it on this robot: body height 0.321 m, 20 messages per second. The watchdog depends only on arrival, not on content, but velocity measurements would be lost if it happens over DDS. `00_read_state` checks it first.

**`unitree_webrtc_connect` has two silent-failure paths.** `pub_sub.publish_without_callback` builds an exception for a closed channel but never raises it, so the message is silently dropped. `publish_request_new` awaits its reply with no timeout. The WebRTC backend checks `channel.readyState` before every `Move` and wraps each request in `asyncio.wait_for` (5 s).

# Phase 2 log: walking policy and atrium navigation in Isaac Lab

Branch: `phase2-rl-sim`. Started 2026-11-02. Deadline 2026-11-08.

Machine for this session: the Dell T2 tower itself (`BASENGD55243`): NVIDIA RTX 4000 Ada Generation (20475 MiB), driver 595.99.02, Ubuntu 24.04.5 LTS, GLIBC 2.39, 24 CPU threads, 62 GiB RAM. The session writes code and runs CPU tests here directly; all Isaac Sim work runs as runner jobs.

## Step 0: runner and reference reading (2026-11-02)

| step | command | outcome |
| :--- | :--- | :--- |
| job branches | `git ls-remote --heads https://github.com/aidanjmartin/ENGR6010_Go2X_SafeRL` | `dell-jobs` `bccb864` and `dell-results` `b223cfa` exist (created by the runner at 14:32) |
| heartbeat | `gh api .../contents/_runner/heartbeat.json?ref=dell-results` | 2026-11-02 14:32:31, idle, no pending jobs, GPU visible (1063 MiB used), 892.8 GB free |
| runner environment | `tr '\0' '\n' < /proc/<runner pid>/environ` | no `CUDA_VISIBLE_DEVICES`, so the hidden-GPU problem from the October weekend (`tourbot` `dell-jobs:notes/weekend-report.md`) is not present |
| runner code | `~/weekend/tourbot/tools/dell-runner/runner.py` at `tourbot` `7e74205` (branch `dell-runner`) | `DATA_DIR` is per job (`~/dell-runner/data/<repo>/<job-id>`); a job's `ref` resolves to the branch tip when the job starts |
| reference repos | `tourbot` (read-only) at `aidan/sim-levels-2-3` `2757cf1`, `aidan/ureca-rl` `2a10b64`, `dell-runner` `7e74205`, `dell-jobs` `6ddd78c` | read `aeb_level1.sdf`, `aeb_level1_rooms.csv`, `tourbot_sim/README.md`, `tourbot_rl/models/visitor.sdf`, `tourbot_rl/tourbot_rl/pedestrian.py`, `tools/dell-runner/README.md`, `WEEKEND_RULES.md`, `notes/weekend-report.md` |
| quick square check (throwaway, outside the repo) | regex parse of the 4,270 `<collision>` boxes in `aeb_level1.sdf` | nothing inside the 3.048 m square at (0, 0); nearest is `glass_3485` at 2.810 m from the center (0.994 m outside the edge); 97 boxes within 3 m of the edge. Repeated by the repository script in Task 2. |

## Version choices (approved by Aidan on 2026-11-02)

| item | choice | primary source |
| :--- | :--- | :--- |
| Isaac Sim | 5.1.0, pip (`isaacsim[all,extscache]==5.1.0`, `https://pypi.nvidia.com`) | Isaac Sim 5.1.0 requirements page: Ubuntu 22.04/24.04, driver 580.65.06 or later, 16 GB VRAM, 32 GB RAM. Install page: Python 3.11, GLIBC 2.35 or later, `OMNI_KIT_ACCEPT_EULA=YES` for non-interactive runs. The same page marks 5.1.0 as no longer receiving fixes. |
| Isaac Lab | v2.3.2, commit `37ddf626871758333d6ed89cf64ad702aef127d0` | Isaac Lab README at v3.0.0: "v2.3.X: Isaac Sim 4.5 / 5.0 / 5.1". v2.3.2 `pip_installation.rst`: `torch==2.7.0 torchvision==0.22.0` from the `cu128` index. Its install docs list Ubuntu 22.04 only (Isaac Sim lists 24.04). |
| base repo | `unitreerobotics/unitree_rl_lab` `main`, commit `4960b84732b0c2ec593dccbfe963fda1bcd7b1e3` (Apache-2.0) | README badges: Isaac Sim 5.1.0, Isaac Lab 2.3.0. Task `Unitree-Go2-Velocity` (`tasks/locomotion/robots/go2/__init__.py`). Its `isaaclab3.0` branch (`2522f89`, 2026-05-25) targets Isaac Lab 3.0 with Isaac Sim 6.0, a pre-release, so it was not used. |
| Go2 model | `unitreerobotics/unitree_ros` commit `3ae893d905d11795ddf1075825ee5010364a1109` (BSD-3-Clause), `robots/go2_description/urdf/go2_description.urdf` | `unitree_rl_lab` README recommends the URDF method for Isaac Sim 5.0 or later. Last change to `go2_description`: `a3b70ca`, 2025-06-19. |
| Python | CPython 3.11.17 through `uv` 0.12.24 (release tarball, SHA-256 `b4dfaef4...bb3da6` checked) | the Dell has only Python 3.12.3 and no conda; standalone CPython lives in `$JOB_CACHE`, nothing system-wide |
| hierarchical navigation example | `Isaac-Navigation-Flat-Anymal-C-v0`, `isaaclab_tasks/manager_based/navigation/config/anymal_c/` at v2.3.2 | read in the pinned checkout |
| detector | OWLv2 CLIP B/16 ensemble, Hugging Face `google/owlv2-base-patch16-ensemble` revision `cfd3195ba4ea9592eec887ded089f4c08eff231d` | `google-research/scenic` `scenic/projects/owl_vit/README.md`, "License": "Both the code and the model checkpoints are licensed under the Apache 2.0 license", linking the repository `LICENSE`. The Hugging Face repository has no license file of its own, only an `apache-2.0` metadata tag. |

Isaac Lab 3.0.0 (Isaac Sim 6.2, Python 3.12) was released 2026-10-20 and moved the navigation example to `contrib/`; it is the fallback if 5.1.0 fails on Ubuntu 24.04.

## Step 1: Isaac setup job

| step | command | outcome |
| :--- | :--- | :--- |
| queue | `jobs/010-isaac-setup.toml` on `dell-jobs` (`1184cbf`), `bash jobs/010_setup_isaac.sh`, `gpu = true`, `ros = false` | queued 14:40 |
| inherited ROS 2 paths | `tr '\0' '\n' < /proc/<runner pid>/environ` | the runner's login shell sources ROS 2 Jazzy, so every job inherits a Python 3.12 `PYTHONPATH`, `LD_LIBRARY_PATH` and `AMENT_PREFIX_PATH` (`runner.py` `job_env` removes only `ROS_*`, `RMW_*`, `GZ_*` and DDS variables). `jobs/common.sh` now unsets them and drops `/opt/ros` from `PATH` (`a132dbd`, pushed before the job started) |
| attempt 1 | job 010 | **stopped by hand** (KILL file) at 14:49: `rsl-rl-lib==3.1.2` asks for `tensordict>=0.7.0`; pip picked tensordict 0.15.4 (needs `torch>=2.11.0a0`) and began downloading torch 2.15.0 over the `torch==2.7.0` that `isaacsim-core 5.1.0` pins. Fix: `jobs/constraints.txt` through `PIP_CONSTRAINT`: torch 2.7.0, torchvision 0.22.0, torchaudio 2.7.0, tensordict 0.8.3 |
| attempt 2 | job 010 | **failed**, exit code 141 (SIGPIPE) after 3 s: `ldd --version \| head -1` under `set -o pipefail`. Attempt 1 passed the same line by timing luck. Fix: `sed -n 1p` |
| attempt 3 | job 010 | **failed**: Isaac Sim 5.1.0, torch 2.7.0+cu128 and CUDA work, but `isaaclab` did not install: building `flatdict==4.0.1` (pinned by `isaaclab`) needs `pkg_resources`, which the newest setuptools in pip's isolated build environment no longer has. pip also reported `packaging 26.3` (Isaac Sim pins 23.0) and `psutil 7.2.2` (Isaac Sim pins 5.9.8; `ipython`, pulled in by `isaaclab_mimic`, wants 7 or later). Fix: constraints `packaging==23.0`, `psutil==5.9.8`, `setuptools<80` |
| attempt 4 | job 010 | **failed** the same way: `PIP_CONSTRAINT` did not reach the isolated build environment. Fix: `pip install --no-build-isolation flatdict==4.0.1` against the venv's setuptools 79.0.1 before `isaaclab.sh --install rsl_rl` |
| attempt 5 | job 010 | install and report complete (output below), then **stopped by hand** at 15:07: `SimulationApp.close()` hung for more than 8 minutes after the report was written. Fix: `go2x/isaac/shutdown.py` ends every Isaac script with `os._exit` after its outputs are flushed |

Report from attempt 5 (`env_report.json`):

| item | value |
| :--- | :--- |
| GPU | NVIDIA RTX 4000 Ada Generation, 20475 MiB, driver 595.99.02 (Vulkan, as seen by Isaac Sim) |
| CPU, RAM (as seen by Isaac Sim) | Intel Core Ultra 9 285K, 24 cores, 63747 MB |
| Python | 3.11.17 |
| PyTorch | 2.7.0+cu128 (CUDA 12.8 build), `torch.cuda.is_available()` = True |
| Isaac Sim | 5.1.0.0 (pip) |
| Isaac Lab | `isaaclab` 0.54.2, `isaaclab_tasks` 0.11.12, `isaaclab_rl` 0.4.7 (repo v2.3.2 `37ddf62`) |
| RSL-RL | `rsl-rl-lib` 3.1.2 |
| unitree_rl_lab | 0.2.1 (`4960b84`) |
| Go2 tasks registered | `Unitree-Go2-Velocity`, `Isaac-Velocity-Flat-Unitree-Go2-v0`, `Isaac-Velocity-Rough-Unitree-Go2-v0` (and Play variants) |
| navigation tasks registered | `Isaac-Navigation-Flat-Anymal-C-v0` (and Play) |
| headless start | Isaac Sim started headless with `OMNI_KIT_ACCEPT_EULA=YES`; it saw the RTX 4000 Ada and skipped the Intel integrated GPU |
| attempt 6 | job 010 | **succeeded** at 15:10 (12 s, every step cached). Remaining pip conflict, not fixed: `fastapi 0.115.7` (Isaac Sim) wants `starlette<0.46`, but Isaac Lab's dependencies installed 0.49.1. Isaac Sim also warns that the CPU power profile is "powersave" (not changed; a system setting) |

## Step 2: walking policy, Task 1

| step | command | outcome |
| :--- | :--- | :--- |
| task config | `go2x/isaac/walk_env_cfg.py`, `go2x/isaac/actuators.py`, `config/walk.yaml` | `unitree_rl_lab` `RobotEnvCfg` with: Go2 from the `unitree_ros` URDF, flat plane, no height scanner, plain floor and light (no Nucleus material or sky texture); velocity targets vx -0.5 to 1.0, vy -0.5 to 0.5, wz -1.0 to 1.0 (curriculum starts at ±0.1 m/s as in `unitree_rl_lab`); friction 0.3 to 1.2, base mass -1 to +3 kg, motor strength 0.8 to 1.2 (new actuator subclass), kp and kd scale 0.8 to 1.2 (`randomize_actuator_gains`), action delay 4 to 8 physics steps (20 to 40 ms, `DelayedPDActuator`), pushes every 5 to 10 s up to 0.5 m/s, observation noise as in `unitree_rl_lab` |
| short training, job 020 attempt 5 (attempts 1 to 4 skipped with job 010) | `bash jobs/020_walk_short.sh` (`train_walk.py --minutes 15 --chunk 25`) | **succeeded**, 15:10 to 15:25: 850 PPO iterations from a random initialization, 4096 environments, 1.05 to 1.18 s per iteration (about 2.2 million environment steps per minute). Rewards logged. By iteration 200 about 95 % of episodes reach the 20 s time limit; about 5 % end on `bad_orientation` (a fall) through iteration 850. Linear tracking reward rises from 0 to 1.3 (of a 1.5 maximum per second). Policy exported (TorchScript and ONNX). |
| joint order | `robot_info.json`, `deploy.yaml` from job 020 | Isaac Lab orders the 12 joints by type (FL, FR, RL, RR hips, then thighs, then calves); `joint_ids_map = [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]` to the SDK order. Default pose, hard and soft limits recorded in `docs/LOWLEVEL_INTERFACE.md` |
| finding: logging artifact | training curve of job 020 | mean reward and episode length show a sawtooth with a 25-iteration period. RSL-RL 3.1.2 restarts its episode statistics at every `learn()` call, and the short job called it every 25 iterations. The per-term rewards (`Episode_Reward/*`, from the environment) are unaffected. Fix: one `learn()` call unless a time limit is set (`4e628d7`) |
| full training, job 030 attempt 1 | `bash jobs/030_walk_full.sh` | **stopped by hand** after 30 s: it had checked out `66580cd`, 9 s before the logging fix was pushed. Re-queued as attempt 2 |
| navigation smoke test, job 025 attempt 1 | `bash jobs/025_nav_smoke.sh` | **failed** in 10 s: `Unable to find source prim path: '/World/envs/env_.*/Obstacles'`. Obstacle and building prims now sit directly under each environment (`534a2df`) |
| navigation smoke test, job 025 attempt 2 | `bash jobs/025_nav_smoke.sh` (256 environments, 6 min) | **succeeded** but with errors in the log: `Filter pattern '/World/envs/env_*/Obs_chair_red_0_seat' did not match the correct number of entries (expected 4864, found 256)`. PhysX filtered contact reporting needs one sensor body per environment; the sensor covered all 19 robot bodies, so the force-based collision check read nothing. Everything else ran: 70 PPO iterations, 5.5 s per iteration, 4.0 GiB GPU memory, all seven reward terms and six termination terms logged. Fix: one filtered contact sensor per robot body (`b2aa16b`) |
| navigation smoke test, job 025 attempt 3 | same | **succeeded**, no filter errors: force-based collision terminations now appear (up to 0.11 of episodes). 60 iterations at 7.0 s each (the 19 sensors add about 1.5 s). With an untrained policy, 0.70 to 0.79 of episodes end by leaving the square (the mean action is a 0.4 m/s forward walk in a 3 m square) and 0.2 to 0.3 by footprint collision |
| full training, job 030 attempt 2 | `bash jobs/030_walk_full.sh` (3000 iterations, one `learn()` call) | **succeeded**, 15:42 to 16:35 (3163 s, 1.05 s per iteration), commit `4022406`. Mean of the last 50 iterations: 0.996 of episodes reach the 20 s limit, 0.004 end on `bad_orientation`; velocity curriculum at its full range (level 1.0); linear tracking reward 1.40 of 1.5 per second; tracking error logged by Isaac Lab 0.22 m/s (xy) and 0.27 rad/s (yaw), over training targets with pushes and randomization. Curve: `results/phase2/walk_train/training_curve.png`. Exported policy: `results/phase2/walk_train/policy.onnx`, `deploy.yaml` |

## Steps 3 to 6: atrium, camera, navigation, language, MuJoCo (code and CPU tests)

CPU tests run in this session in a scratchpad environment: CPython 3.11.17 (uv 0.12.24), numpy 2.4.6, scipy 1.17.1, PyYAML, torch 2.7.0+cpu, `rsl-rl-lib` 3.1.2, `tensordict` 0.8.3, `mujoco` 3.16.0, pytest; ROS 2's `PYTHONPATH` removed (it broke pytest with `No module named 'lark'`).

| step | command | outcome |
| :--- | :--- | :--- |
| square check and building export | `python scripts/atrium/convert_building.py --tourbot <clone>` | `aeb_level1.sdf` at `2757cf16`, SHA-256 `651f19e7...f387`: 4269 boxes (wall 3029, glass 544, rail 336, low 319, closed_door 21, stairs 16, barrier 4). None inside or touching the square. Nearest: `glass_3485` 2.810 m from the center (0.994 m from the edge), `rail_4163` 2.949 m, `rail_4162` 2.955 m. 97 within 3 m of the edge, written to `config/building_level1_near_square.json`; report in `results/phase2/square_check.json` |
| finding: building outside the drawn area | same numbers | the drawn image covers the square plus 0.3 m; the nearest box is 0.99 m outside the edge, so the static channel never contains building boxes. The building appears only in the rendered camera view |
| camera pose | `coverage_check()` over hfov 55 to 90 degrees and three aim points | 90 degrees at (3.0, 0.0, 4.0) covers everything but leaves the square about 300 by 200 pixels (2 pixels per 5.7 cm at the far edge). Chosen: 60 degrees aimed at the center, 3.4 pixels per 5.7 cm at the far edge, every check point at least 33 pixels inside the image (`results/phase2/camera_coverage.json`) |
| robot footprint | Unitree Go2 product page, `https://www.unitree.com/go2` | "Dimension of standing: 70cm x 31cm x 40cm" for Air, Pro and EDU; footprint 0.70 by 0.31 m |
| scene sampler | 300 layouts, seed 0 | 2.9 ms per layout. Most rejections are "start too close to target" in the small square. Targets: chair 133, person 126, table 41 |
| evaluation scenarios | `python scripts/atrium/make_scenarios.py` | 200 scenarios, seed 20261108: targets chair 110, table 24, person 66; 340 chairs, 94 tables, 207 people of whom 50 walk. Fewer walk than the 0.5 setting because a walking path that crosses furniture is redrawn, and a redraw may come out standing |
| tests | `pytest -q tests/` | 60 passed. First runs found two code bugs and two test bugs: (1) the camera feed treated empty buffer slots as ready frames and returned blank images; fixed. (2) `np.cross` on 2-D vectors is deprecated in numpy 2; replaced. Test bugs: a reachability test whose goal region reached through a wall, and a float compared with `==` |
| CNN policy | `go2x/nav_policy.py` | RSL-RL 3.1.2 has only MLP actor-critics (`rsl_rl/modules/actor_critic.py` asserts 1-D observations); `CNNActorCritic` subclasses it and replaces the actor and critic networks; registered into the runner's namespace |
| MuJoCo model | `unitreerobotics/unitree_mujoco` `1eb6642` (BSD-3-Clause), `unitree_robots/go2/go2.xml` | torque motors in SDK order with the URDF effort limits (23.7, 45.43 N m); joints carry damping 0.1, friction loss 0.2 and armature 0.01, which the Isaac model does not; `scene.xml` sets no timestep (MuJoCo default 0.002 s), so the check sets 0.005 s. CPU smoke run of the script on job 020's policy, 2 episodes of 2 s: runs, no falls (numbers not meaningful) |

## Step 7: Isaac Sim rendering crashes on the Dell's driver (2026-11-02)

| step | command | outcome |
| :--- | :--- | :--- |
| jobs 040, 045, 065 | runner, after job 030 | **failed** within 7 s, exit code 139 (segmentation fault). All three start Isaac Sim with `enable_cameras` (the `isaaclab.python.headless.rendering.kit` experience). Native stack: `UsdManager::createHydraEngine` then `destroyHydraEngine` into `librtx.scenedb.plugin.so`: the RTX engine fails to start and crashes while cleaning up. Python stack (`py-spy` file written by the crash reporter): `SimulationApp.__init__` > `create_new_stage`. No error is logged before the crash. Jobs without cameras (010 to 030, 048, 050, 060) were not affected |
| stray process | `nvidia-smi --query-compute-apps`, `ps` | job 010 attempt 5's `report_env.py` (the hung `SimulationApp.close()`, step 1) was still running after 7 hours, holding 231 MiB of GPU memory and the Kit key-value database lock ("another kit process is locking it"). Stopped with `kill` |
| probe, job 041 | `bash jobs/041_render_probe.sh`: one camera frame under each launch setting | the default, `VK_ICD_FILENAMES` set to the NVIDIA driver only (hiding the Intel GPU), `DISPLAY` unset, and both: **all four crash** the same way. A fifth variant (`--/renderer/multiGpu/enabled=false`) did not run (an argument-parsing slip in the job script) |
| cause | GitHub `isaac-sim/IsaacSim` discussion #648 and issue #687 | the same `librtx.scenedb` crash on 595-series drivers. A maintainer (discussion #648): driver 595.71.05 "is not validated for Isaac Sim 5.1.0" and the supported Linux driver is 580.65.06. `--/rtx/verifyDriverVersion/enabled=false` did not help the reporter. The Dell has 595.99.02. Changing the driver is a system change outside this session |
| decision | | Isaac Sim keeps doing all physics and training (unaffected). Every image and video is drawn with MuJoCo 3.16.0's OpenGL renderer through the X display the runner passes in `DISPLAY` (GLFW). EGL failed here, outside and inside the sandbox. MuJoCo draws Unitree's Go2 model (`unitree_mujoco` `go2.xml`) posed at recorded Isaac states, so the motion shown is the Isaac motion. `go2x/mj_scene.py`; `tests/test_mj_scene.py` checks that the MuJoCo camera projects world points to the same pixels as `Camera.project` and the floor homography (within 1e-6 pixel) |
| walking video | `eval_walk.py --trajectory`, then `scripts/render_walk_video.py` | environment 0's base pose and 12 joint angles every policy step; chase camera, checker floor (0.5 m squares), velocity target in the header |
| scene renders (Task 3) | `scripts/render_scenes_mujoco.py` | the same sampler seed, part geometry (`object_parts`, `part_poses`), building boxes, camera model and output files as `scripts/isaac/render_scenes.py`; the robot stands at the start pose after a 1 s PD hold in MuJoCo. Local check of 2 scenes: chairs, table, people and the robot appear where the ground truth puts them |
| navigation videos | `record_nav_videos.py` (Isaac, no camera) saves poses; `scripts/render_nav_videos.py` draws them | each frame: the fixed camera view with people moving, and the controller's drawn input turned 90 degrees to match the camera view |
| finding: `scene.xml` has steps | `unitree_mujoco/unitree_robots/go2/scene.xml` | Unitree's scene places a row of steps and boxes from x = 1.2 m. The Task 6 check loaded it, so a robot walking forward for 20 s would have hit them. All MuJoCo use now builds a flat scene from `go2.xml` |
| walking evaluation, job 040 attempt 3 | `bash jobs/040_walk_eval.sh` | **succeeded** at 22:29 (commit `8cf707b`): 100 episodes in Isaac Lab, then the 20 s video drawn with MuJoCo (`results/phase2/walk_eval/walk_20s.mp4`, checked by eye) |
| scene renders, job 045 attempt 3 | `bash jobs/045_render_scenes.sh` | **succeeded** in 17 s (commit `8cf707b`): 60 scenes with ground truth (`results/phase2/scenes/`) |

## Step 8: navigation training (2026-11-02)

| step | command | outcome |
| :--- | :--- | :--- |
| baselines, job 048 | `bash jobs/048_nav_baselines.sh` (200 scenarios, 50 in parallel) | **succeeded** in 66 s. Results in `results/phase2/nav_baselines/` (numbers in `docs/PHASE2_SIM.md`) |
| training attempt 1, job 050 | `bash jobs/050_nav_train.sh` (1024 environments, 1500 iterations) | **succeeded**, 16:37 to 22:11 (20,017 s, 12.9 to 13.1 s per iteration). Success rose to about 0.12 of episodes by iteration 900 and stayed there. The action noise fell from 0.50 to 0.07 by iteration 600. Episodes ending outside the square: 0.16 at iteration 0, 0.68 at 75, 0.35 at 1499 |
| evaluation attempt 1, job 060 | `bash jobs/060_nav_eval.sh` | **succeeded**: success 0.125, 0.365 out of the square, 0.27 time limit (`results/phase2/nav_eval_attempt1/`) |
| finding: leaving the square was free | reward terms in `config/nav.yaml` | leaving the square ended the episode with no penalty, while a collision cost -10 and every step cost -0.02. The policy learned to end episodes by walking out. Changes for attempt 2 (`b448af5`): an `out_of_square` reward term of -10 (logged separately); the critic also reads 6 critic-only values (front-to-target vector in the robot frame, distance to the target, planar velocity, yaw rate), while the policy input stays the image plus the previous action; entropy coefficient 0.005 to 0.01 |
| training attempt 2, job 050 | same | **failed** after 35 s: CUDA out of memory. The critic group repeated the image, so the rollout buffer held a second 4.3 GB copy (1024 environments, 32 steps, 32,768 values). Fix (`229521c`): the critic group holds only the 6 values and the critic reads the policy group followed by it |
| training attempt 3, job 050, check at iteration 300 | `bash jobs/050_nav_train.sh` (started 22:29:47, 12.6 s per iteration) | running. Leaving the square fell from 0.58 of episodes (iteration 25) to 0.005 to 0.01 (iterations 175 to 303); attempt 1 had about 0.61 at iteration 300. Training success rose to 0.15 to 0.18 by iteration 75 and stayed there to iteration 303; reaching the 20 s limit rose to 0.62; footprint collisions 0.17 to 0.29. The progress term averages 1.04 per episode (Isaac Lab logs per second of the 20 s limit; 0.052 x 20), or 0.26 m of net approach, while 512 sampled starts are 1.55 m from the target on average (median 1.48 m, 10th to 90th percentile 1.08 to 2.09 m). Training success uses sampled actions (noise standard deviation 0.24 of the action range); job 060 measures the noise-free policy. Left running |
| training attempt 3, job 050 | same (commit `8cf707b`) | **succeeded**, 22:29 to 03:44 (5.25 h). Training success stayed at 0.15 to 0.20 from iteration 100 to 1500; the 20 s limit ended 0.50 to 0.62 of episodes; leaving the square stayed at 0.005 to 0.015. Curve: `results/phase2/nav_train_attempt3/nav_training_curve.png` |
| evaluation attempt 3, job 060 | `bash jobs/060_nav_eval.sh` (commit `a671952`) | **succeeded** in 97 s: success 0.170 ± 0.027, collision 0.310 ± 0.033, out of the square 0.005, 20 s limit 0.510 ± 0.035 (n = 200). Below both baselines (0.580 and 0.695). Table in `docs/PHASE2_SIM.md` |
| videos attempt 2, job 065 | `bash jobs/065_nav_videos.sh` | **failed** after 18 s: `RuntimeError: Inplace update to inference tensor outside InferenceMode` in `env.reset()` between rollouts. The first rollout ran inside `torch.inference_mode()`, so the state tensors it made could not be written outside it. Fix (`09edccd`): the reset loop runs inside inference mode too. Re-queued as attempt 3 |
| videos attempt 3, job 065 | same (commit `09edccd`) | **succeeded** in 66 s: six videos, one success and one failure per controller (`results/phase2/nav_videos/`, 1.1 MB). Checked by eye: in the policy failure (scenario 0) the robot turns in place near its start for 20 s |

## Step 9: language target and MuJoCo check (2026-11-03)

| step | command | outcome |
| :--- | :--- | :--- |
| grounding, job 070 attempt 2 | `bash jobs/070_grounding.sh` (commit `a671952`) | **succeeded** in 32 s: OWLv2 on job 045's 60 MuJoCo scenes, 132 prompts, 0.614 ± 0.043 correct. Colored chairs 0.851, table 0.960, "go to the chair" 0 of 15, "go to the person" 0 of 25 (`results/phase2/grounding/`) |
| finding: people are cylinders | `results/phase2/grounding/grounding_examples/` | the scene models a person as an orange cylinder, in Isaac Lab and in the renders, so an open-vocabulary detector has no person to find. The person result measures the scene model, not OWLv2 |
| MuJoCo check, job 080 attempt 2 | `bash jobs/080_mujoco_check.sh` (commit `a671952`) | **succeeded** in 77 s: 100 episodes, 0 falls; tracking errors vx 0.125, vy 0.088 m/s, wz 0.194 rad/s, about twice Isaac Lab's (`results/phase2/mujoco_check/`) |

## Step 10: why the navigation policy stalls, and attempt 4 (2026-11-04)

| step | command | outcome |
| :--- | :--- | :--- |
| probe the trained actors | `python scripts/probe_nav_policy.py <model_final.pt> world` (CPU; 512 sampled starts, robot turned so the target sits at each bearing from -180 to 180 degrees, noise-free image) | neither policy steers by the image. Attempt 3: vx 0.02 to 0.03 m/s and wz +0.62 to +0.64 rad/s at every bearing (it turns left in place); attempt 1: vx about 0.35 m/s and wz about -0.26 rad/s at every bearing (it drives a right-hand arc, which explains its exits from the square). Standard deviation over starts about 0.1 m/s and 0.27 rad/s |
| can the network read the bearing at all? | `python scripts/bearing_supervised_check.py 3000 world` (CPU, the same `ImageMLP` trained by regression on noise-free drawn images, 128 per step) | slowly and poorly in the world view: held-out median bearing error 77 degrees after 250 steps, 30 after 1500, 23 after 2750. The world view puts the robot anywhere in the image at any heading, so the network has to locate a 12 by 5 pixel robot, read its heading marker and relate it to the target |
| robot view | `go2x/atrium/topdown.py` robot view (`config/topdown.yaml` `robot_view`): the same four channels on a grid centered on the robot's estimated pose, robot facing the top of the image, 3.0 m each way (9.4 cm per pixel), ground outside the square drawn as static | `python scripts/bearing_supervised_check.py 1000 robot`: median error 1.6 degrees after 250 steps, 0.8 after 1000. 4 new CPU tests (placement, independence of heading and position, outside-square marking, unchanged noise draws); 66 pass |
| change for attempt 4 (`a75cd1d`) | `config/nav.yaml` `policy.image_view: robot` | the policy reads the robot view; rewards, network, PPO settings and camera realism are as in attempt 3, so the image view is the only change. The baselines keep the world view (`eval_nav.py` sets it for them); their results (job 048) stand, because the camera noise draws are the same in both views. Job 065 records the policy and the baselines in separate runs |
| re-queue | `dell-jobs` `62b0f55`: 050, 060, 065 at attempt 4 | 050 started 13:38 |
| check at iteration 100 | training log | training success 0.60 (attempt 3: 0.15 at the same iteration); the 20 s limit fell to 0.001 |
| training attempt 4, job 050 | `bash jobs/050_nav_train.sh` (commit `a75cd1d`) | **succeeded**, 13:38 to 19:43 (6.1 h). Last 50 iterations: success 0.771, footprint collision 0.166, force collision 0.058, 20 s limit 0.001. Action noise standard deviation grew from 0.5 to 3.27 (actions are clipped, so the entropy bonus raises it at no cost); training success flat at 0.73 to 0.77 from iteration 300 |
| evaluation attempt 4, job 060 | `bash jobs/060_nav_eval.sh` | **succeeded** in 55 s: success 0.780 ± 0.029, collision 0.200 ± 0.028, fall 0.010, out of the square 0.005, 20 s limit 0.005 (n = 200). Above A* (0.695 ± 0.033) and straight to target (0.580 ± 0.035) |
| probe of attempt 4 | `python scripts/probe_nav_policy.py <model_final.pt> robot` | steers by the image: wz has the bearing's sign (-0.84 rad/s at -90 degrees, +1.00 at +90), vx about 0.55 m/s within 30 degrees of ahead, under 0.02 m/s beyond 120 degrees |
| videos attempt 4, job 065 | `bash jobs/065_nav_videos.sh` (two recording runs) | **succeeded** in 40 s: six videos (`results/phase2/nav_videos/`). Policy success, scenario 0, checked by eye: the robot turns to the yellow target chair and stops at it. These replace attempt 3's videos, which stay in commit `a92dcc8` |

## Step 11: wireless robot runner, MuJoCo live simulation and digital twin (2026-11-06)

Aidan lifted the ban on hardware deployment on 2026-11-06 ("i am lifting the ban on hardware deployment, but i will tell you when we will do it") and asked for wireless runs from the laptop with odometry and a digital twin recorded, and for the live simulation in MuJoCo. No command has been sent to the robot yet.

| step | command | outcome |
| :--- | :--- | :--- |
| library check | `unitree_webrtc_connect` `e0abc57` source | Wi-Fi methods `LocalSTA` (same network, robot's address) and `LocalAP` (robot hotspot); low state on `rt/lf/lowstate` with 12 motor angles in SDK order. Only velocity requests go over WebRTC, so a wireless run uses Unitree's walking controller, not the Task 1 walking policy (joint commands need wired DDS) |
| export actor | `python scripts/export_nav_actor.py --checkpoint <050 attempt 4 model_final.pt> --out results/phase2/nav_policy/nav_actor.pt` | 1.2 MB, 302,675 values; loads with torch alone (`go2x/nav_net.py` split from `go2x/nav_policy.py` so RSL-RL is not needed) |
| deploy code | `go2x/deploy/` (nav runtime, MuJoCo backend, Wi-Fi backend), `scripts/robot/` (runner, twin renderer, link check, scenario list, laptop setup), `config/safety_limits_nav.yaml` (proposed), `docs/ROBOT_RUN.md` | Phase 1 code unchanged: the Wi-Fi backend subclasses `go2x/robot/webrtc.py`, the runner uses `go2x/robot/safety.py` and its logger. 7 new CPU tests (odometry mapping, input layout, scene, an end-to-end mock run, refusal of walking people on the robot); 72 pass |
| first MuJoCo run | `run_nav_policy.py --backend mujoco --scenario 0 --record` | crashed in the physics thread (`import mujoco.viewer` inside the function made `mujoco` local); the watchdog stopped the run with zero velocity, as designed. Fixed; rerun: success in 8.5 s of policy time |
| digital twin check | `render_twin.py` on that run | the twin, built from the log alone, matches the simulation's own video in pose and leg angles, so the odometry mapping undoes the backend's offset (1.7 m, -0.6 m, 0.9 rad) |
| live run, scenario 21 | `--viewer --record` | the viewer opens on the Dell's display. Stopped by the footprint check: the robot cut a chair corner (Isaac: success) |
| limits comparison | 30 scenarios x 2 limit sets, MuJoCo | success 0.733 ± 0.081 with the proposed robot limits, 0.833 ± 0.068 with the training limits, 0.767 ± 0.077 in Isaac on the same 30; not separable at n = 30 (table in `docs/PHASE2_SIM.md`) |
| live recordings | `--viewer --record`, scenarios 2, 23, 101 | all successes (8.8, 8.3, 11.6 s of video); `results/phase2/live_sim/`, plus the twin of scenario 23 |
| laptop setup check | `bash scripts/robot/setup_laptop.sh` in an empty folder | sparse clone of `unitree_mujoco` at `1eb6642`; `go2.xml` compiles with its 16 meshes |

## Step 12: overhead camera (2026-11-06)

Aidan: the camera goes on the second floor, set up from the MacBook (24 GB); a USB webcam on a long cable; OWLv2 for obstacles with tags as a backup; printed ArUco tags for calibration.

| step | command | outcome |
| :--- | :--- | :--- |
| robot link check | `ip -brief addr`, carrier of both Ethernet ports | **not run**: the robot is not reachable from the Dell. The built-in port is on the campus network (10.131.x.x); the USB-C adapter (`cdc_ncm`) has no link. Checks moved to the MacBook (commands in the reply to Aidan) |
| Mac check | `mujoco` 3.16.0 `rendering/classic/gl_context.py` | offscreen rendering on macOS defaults to CGL, which works off the main thread (recording, twin). The live viewer needs `mjpython`; the MuJoCo backend now runs on without it (`2ad82ec`) |
| packages | `uv pip install opencv-python==4.11.0.86 transformers==4.57.1` (scratch environment) | ArUco is in the main OpenCV package. OWLv2 then failed to import: a `torchvision` 0.22.0 build that did not match the CPU torch (`operator torchvision::nms does not exist`). Removed (nothing here uses it); `env/requirements-robot.txt` warns against it |
| vision code | `go2x/vision/` (camera model with distortion, tag detection and plane geometry, OWLv2, perception, camera mode for the runner, frame sources, drawing), `scripts/camera/` (tags PDF, lens and pose calibration, preview, camera list), `config/tags.yaml` | tag PDF checked by rasterizing at 100 dpi: the robot tag measures 18.01 cm (want 18.00) and the board's 24 corners are all detected |
| first OWLv2 run | `--camera sim --obstacles owl` | detector thread crashed on import; the runner waited until the script cap without standing the robot up. Fixed: thread errors stop the run with the reason; the time cap starts after the camera is ready |
| flicker | scenario 3 detections | the chair (scores 0.21 to 0.26, threshold 0.20) was in 3 of 6 passes; the layout lost it. Fixed: 6 s persistence, 3 passes before start |
| colored targets | scenarios 9 and 14 | the target chair was found only by the colored query; the plain query missed it. Fixed: every pass asks for each color, and colored detections are chairs of that color |
| exit crash | end of an OWLv2 run | `terminate called without an active exception` after all files were written (detector thread still running). Fixed: threads are joined |
| results | 4 OWLv2 runs, 1 tag run, calibration tests | table in `docs/PHASE2_SIM.md` "Overhead camera in MuJoCo"; 82 tests pass |

## Problems found (Phase 2)

**Isaac Sim 5.1 cannot render on the Dell's driver.** Every app with cameras crashes in `librtx.scenedb.plugin.so` at start (exit code 139) on driver 595.99.02; the supported Linux driver is 580.65.06 (step 7). Physics and training are unaffected. All images and videos are drawn with MuJoCo from recorded Isaac states. A driver change is a system setting and was not made.

**The navigation policy was below both baselines until attempt 4. Resolved on 2026-11-04.** Attempts 1 and 3 did not steer by the world-view image (step 10). With the robot view, attempt 4 reaches 0.780 ± 0.029 against 0.695 for A*.

**Action noise grows without limit in attempt 4.** The noise standard deviation rose from 0.5 to 3.27: actions are clipped, so the entropy bonus raises the noise at no cost. The evaluation uses the noise-free action. A smaller entropy coefficient or a squashed action distribution would stop it; not changed.

**People are cylinders, so person prompts cannot succeed.** OWLv2 grounds 0 of 25 "go to the person" prompts. The scene models a person as an orange cylinder; the result measures the scene model, not the detector.

**"go to the chair" fails where colored-chair prompts work.** 0 of 15 against 0.851 of 67 for "go to the {color} chair". Not yet checked.

**The two collision checks disagree in 0.22 to 0.38 of episodes.** The footprint check fires more often than the contact-force check for every controller. A likely reason, not yet checked: the footprint rectangle (0.70 by 0.31 m) is larger than what the robot puts near the floor.

**The walking policy tracks about twice as badly in MuJoCo.** Tracking errors vx 0.125 against 0.060 m/s and wz 0.194 against 0.101 rad/s, with no falls in 100 episodes. Unitree's MuJoCo model adds joint damping, friction loss and armature that the Isaac model lacks; their effect is not yet checked.

**Fewer people walk than configured.** 50 of 207 people in the evaluation scenarios walk, against a 0.5 setting, because a walking path that crosses furniture is redrawn and the redraw may come out standing.

**A pip conflict remains in the Isaac environment.** `fastapi 0.115.7` (Isaac Sim) wants `starlette<0.46`; Isaac Lab's dependencies install 0.49.1. No job has failed from it. Isaac Sim also warns that the CPU power profile is "powersave" (a system setting, not changed).

**The wireless link and the proposed robot limits are untested on the robot.** `go2x/deploy/wireless.py` changes only the connection method and adds the low-state subscription to the backend tested over Ethernet on 2026-10-17. `config/safety_limits_nav.yaml` needs Aidan's approval before the first run.

**OWLv2 positions are biased toward the camera.** In MuJoCo the detected obstacle centers were 0.1 to 0.35 m nearer the camera than the truth, which shrinks the virtual bumper's margin. `config/tags.yaml` `owl.half_depth_m` corrects it; it has to be tuned on the real camera with a measured chair.

**Undetected obstacles are invisible.** The policy and the virtual bumper know only what the camera pipeline reports.

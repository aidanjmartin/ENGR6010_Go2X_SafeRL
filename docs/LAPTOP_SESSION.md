# Laptop session brief: Go2 X Phase 1 motion tests

This brief is for Claude Code on Aidan's Ubuntu 24.04 laptop (NVIDIA RTX 4000 Ada Laptop GPU), with the Unitree Go2 X connected by a USB-C Ethernet tether. Read all of it before doing anything.

## Context

- **Project:** ENGR 6010 (AI in Robotics), MTSU. Phase 1 is due **Friday 2026-10-16**, and the robot is available for **a few hours that afternoon**. Goal 1 is the only goal that needs the robot: the laptop sends body-velocity commands that move the Go2 X, with logs to show it.
- **Project repo:** https://github.com/aidanjmartin/ENGR6010_Go2X_SafeRL. [`README.md`](../README.md) and [`LOG.md`](../LOG.md) record everything done so far on the Mac.
- **Reference repo, read-only:** https://github.com/aidanjmartin/ENGR6010_Exam1 at commit `90150da`. It is a graded submission. Never commit to it, push to it, edit its files, or create files inside its folder.
- **Robot:** Go2 X, not EDU. High-level commands only: body velocity and Unitree's predefined modes. No joint-level control, no user code on the robot, indoor use only. Never suggest a firmware update. Never send `Damp`, which removes leg torque and drops the robot.
- **What Exam 1 proved, read-only and over this same tether:**
  - The robot answers DDS. `capture_image.py` returned a camera frame, and `rt/lf/lowstate` arrived at about 26 Hz.
  - The robot answers WebRTC. The handshake is `data2 = 2` (firmware 1.1.11 to 1.1.14, so no AES key is needed), and `rt/lf/sportmodestate` arrived at 20 Hz.
  - No motion command has ever been sent.
- **Terms:**
  - **DDS** (Data Distribution Service): the publish/subscribe middleware Unitree's SDK uses, through CycloneDDS.
  - **WebRTC:** the browser real-time protocol the Unitree phone app uses. `unitree_webrtc_connect` speaks it from Python.
  - **Sport mode:** Unitree's onboard high-level locomotion controller. It accepts `Move(vx, vy, vyaw)`, `StandUp`, `BalanceStand` and `StopMove` requests.
  - **MCF:** Unitree's multi-control framework, used since firmware 1.1.7. It uses the same sport request topic, and the API ids we use are the same.

## Rules

1. **Show every command before running it and wait for Aidan's OK.** This always applies to anything that installs software, changes system configuration (`sudo`, `nmcli`), pushes to GitHub, or touches the robot.
2. **Aidan runs the motion scripts in his own terminal, not through your Bash tool.** The scripts ask him to type `yes` after the safety checklist, and they stop the robot on any key press. Both features need a real terminal. Give him the exact command, wait for him to paste the output (or read the log folder it names), then discuss.
3. **One script at a time.** Before each script, Aidan confirms out loud that there is 2 m of clear floor around the robot and that the handheld remote is in his hand and on. Never chain scripts.
4. **Change nothing in `config/safety_limits.yaml`** (0.2 m/s, 0.5 rad/s, rate limits, 0.5 s state watchdog, script time cap) unless Aidan asks.
5. **Make only the change discussed.** If something looks worth doing, mention it and stop.
6. **Check primary sources** (the library source at the pinned commit) before claiming how an API behaves, and name the file.
7. **Update `LOG.md` as you go,** using its existing step / command / outcome tables and its "Problems found" section.
8. **Commits and pushes must not credit Claude:** no Co-Authored-By lines and no "Generated with Claude Code".
9. **Never commit `.env`,** the robot's IP address, or its serial number. The logger drops the serial number field `sn`, and `data/` is gitignored.
10. **The phone app must be disconnected** before any WebRTC connection (go2_ros2_sdk README, cited in Exam 1 `q4_data_sim/data_collection_plan.md`).

## 1. Clone

If a "Ubuntu laptop setup" section already exists in `LOG.md` (from a session on 2026-10-15), steps 1 and 2 may be done already. Verify, then skip what is done.

```bash
git clone https://github.com/aidanjmartin/ENGR6010_Go2X_SafeRL ~/ENGR6010_Go2X_SafeRL
git clone https://github.com/aidanjmartin/ENGR6010_Exam1 ~/ENGR6010_Exam1_ref
git -C ~/ENGR6010_Exam1_ref checkout 90150da
```

Read `README.md`, `LOG.md` and `env/environment-ubuntu.yml` in the project. Read `q2_sdk/LOG.md` and `q3_apps/LOG.md` in Exam 1.

## 2. Environment

Follow the header of `env/environment-ubuntu.yml` and Exam 1's `q2_sdk/LOG.md`. Show each command first.

```bash
lsb_release -a; nvidia-smi; nproc; ip -brief link       # read-only inspection; report it
sudo apt install build-essential cmake portaudio19-dev    # as in Exam 1
# Miniconda only if conda is missing, as in Exam 1:
#   bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
git clone -b releases/0.10.x https://github.com/eclipse-cyclonedds/cyclonedds ~/go2_deps/cyclonedds
git -C ~/go2_deps/cyclonedds checkout 5041f35
mkdir -p ~/go2_deps/cyclonedds/build && cd ~/go2_deps/cyclonedds/build \
  && cmake .. -DCMAKE_INSTALL_PREFIX=../install -DBUILD_EXAMPLES=OFF && cmake --build . --target install
git clone https://github.com/unitreerobotics/unitree_sdk2_python ~/go2_deps/unitree_sdk2_python
git -C ~/go2_deps/unitree_sdk2_python checkout 814556d
cd ~/ENGR6010_Go2X_SafeRL
conda env create -p .venv -f env/environment-ubuntu.yml
CYCLONEDDS_HOME=~/go2_deps/cyclonedds/install .venv/bin/pip install -e ~/go2_deps/unitree_sdk2_python
.venv/bin/pip install -e . --no-deps
.venv/bin/python -c "import unitree_sdk2py, cyclonedds, unitree_webrtc_connect, torch; print(torch.cuda.is_available())"
.venv/bin/pip check
.venv/bin/pytest -q                                      # expect 15 passed
```

This is the first time both robot libraries share one environment. Exam 1 used Python 3.11 for DDS and 3.10 for WebRTC. Log any conflict as a finding.

## 3. Network

Create the connection only if it doesn't already exist (`nmcli con show`). The commands are the same as Exam 1's `q2_sdk/LOG.md`:

```bash
nmcli con add type ethernet ifname <adapter> con-name go2-eth \
  ipv4.method manual ipv4.addresses 192.168.123.99/24 ipv4.never-default yes ipv6.method link-local
nmcli con up go2-eth
ip -brief addr show <adapter>          # expect 192.168.123.99/24
ping -c 3 8.8.8.8                       # the normal internet route must be unchanged
```

**Find the robot.**
1. Exam 1 found the main board at the address Unitree documents for it; `probe_signaling.py` lists it as `192.168.123.161`. Try that first: `ping -c 3 192.168.123.161`.
2. Only if that fails, sweep the subnet without installing anything:
   `for i in $(seq 1 254); do ping -c1 -W0.2 192.168.123.$i >/dev/null && echo 192.168.123.$i; done`
3. Exclude `.99`, which is the laptop.

Put the address and adapter name in the untracked `.env` (copy `.env.example`): `GO2_IP=...` and `GO2_ETH_IFACE=<adapter>`.

## 4. Read-only state tests (no motion)

Reproduce Exam 1, then check the stream the safety layer depends on. Run one link at a time.

| # | command | expect (Exam 1) |
| :--- | :--- | :--- |
| 4a | `.venv/bin/python ~/ENGR6010_Exam1_ref/q3_apps/probe_signaling.py <GO2_IP>` | port 9991 open, `data2 = 2` |
| 4b | `.venv/bin/python ~/ENGR6010_Exam1_ref/q2_sdk/examples/wireless_controller_go2.py <adapter>` (Ctrl+C after a few seconds) | `LowState` at about 26 per second |
| 4c | `cd /tmp && ~/ENGR6010_Go2X_SafeRL/.venv/bin/python ~/go2_deps/unitree_sdk2_python/example/go2/front_camera/capture_image.py <adapter>` | one 1920 x 1080 JPEG, written to `/tmp` and not to either repo |
| 4d | `.venv/bin/python scripts/motion/00_read_state.py --backend dds --seconds 10` | **new**: `rt/lf/sportmodestate` over DDS was never read in Exam 1. Record the rate and the largest gap. |
| 4e | `.venv/bin/python scripts/motion/00_read_state.py --backend webrtc --seconds 10` | about 20 per second (Exam 1). It also prints the motion-switcher mode; record it. |

Run 4c from `/tmp` so its JPEG lands outside both repos. Exam 1's `webrtc_read_state.py` reads a `.env` inside the Exam 1 folder; don't create one there. Step 4e records the same state stream instead.

**Gate:** a backend may run motion scripts only if its `00_read_state` run received state with no gap over 0.5 s. If a backend gets no sport-mode state, record that and skip it. Its motion scripts would be stopped by the watchdog in any case.

## 5. Motion tests

Choose the first backend with Aidan. The suggested order is **DDS first**: it is Unitree's own SDK, its `Move` reports a send error code, and it needs no asyncio. Then WebRTC. For each backend, run in order:

| # | Aidan runs, in his own terminal | what it does |
| :--- | :--- | :--- |
| 1 | `.venv/bin/python scripts/motion/01_stand.py --backend dds` | StandUp, BalanceStand, hold 10 s at zero velocity |
| 2 | `.venv/bin/python scripts/motion/02_forward.py --backend dds` | forward 0.5 m at 0.2 m/s, then stop |
| 3 | `.venv/bin/python scripts/motion/03_rotate.py --backend dds` | rotate in place 90 deg at 0.3 rad/s, counterclockwise |
| 4 | `.venv/bin/python scripts/motion/04_lateral.py --backend dds` | 0.3 m to the robot's left at 0.15 m/s |
| 5 | `.venv/bin/python scripts/motion/05_square.py --backend dds` | 1 m square at 0.2 m/s with 90 deg turns, about 55 s |

Then repeat 1 to 5 with `--backend webrtc`.

**Before each script:** Aidan confirms 2 m of clear floor and the remote in hand, then types `yes` at the script's prompt. If the robot is already standing from the previous script, `--skip-stand` skips StandUp and BalanceStand.

**What each script does on its own:**
- Clamps every command to the Phase 1 limits and ramps it.
- Stops on any key, on Ctrl+C, on a 0.5 s state gap, on any exception, and at the time cap.
- "Stop" means five zero-velocity commands, then `StopMove`.
- Writes `data/<UTC time>_<script>_<backend>/` (`header.json`, `state.jsonl`, `commands.jsonl`, `events.jsonl`).
- Prints a summary: commanded vs. measured velocity, time to 10 % and 50 % of the commanded value, commanded travel vs. odometry displacement, and the stop reason.

**After each script:**
- Read the summary with Aidan.
- Add a row to `LOG.md` with the command, the stop reason, and the summary numbers.
- Agree on whether to go on.
- Re-run a script with `.venv/bin/python scripts/motion/summarize_log.py data/<folder>` if needed.

**Stop and discuss, without retrying blindly, if:**
- the robot does not move;
- it moves the wrong way;
- the stop reason is anything other than "end of script";
- a sport call prints a nonzero return code.

**Velocity frame (open question):** the frame of `SportModeState.velocity` (body or world) is not documented in the SDK source. The summary shows it both as reported and rotated world→body by the IMU yaw. After `03_rotate` and `05_square`, the column that stays near the commanded +0.2 m/s on every side of the square is the right one. Record the answer in `LOG.md`.

**If a Move is accepted but the robot does not walk:**
- Check the `mode` field from step 4.
- Check the WebRTC motion-switcher mode.
- Check the `StandUp` / `BalanceStand` return codes.
- Report them. Don't send mode switches (motion switcher api 1002), `Damp`, or any other sport API without discussing it first.

## 6. Record results

- **`LOG.md`:** a dated section "Robot session (2026-10-16)" with a machine line. Include one row per run: backend, script, stop reason, state rate, commanded vs. measured velocity, delay, displacement. Add every failure to "Problems found", and the velocity frame answer.
- **`README.md`:**
  - Fill in the "Results" table: one row per backend and script with measured velocity, rough delay and displacement.
  - Update the "Status" table, changing "untested on hardware" to what happened.
  - Update the "untested on hardware" docstrings in `go2x/robot/dds.py` and `webrtc.py` only to state what was tested and on what date.
- **Commit and push:** commit `LOG.md` and `README.md` (not `data/`), after showing Aidan the commands. Short summary CSVs or plots from the logs may go in `results/robot/` if Aidan wants them.

## 7. Optional: GPU training

Only if Aidan asks. Run `.venv/bin/python scripts/rl/train_ppo.py --timesteps 10000000 --n-envs <cores> --name full`, then `evaluate.py`, `record_video.py` and `plot_training.py` (see the README). Stable-Baselines3 PPO with an MLP policy is CPU-bound (`stable_baselines3/common/on_policy_algorithm.py:152`), so the GPU won't speed this run up much.

## If neither backend moves the robot

Stop. Don't improvise mode switches, other sport APIs, or library changes on the robot. Instead:

1. Collect what each attempt returned: the `StandUp`, `BalanceStand`, `Move` and `StopMove` codes, the state `mode` before and after, the WebRTC motion-switcher mode, and any traceback.
2. Help Aidan choose the next approach. Lay out the options with their risks and the primary source for each. Candidates to research, not to run:
   - the WebRTC normal-mode path (`examples/go2/data_channel/sportmode/sportmode.py`, which switches the motion mode);
   - the obstacle-avoidance move API (`OBSTACLES_AVOID_API` in `unitree_webrtc_connect/constants.py`);
   - the Go2 Wi-Fi hotspot (out of Phase 1 scope unless Aidan decides otherwise).
3. Record the failure fully in `LOG.md`. A well-documented negative result still counts as Phase 1 evidence.

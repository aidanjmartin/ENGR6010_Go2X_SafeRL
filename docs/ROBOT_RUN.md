# Running the navigation policy on the Go2 X over Wi-Fi

This is the procedure for the first real-robot runs of the Task 4 navigation policy (job 050 attempt 4). The laptop runs the policy and talks to the robot over Wi-Fi (WebRTC). Everything is recorded, and each run can be replayed as a digital twin in MuJoCo.

**Status: Plan, not yet run on the robot.** The software path is tested in MuJoCo and against the mock robot (`tests/test_deploy.py`). The Wi-Fi link (`go2x/deploy/wireless.py`) differs from the tested Ethernet backend only in its connection method and one extra subscription, and it has not been tried yet.

## What runs, and what does not

- **The navigation policy runs; the walking policy does not.** The navigation policy sends velocity targets (vx, vy, wz) at 10 Hz. On the robot, Unitree's own walking controller follows them (sport mode `Move`), as in the Phase 1 robot session. The Task 1 walking policy needs low-level joint commands, which go over the wired DDS (Data Distribution Service) network only. It is not part of a wireless run.
- **No camera.** The policy's input is drawn from the scenario file (where the chairs, table and people are) and the robot's odometry. The floor must match the scenario's setup sheet. The policy cannot see anything that is not in the file.
- **No walking people.** People in the scenario are standing (or there are none). The runner refuses scenarios with walking people on the robot.

## One-time laptop setup (Mac)

```bash
cd ENGR6010_Go2X_SafeRL
git fetch origin
git checkout phase2-rl-sim
git pull
.venv/bin/pip install -r env/requirements-robot.txt   # scipy, mujoco, pillow on top of the Phase 1 .venv
.venv/bin/pip install -e . --no-deps
bash scripts/robot/setup_laptop.sh                     # Unitree's MuJoCo Go2 model, pinned commit, into third_party/
```

The trained policies are in the repository: `results/phase2/nav_policy/nav_actor.pt` (navigation actor) and `results/phase2/walk_train/policy.pt` (walking policy, used only by the MuJoCo backend).

Check the laptop install without the robot (about 20 s, writes a video):

```bash
.venv/bin/python scripts/robot/run_nav_policy.py --backend mujoco --scenario 2 --record \
    --model_dir third_party/unitree_mujoco
```

On macOS the live viewer (`--viewer`) needs MuJoCo's `mjpython` launcher in place of `python`; `--record` works with plain `python`.

## Wi-Fi connection

Two ways, both from `unitree_webrtc_connect` (`e0abc57`):

| method | how | `--method` |
| :--- | :--- | :--- |
| same network | put the robot on a Wi-Fi network with the Unitree app; the laptop joins the same network; put the robot's Wi-Fi address in `.env` as `GO2_IP` (or pass `--ip`) | `sta` |
| robot hotspot | the laptop joins the robot's own Wi-Fi hotspot; no address needed | `ap` |

Exam 1 found the robot on firmware 1.1.11 to 1.1.14, which needs no AES key. If the firmware has been updated to 1.1.15 or later, the connection needs the per-device key in `.env` as `GO2_AES_KEY` (see the library's `examples/fetch_aes_key.py`).

## Robot-day procedure

1. **Link check, read-only** (sends nothing; the robot can lie down):
   ```bash
   .venv/bin/python scripts/robot/check_link.py --method sta --seconds 10
   ```
   It must print `LINK OK`, sport state near 20 messages per second, and 12 joint angles. Note the battery charge.
2. **Pick a scenario** that fits the floor:
   ```bash
   .venv/bin/python scripts/robot/list_scenarios.py --max-objects 1
   ```
   Start with one obstacle (for example scenario 2, one chair). Scenarios with walking people are left out.
3. **Set up the floor.** Print the sheet without connecting:
   ```bash
   .venv/bin/python scripts/robot/run_nav_policy.py --backend webrtc --scenario 2 --sheet-only
   ```
   Mark the 3.048 m (10 ft) square, its center (the origin) and its +x and +y directions. Place each object at its listed position and heading (chairs: 0.50 by 0.50 m footprint; table: 1.20 by 0.75 m; person: a standing person or a 0.5 m wide stand-in). Place the robot's base center at the start position, facing the start heading.
4. **Run:**
   ```bash
   .venv/bin/python scripts/robot/run_nav_policy.py --backend webrtc --method sta --scenario 2
   ```
   Type `yes` after checking the list it prints. The robot stands up, holds 3 s, then the policy runs for up to 20 s.
5. **Stops.** Any key or Ctrl+C stops the robot at once (zero velocity, then StopMove; never Damp). The runner also stops by itself:
   - on success (front within 0.5 m of the target, under 0.1 m/s);
   - when the footprint would touch an obstacle's footprint;
   - when the robot center leaves the square;
   - at roll or pitch above 0.8 rad;
   - when state stops arriving for 0.5 s;
   - after 20 s of policy time.
   Keep the handheld remote in hand.
6. **Digital twin video** of the run:
   ```bash
   .venv/bin/python scripts/robot/render_twin.py data/<run folder> --model_dir third_party/unitree_mujoco
   ```

## What a run records

Each run makes `data/<UTC time>_nav_policy_<backend>/` (gitignored):

| file | content |
| :--- | :--- |
| `header.json` | start time, host, library versions, safety limits, policy file |
| `state.jsonl` | every sport-mode state message: odometry position, body-frame velocity, IMU roll, pitch and yaw, body height (about 20 Hz) |
| `lowstate.jsonl` | every low-state message: 12 joint angles and speeds, IMU, foot force, battery |
| `commands.jsonl` | every command: what the policy asked for and what the safety layer sent |
| `events.jsonl` | stand, every policy step (pose, action, command, distance), stop with its reason |
| `run.json` | the scenario, the odometry-to-square mapping, the outcome |
| `policy.npz` | per policy step: time, pose in the square, odometry, speed, the drawn input image, action, command, target distance |
| `twin.mp4` | after `render_twin.py`: the run replayed on Unitree's Go2 model from the log alone |

## Safety limits (PROPOSED, need Aidan's approval)

`config/safety_limits_nav.yaml`: forward speed at most 0.5 m/s (the policy was trained to 0.8), sideways 0.3 m/s, turning 1.0 rad/s. Speed changes are rate-limited to 1.0 m/s² and 2.0 rad/s². The script time cap is 60 s. Phase 1's `config/safety_limits.yaml` (0.2 m/s, 0.5 rad/s) is unchanged and still used by the Phase 1 scripts. Measured effect of these limits in MuJoCo: see `docs/PHASE2_SIM.md`, "Robot runner in MuJoCo".

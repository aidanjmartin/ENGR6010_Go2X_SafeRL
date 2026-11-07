# Running the navigation policy on the Go2 X over Wi-Fi

This is the procedure for the first real-robot runs of the Task 4 navigation policy (job 050 attempt 4). The laptop runs the policy and talks to the robot over Wi-Fi (WebRTC). Everything is recorded, and each run can be replayed as a digital twin in MuJoCo.

**Status: run on the robot on 2026-11-06 with the overhead camera, over the robot's hotspot: 14 successes in 19 runs** (`results/robot/nav_camera_runs_webrtc.md`, `LOG.md` step 13). The setup below is the one used there.

## One command (after the setup below)

Join the robot's Wi-Fi hotspot (the Unitree Go app shows its name and password; this Mac then has no internet), then:

```bash
.venv/bin/python scripts/robot/check_link.py --method ap --seconds 10     # read-only: LINK OK, about 20 sport-state messages per second
.venv/bin/python scripts/robot/start_run.py --target "red chair"          # or "blue chair", "green chair", "yellow chair", "table"
```

`start_run.py`:
1. locks the webcam's exposure (`config/camera_real.yaml` `controls`);
2. finds the camera that sees the four corner tags (OpenCV's camera numbers change between plug-ins);
3. stops if the camera or a corner tag moved since the floor calibration (`--recalibrate-floor` solves it again);
4. checks that the OWLv2 weights are cached (it runs offline);
5. checks that the robot answers;
6. opens the preview: click its window, check the robot's outline, heading arrow and target, press q there;
7. runs the policy after a typed `yes`;
8. renders the digital-twin video.

`--check-only` runs steps 1 to 5. A chair's color is its tag's (`config/tags.yaml` `obstacles`), whatever the chair looks like.

## What runs, and what does not

- **The navigation policy runs; the walking policy does not.** The navigation policy sends velocity targets (vx, vy, wz) at 10 Hz. On the robot, Unitree's own walking controller follows them (sport mode `Move`), as in the Phase 1 robot session. The Task 1 walking policy needs low-level joint commands, which go over the wired DDS (Data Distribution Service) network only. It is not part of a wireless run.
- **Two ways to know where things are.**
  - **With the overhead camera (`--camera usb`, the main mode):** a USB webcam on the second floor watches the square. A printed ArUco tag on the robot's back gives its position and heading every frame; odometry fills gaps of up to 2 s, and the robot stops if the tag stays hidden longer. OWLv2 finds the chairs, table and people about once a second; optional tags on obstacles give exact positions. The target is a phrase (`--target "red chair"`). See "Overhead camera" below.
  - **Without a camera (`--camera none`):** the obstacles come from a scenario file and the robot's pose from odometry alone. The floor must match the scenario's setup sheet, and walking people are refused.
- **Anything the camera does not detect is invisible to the policy and to the virtual bumper.** Keep the square clear of everything except the obstacles in the preview.

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

## Overhead camera

### What to prepare

- **Webcam:** 1080p, on the second floor, looking down at the square; the whole square and its corner tags must be in view. A USB 2.0 cable longer than 5 m needs an active (powered) extension cable. Used on 2026-11-06: a Logitech Brio 100 (fixed focus) 6.8 m above the floor. Its exposure is locked through `third_party/uvc-util` (built by `setup_laptop.sh`; OpenCV cannot set it on macOS): on auto exposure the dark floor washed out the tags.
- **Prints:** `docs/print/tags.pdf` (or regenerate it with `python scripts/camera/make_tags.py --out tags.pdf`), printed at 100 % ("actual size"), 11 letter pages:
  - corner tags 0 to 3;
  - robot tag 10;
  - obstacle tags 20 to 24 (optional);
  - the ChArUco lens-calibration board.
- **Measure, then enter in `config/tags.yaml`:**
  - the black square of one printed tag (`tag_size_m`, 0.18 m if printed true to size);
  - one square of the board (`charuco.square_m`);
  - the robot tag's height above the floor with the robot standing (`robot.height_m`). A wrong height shifts the robot's position away from the camera; `tests/test_vision.py` shows the effect.
- **Mounting:**
  - **Corner tags:** flat on the floor, each tag's center exactly on a corner of the 3.048 m square; any rotation is fine.
  - **Robot tag:** on a stiff piece of cardboard, flat on the robot's back, centered left-right, with the printed FRONT arrow toward the head. Overhanging the body is fine. If it has to go elsewhere, measure and set `robot.forward_offset_m` (tag center ahead of the body's middle) and `robot.yaw_offset_deg` (arrow turned; 2026-11-06: 0.171 m and 90 degrees, tag at the front, arrow to the robot's right). Keep the white margin unbroken all round the black border, and no glossy tape over the face.
  - **Obstacle tags (optional):** flat on the seat or table top, centered.
- **OWLv2 weights:** download them once while the laptop still has internet, before joining the robot's hotspot (`preview.py --owl` does it, about 0.6 GB).

### Calibration (once per camera placement)

```bash
.venv/bin/python scripts/camera/list_cameras.py                  # find the webcam's index (0 is usually the built-in camera)
.venv/bin/python scripts/camera/calibrate_intrinsics.py --camera 1 --width 1920 --height 1080
#   hold the board in front of the camera and move and tilt it until 25 views are kept; error under 1 px is good
.venv/bin/python scripts/camera/calibrate_extrinsics.py --camera 1
#   corner tags in place, square clear; prints the camera height and the robot tag's size in pixels,
#   and writes calib_check.png: the drawn square must sit on the tape
```

The lens calibration (first command) can be done anywhere, but the focus and zoom must not change afterwards. The camera pose (second command) must be redone whenever the camera moves.

### Setting up a run from the laptop

```bash
.venv/bin/python scripts/camera/preview.py --camera 1 --owl --target "red chair"
```

The preview shows:
- the square;
- every tag;
- the robot's footprint and heading from its tag;
- OWLv2's boxes;
- the layout the policy would get, with the target in yellow.

Place the obstacles and the robot until the preview matches the floor, then quit (q). Detection settings are in `config/tags.yaml` `owl`:

| setting | what it does |
| :--- | :--- |
| `score_threshold` | lower it if obstacles flicker out of the preview; raise it if things that are not there appear |
| `half_depth_m` | moves each detection from the box's bottom edge to the object's center: place a chair at a measured spot and adjust until the preview puts it there |

In MuJoCo, OWLv2's positions were 0.1 to 0.35 m nearer the camera than the truth.

### The run

```bash
.venv/bin/python scripts/robot/run_nav_policy.py --backend webrtc --method ap --camera usb --camera-index 1 --target "red chair"
```

The robot waits, without moving, until the camera has seen its tag and the target, and until OWLv2 has made 3 passes. Then it stands up and the policy runs. Extra files in the run folder:

| file | content |
| :--- | :--- |
| `camera.mp4` | the camera at the policy rate, with the square, tags, robot pose, obstacles and target drawn |
| `detections.jsonl` | every OWLv2 pass |
| `perception.jsonl` | every layout the policy got |

In `policy.npz` and `events.jsonl`, each step records whether its pose came from the camera or from odometry. `render_twin.py` draws camera runs from those poses.

## Wi-Fi connection

Two ways, both from `unitree_webrtc_connect` (`e0abc57`):

| method | how | `--method` |
| :--- | :--- | :--- |
| same network | put the robot on a Wi-Fi network with the Unitree app; the laptop joins the same network; put the robot's Wi-Fi address in `.env` as `GO2_IP` (or pass `--ip`) | `sta` |
| robot hotspot | the laptop joins the robot's own Wi-Fi hotspot; no address needed | `ap` |

Exam 1 found the robot on firmware 1.1.11 to 1.1.14, which needs no AES key. If the firmware has been updated to 1.1.15 or later, the connection needs the per-device key in `.env` as `GO2_AES_KEY` (see the library's `examples/fetch_aes_key.py`).

## Robot-day procedure

1. **Link check over the Ethernet cable first, read-only** (sends nothing; the robot can lie down). The temporary address is the one from the Phase 1 session (`en9` is the USB-C adapter; check its name with `ifconfig`):
   ```bash
   sudo ifconfig en9 alias 192.168.123.99 netmask 255.255.255.0
   ping -c 3 192.168.123.161
   nc -z 192.168.123.161 9991 && echo "WebRTC port open"
   .venv/bin/python scripts/robot/check_link.py --method sta --ip 192.168.123.161 --seconds 10
   ```
   Then unplug the cable, join the robot's Wi-Fi hotspot, and check the wireless link the same way:
   ```bash
   .venv/bin/python scripts/robot/check_link.py --method ap --seconds 10
   ```
   It must print `LINK OK`, sport state near 20 messages per second, and 12 joint angles. Note the battery charge.
2. **Pick a scenario** that fits the floor:
   ```bash
   .venv/bin/python scripts/robot/list_scenarios.py --max-objects 1
   ```
   Start with one obstacle (for example scenario 2, one chair). Scenarios with walking people are left out.
3. **Set up the floor.** With the camera, use `preview.py` (above). Without it, print the sheet without connecting:
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

## Safety limits (approved by Aidan on 2026-11-06)

`config/safety_limits_nav.yaml`: forward speed at most 0.8 m/s (the training limit; 0.5 for the first run), sideways 0.3 m/s, turning 1.0 rad/s. Speed changes are rate-limited to 1.0 m/s² and 2.0 rad/s². The script time cap is 60 s. Phase 1's `config/safety_limits.yaml` (0.2 m/s, 0.5 rad/s) is unchanged and still used by the Phase 1 scripts. Measured effect of these limits in MuJoCo: see `docs/PHASE2_SIM.md`, "Robot runner in MuJoCo".

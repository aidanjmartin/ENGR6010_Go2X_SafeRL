# Navigation policy on the Go2 X with the overhead camera (2026-11-06 evening, run folders in UTC)

Task 4 navigation policy (job 050 attempt 4, `results/phase2/nav_policy/nav_actor.pt`) on the Go2 X over the robot's
Wi-Fi hotspot (WebRTC, LocalAP), robot pose and obstacles from the overhead Brio 100 webcam (`go2x/vision`), started
with `scripts/robot/start_run.py --target "<color> chair" | table`. Floor: the 3.048 m square marked by corner
tags 0 to 3, four identical chairs carrying tags 20 to 23 (red, blue, green, yellow by tag, not by look), a trash
can carrying tag 24 (a table to the policy: 1.2 by 0.75 m, larger than the can), one person standing in the square
in runs 014135 to 014646. The first run used `max_vx` 0.5 m/s, all later runs 0.8 m/s (`config/safety_limits_nav.yaml`).

Table from `scripts/robot/summarize_nav_runs.py` (raw logs in `data/`, not in git). "Target distance" is from the
robot's front to the target's footprint; success needs it under 0.5 m at under 0.1 m/s.

| run (UTC) | target | outcome | steps | target distance start → end (m) | path (m) | max vx asked (m/s) | pose camera / odometry |
| :--- | :--- | :--- | ---: | :--- | ---: | ---: | :--- |
| 20261107T012917Z | red chair | success: front 0.38 m from the target, speed 0.10 m/s | 13 | 0.87 → 0.41 | 0.21 | 0.00 | 13 / 0 |
| 20261107T013603Z | blue chair | success: front 0.48 m from the target, speed 0.09 m/s | 22 | 1.43 → 0.50 | 0.45 | 0.80 | 22 / 0 |
| 20261107T013658Z | yellow chair | success: front 0.23 m from the target, speed 0.09 m/s | 21 | 1.39 → 0.24 | 1.22 | 0.80 | 21 / 0 |
| 20261107T013815Z | yellow chair | stopped: virtual bumper: the footprint touches an obstacle's footprint | 13 | 1.95 → 1.63 | 0.20 | 0.00 | 13 / 0 |
| 20261107T013905Z | yellow chair | success: front 0.27 m from the target, speed 0.08 m/s | 17 | 1.09 → 0.27 | 0.88 | 0.80 | 17 / 0 |
| 20261107T014031Z | red chair | stopped: virtual bumper: the footprint touches an obstacle's footprint | 31 | 1.96 → 0.04 | 1.42 | 0.80 | 31 / 0 |
| 20261107T014135Z | yellow chair | success: front 0.12 m from the target, speed 0.07 m/s | 37 | 1.93 → 0.12 | 1.63 | 0.80 | 37 / 0 |
| 20261107T014228Z | table | stopped: the camera has not seen the robot tag for 2.0 s | 40 | 1.58 → 0.62 | 0.67 | 0.09 | 27 / 13 |
| 20261107T014349Z | table | success: front 0.45 m from the target, speed 0.04 m/s | 17 | 1.09 → 0.48 | 0.28 | 0.00 | 17 / 0 |
| 20261107T014500Z | table | success: front 0.42 m from the target, speed 0.01 m/s | 0 | | | | |
| 20261107T014646Z | blue chair | stopped: virtual bumper: the footprint touches an obstacle's footprint | 33 | 2.06 → 1.30 | 0.77 | 0.80 | 33 / 0 |
| 20261107T014919Z | blue chair | success: front 0.14 m from the target, speed 0.01 m/s | 0 | | | | |
| 20261107T014956Z | red chair | never connected (only: closed) | 0 | | | | |
| 20261107T015033Z | red chair | success: front 0.49 m from the target, speed 0.06 m/s | 9 | 0.52 → 0.50 | 0.11 | 0.00 | 9 / 0 |
| 20261107T015237Z | blue chair | success: front 0.24 m from the target, speed 0.05 m/s | 23 | 0.98 → 0.25 | 0.89 | 0.80 | 22 / 1 |
| 20261107T015345Z | yellow chair | success: front 0.17 m from the target, speed 0.09 m/s | 50 | 2.40 → 0.19 | 2.06 | 0.80 | 50 / 0 |
| 20261107T015441Z | red chair | success: front 0.34 m from the target, speed 0.06 m/s | 31 | 2.31 → 0.34 | 1.58 | 0.80 | 31 / 0 |
| 20261107T015532Z | orange chair | success: front 0.28 m from the target, speed 0.01 m/s | 0 | | | | |
| 20261107T015621Z | green chair | stopped: virtual bumper: the footprint touches an obstacle's footprint | 39 | 2.09 → 0.08 | 1.64 | 0.80 | 39 / 0 |
| 20261107T015741Z | blue chair | success: front 0.27 m from the target, speed 0.01 m/s | 22 | 2.06 → 0.33 | 1.58 | 0.80 | 22 / 0 |
| 20261107T015848Z | green chair | success: front 0.45 m from the target, speed 0.00 m/s | 35 | 2.16 → 0.55 | 1.19 | 0.80 | 35 / 0 |
| 20261107T020000Z | red chair | stopped: the camera has not seen the robot tag for ever | 0 | | | | |
| 20261107T020049Z | red chair | success: front 0.39 m from the target, speed 0.08 m/s | 49 | 2.12 → 0.40 | 1.61 | 0.80 | 49 / 0 |
| 20261107T020150Z | blue chair | success: front 0.33 m from the target, speed 0.00 m/s | 19 | 1.42 → 0.43 | 0.59 | 0.80 | 19 / 0 |
| 20261107T020318Z | blue chair | stopped: the camera has not seen the robot tag for ever | 0 | | | | |

- 25 run folders; 19 with policy steps: 14 success, 4 virtual bumper, 1 camera lost; success 14/19.
- Without policy steps: 3 started at target, 1 no connection, 2 no policy steps.
- Start under 1.9 m from the target: 8/9 success; 1.9 m or more: 6/10.
- Pose from the camera on 507 of 521 policy steps.

Notes:
- 015532 asked for an "orange chair", not a configured color; it then meant any chair and the robot already stood
  within 0.5 m of one. Unknown colors are now refused.
- 020000 and 020318 stopped at the first policy step: no robot-tag fix in the 0.5 s before it, right after the
  stand-up. The runner now waits up to 3 s for a fix after the stand-up.
- Virtual-bumper stops: 014031 and 015621 entered the success distance at 0.39 and 0.20 m/s and touched the target
  chair's footprint on the next step; 013815 most likely touched the trash can's 1.2 by 0.75 m table footprint;
  014646 touched a table detected by OWLv2 at score 0.22 while tag 24 was covered.

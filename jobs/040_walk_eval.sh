#!/usr/bin/env bash
# Job 040: evaluate the walking policy from job 030 (100 seeded 20 s episodes), log transitions, and make the 20 s
# video. Isaac runs without its renderer (RTX crashes on the Dell's driver, job 041); the video poses Unitree's
# Go2 model in MuJoCo at environment 0's recorded states.
set -euo pipefail
source "$(dirname "$0")/common.sh"
activate_venv
POLICY="$DATA_ROOT/030-walk-full/walk/exported/policy.pt"
stamp "evaluate $POLICY"
python scripts/isaac/eval_walk.py --policy "$POLICY" --out "$RESULTS_DIR" \
    --transitions "$DATA_DIR/transitions_walk_eval.npz" --trajectory "$DATA_DIR/walk_env0_trajectory.npz"
setup_mujoco
python scripts/render_walk_video.py --model_dir "$MJ_DIR" --trajectory "$DATA_DIR/walk_env0_trajectory.npz" \
    --out "$RESULTS_DIR/walk_20s.mp4"
ls -la "$DATA_DIR" "$RESULTS_DIR"
stamp "done"

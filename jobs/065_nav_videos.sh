#!/usr/bin/env bash
# Job 065: rollout videos, one success and one failure per controller. Isaac replays the scenarios without a
# camera and saves poses; MuJoCo draws the fixed camera's view (Isaac RTX crashes on the Dell's driver, job 041).
set -euo pipefail
source "$(dirname "$0")/common.sh"
activate_venv
WALK_POLICY="$DATA_ROOT/030-walk-full/walk/exported/policy.pt"
rm -rf "$DATA_DIR/rollouts"
# two runs: the policy and the baselines see different image views (config/nav.yaml policy.image_view)
python scripts/isaac/record_nav_videos.py --walk_policy "$WALK_POLICY" --controllers policy \
    --checkpoint "$DATA_ROOT/050-nav-train/nav/model_final.pt" \
    --evals "$DATA_ROOT/060-nav-eval/nav_eval_policy.json" --out "$DATA_DIR/rollouts"
python scripts/isaac/record_nav_videos.py --walk_policy "$WALK_POLICY" --controllers straight astar \
    --evals "$DATA_ROOT/048-nav-baselines/nav_eval_straight.json" "$DATA_ROOT/048-nav-baselines/nav_eval_astar.json" \
    --out "$DATA_DIR/rollouts"
cp "$DATA_DIR"/rollouts/nav_videos_*.json "$RESULTS_DIR/"
setup_mujoco
python scripts/render_nav_videos.py --model_dir "$MJ_DIR" --rollouts "$DATA_DIR/rollouts" --out "$RESULTS_DIR"
ls -la "$RESULTS_DIR"
stamp "done"

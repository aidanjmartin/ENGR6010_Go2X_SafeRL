#!/usr/bin/env bash
# Job 065: rendered-camera rollout videos, one success and one failure per controller.
set -euo pipefail
source "$(dirname "$0")/common.sh"
activate_venv
WALK_POLICY="$DATA_ROOT/030-walk-full/walk/exported/policy.pt"
python scripts/isaac/record_nav_videos.py --walk_policy "$WALK_POLICY" \
    --checkpoint "$DATA_ROOT/050-nav-train/nav/model_final.pt" \
    --evals "$DATA_ROOT/060-nav-eval/nav_eval_policy.json" "$DATA_ROOT/048-nav-baselines/nav_eval_straight.json" \
            "$DATA_ROOT/048-nav-baselines/nav_eval_astar.json" \
    --out "$RESULTS_DIR"
ls -la "$RESULTS_DIR"
stamp "done"

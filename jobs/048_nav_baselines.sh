#!/usr/bin/env bash
# Job 048: evaluate both baselines (straight to target; A* with pure pursuit) on the 200 fixed scenarios.
set -euo pipefail
source "$(dirname "$0")/common.sh"
activate_venv
WALK_POLICY="$DATA_ROOT/030-walk-full/walk/exported/policy.pt"
for c in straight astar; do
    stamp "baseline $c"
    python scripts/isaac/eval_nav.py --controller "$c" --walk_policy "$WALK_POLICY" --out "$RESULTS_DIR"
done
cp "$RESULTS_DIR"/nav_eval_*.json "$DATA_DIR/"
stamp "done"

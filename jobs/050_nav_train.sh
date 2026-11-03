#!/usr/bin/env bash
# Job 050: train the navigation policy over the frozen walking policy from job 030.
set -euo pipefail
source "$(dirname "$0")/common.sh"
activate_venv
WALK_POLICY="$DATA_ROOT/030-walk-full/walk/exported/policy.pt"
# keep an earlier attempt's checkpoints instead of training into the same folder
if [ -d "$DATA_DIR/nav" ]; then mv "$DATA_DIR/nav" "$DATA_DIR/nav_before_$(date +%Y%m%dT%H%M%S)"; fi
python scripts/isaac/train_nav.py --walk_policy "$WALK_POLICY" --out "$DATA_DIR/nav"
cp "$DATA_DIR/nav/timing.json" "$DATA_DIR/nav/agent.json" "$RESULTS_DIR/"
python scripts/isaac/plot_training.py --run "$DATA_DIR/nav" --out "$RESULTS_DIR/nav_training_curve.png" \
    --json "$RESULTS_DIR/nav_training_scalars.json" --title "Navigation policy training"
stamp "done"

#!/usr/bin/env bash
# Job 030: full walking-policy training from a random initialization (no checkpoint is loaded).
set -euo pipefail
source "$(dirname "$0")/common.sh"
activate_venv
stamp "full training"
python scripts/isaac/train_walk.py --out "$DATA_DIR/walk"
cp "$DATA_DIR/walk/robot_info.json" "$DATA_DIR/walk/timing.json" "$DATA_DIR/walk/params/deploy.yaml" "$RESULTS_DIR/"
cp "$DATA_DIR/walk/exported/policy.pt" "$DATA_DIR/walk/exported/policy.onnx" "$RESULTS_DIR/"
python scripts/isaac/plot_training.py --run "$DATA_DIR/walk" --out "$RESULTS_DIR/training_curve.png" \
    --json "$RESULTS_DIR/training_scalars.json"
ls -la "$DATA_DIR/walk" | tail -5
stamp "done"

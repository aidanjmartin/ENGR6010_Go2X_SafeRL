#!/usr/bin/env bash
# Job 020: about 15 minutes of walking-policy training, to confirm training runs and logs rewards.
set -euo pipefail
source "$(dirname "$0")/common.sh"
activate_venv
MIN=$(python -c "from go2x.config import load_config; print(load_config('walk')['training']['short_job_minutes'])")
stamp "short training, $MIN min"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
python scripts/isaac/train_walk.py --out "$DATA_DIR/walk" --minutes "$MIN" --chunk 25
cp "$DATA_DIR/walk/robot_info.json" "$DATA_DIR/walk/timing.json" "$RESULTS_DIR/"
cp "$DATA_DIR/walk/params/deploy.yaml" "$RESULTS_DIR/" 2>/dev/null || true
python scripts/isaac/plot_training.py --run "$DATA_DIR/walk" --out "$RESULTS_DIR/training_curve_short.png" \
    --json "$RESULTS_DIR/training_scalars_short.json"
stamp "done"

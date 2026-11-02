#!/usr/bin/env bash
# Job 025: navigation pipeline smoke test on job 020's short walking policy (a few minutes of PPO).
set -euo pipefail
source "$(dirname "$0")/common.sh"
activate_venv
WALK_POLICY="$DATA_ROOT/020-walk-short/walk/exported/policy.pt"
ls -la "$WALK_POLICY"
python scripts/isaac/train_nav.py --walk_policy "$WALK_POLICY" --out "$DATA_DIR/nav" --minutes 6 --chunk 10 --num_envs 256
cp "$DATA_DIR/nav/timing.json" "$RESULTS_DIR/"
python scripts/isaac/plot_training.py --run "$DATA_DIR/nav" --out "$RESULTS_DIR/nav_curve_smoke.png" \
    --json "$RESULTS_DIR/nav_scalars_smoke.json" --title "Navigation smoke test"
stamp "done"

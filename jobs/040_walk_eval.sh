#!/usr/bin/env bash
# Job 040: evaluate the walking policy from job 030 (100 seeded 20 s episodes), log transitions, record 20 s video.
set -euo pipefail
source "$(dirname "$0")/common.sh"
activate_venv
POLICY="$DATA_ROOT/030-walk-full/walk/exported/policy.pt"
stamp "evaluate $POLICY"
python scripts/isaac/eval_walk.py --policy "$POLICY" --out "$RESULTS_DIR" \
    --transitions "$DATA_DIR/transitions_walk_eval.npz" --video "$RESULTS_DIR/walk_20s.mp4"
ls -la "$DATA_DIR" "$RESULTS_DIR"
stamp "done"

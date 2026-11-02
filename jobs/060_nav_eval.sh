#!/usr/bin/env bash
# Job 060: evaluate the trained navigation policy on the 200 fixed scenarios.
set -euo pipefail
source "$(dirname "$0")/common.sh"
activate_venv
WALK_POLICY="$DATA_ROOT/030-walk-full/walk/exported/policy.pt"
python scripts/isaac/eval_nav.py --controller policy --checkpoint "$DATA_ROOT/050-nav-train/nav/model_final.pt" \
    --walk_policy "$WALK_POLICY" --out "$RESULTS_DIR"
stamp "done"

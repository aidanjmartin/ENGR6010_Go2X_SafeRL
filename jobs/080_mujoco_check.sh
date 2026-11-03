#!/usr/bin/env bash
# Job 080: Task 6, the walking policy in MuJoCo (unitree_mujoco Go2) on job 040's velocity targets. CPU only.
set -euo pipefail
source "$(dirname "$0")/common.sh"
activate_venv
setup_mujoco
E="$DATA_ROOT/040-walk-eval"
python scripts/mujoco_check.py --model_dir "$MJ_DIR" --policy "$DATA_ROOT/030-walk-full/walk/exported/policy.pt" \
    --commands "$E/walk_eval_commands.npz" --isaac_eval "$(ls -d "$E"/output-attempt* | tail -1)/walk_eval.json" --out "$RESULTS_DIR"
stamp "done"

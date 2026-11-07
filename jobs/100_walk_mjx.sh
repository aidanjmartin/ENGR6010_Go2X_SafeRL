#!/usr/bin/env bash
# Jobs 100 to 115: train the walking policy in MuJoCo MJX with Brax PPO (scripts/mjx/train_walk.py), GPU.
#   bash jobs/100_walk_mjx.sh <run name> [train_walk.py options]
# Each run writes to $DATA_DIR/<run name>; the small files (policy.pt, params.pkl, progress, timing) go to results.
set -euo pipefail
source "$(dirname "$0")/common.sh"
activate_venv
setup_mujoco
setup_mjx
name="$1"; shift
stamp "training $name: $*"
"$MJX_VENV/bin/python" scripts/mjx/train_walk.py --model_dir "$MJ_DIR" --out "$DATA_DIR/$name" "$@" 2>&1 | grep -v "import warp\|mujoco_warp"
mkdir -p "$RESULTS_DIR/$name"
cp "$DATA_DIR/$name"/{policy.pt,params.pkl,progress.json,timing.json,config.json} "$RESULTS_DIR/$name/"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv > "$RESULTS_DIR/gpu.txt"
stamp "done"

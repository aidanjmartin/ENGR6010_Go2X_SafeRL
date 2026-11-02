#!/usr/bin/env bash
# Job 080: Task 6, the walking policy in MuJoCo (unitree_mujoco Go2) on job 040's velocity targets. CPU only.
set -euo pipefail
source "$(dirname "$0")/common.sh"
activate_venv
python -m pip install -q "mujoco==$(python -c "from go2x.config import load_config; print(load_config('mujoco')['mujoco_version'])")"
MJ_DIR="$SRC_DIR/unitree_mujoco"
MJ_COMMIT=$(python -c "from go2x.config import load_config; print(load_config('mujoco')['model']['commit'])")
if [ ! -d "$MJ_DIR/.git" ]; then
    git clone -q --filter=blob:none --no-checkout https://github.com/unitreerobotics/unitree_mujoco.git "$MJ_DIR"
    git -C "$MJ_DIR" sparse-checkout set unitree_robots/go2
fi
git -C "$MJ_DIR" checkout -q --detach "$MJ_COMMIT"
echo "unitree_mujoco at $(git -C "$MJ_DIR" rev-parse HEAD)"
E="$DATA_ROOT/040-walk-eval"
python scripts/mujoco_check.py --model_dir "$MJ_DIR" --policy "$DATA_ROOT/030-walk-full/walk/exported/policy.pt" \
    --commands "$E/walk_eval_commands.npz" --isaac_eval "$(ls -d "$E"/output-attempt* | tail -1)/walk_eval.json" --out "$RESULTS_DIR"
stamp "done"

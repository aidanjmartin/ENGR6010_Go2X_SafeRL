#!/usr/bin/env bash
# Jobs 130 to 135: navigation policy training in CPU MuJoCo (scripts/mjnav/train_nav.py) over a frozen walker.
#   bash jobs/130_nav_mujoco.sh <run name> <walker policy.pt, relative to $DATA_ROOT> [train_nav.py options; --init relative to $DATA_ROOT]
# Physics on the CPU (MuJoCo rollout, 20 threads); drawing, PPO and the networks on the GPU.
set -euo pipefail
source "$(dirname "$0")/common.sh"
activate_venv
setup_mujoco
name="$1"; walker="$DATA_ROOT/$2"; shift 2
# --init takes a checkpoint path relative to $DATA_ROOT
args=()
while [ $# -gt 0 ]; do
    if [ "$1" = "--init" ]; then args+=(--init "$DATA_ROOT/$2"); shift 2; else args+=("$1"); shift; fi
done
set -- "${args[@]}"
stamp "training $name over $walker: $*"
python scripts/mjnav/train_nav.py --model_dir "$MJ_DIR" --walk_policy "$walker" --out "$DATA_DIR/$name" "$@" \
    | grep -v "^\s*$" | grep -E "train_nav_mujoco|Learning iteration|Mean reward|Mean episode length|/episode/|wrote" || true
test -f "$DATA_DIR/$name/nav_actor.pt"
mkdir -p "$RESULTS_DIR/$name"
cp "$DATA_DIR/$name"/{nav_actor.pt,agent.json,run.json,scalars.json,timing.json} "$RESULTS_DIR/$name/"
stamp "done"

#!/usr/bin/env bash
# Job 120: every walking policy (Isaac job 030 and the MJX runs, jobs 110 to 115) in CPU MuJoCo on the full
# go2.xml under the test conditions of scripts/mjx/eval_walk.py, 100 episodes each. CPU only.
set -euo pipefail
source "$(dirname "$0")/common.sh"
activate_venv
setup_mujoco
setup_mjx
args=(--policy "isaac=$DATA_ROOT/030-walk-full/walk/exported/policy.pt")
for run in 110-walk-mjx-s0:mjx_s0 111-walk-mjx-s1:mjx_s1 112-walk-mjx-s2:mjx_s2 113-walk-mjx-nodr:mjx_nodr \
           114-walk-mjx-sym:mjx_sym 115-walk-mjx-push1:mjx_push1; do
    job="${run%%:*}"; name="${run##*:}"
    p="$DATA_ROOT/$job/$name/policy.pt"
    if [ -f "$p" ]; then args+=(--policy "$name=$p"); else echo "missing $p"; fi
done
"$MJX_VENV/bin/python" scripts/mjx/eval_walk.py --model_dir "$MJ_DIR" \
    --commands "$DATA_ROOT/040-walk-eval/walk_eval_commands.npz" "${args[@]}" --out "$RESULTS_DIR" --episodes 100 --workers 22 \
    2>&1 | grep -v "import warp\|mujoco_warp"
stamp "done"

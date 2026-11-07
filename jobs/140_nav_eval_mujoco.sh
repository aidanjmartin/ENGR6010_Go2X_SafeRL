#!/usr/bin/env bash
# Job 140: navigation controllers in CPU MuJoCo on the 200 fixed scenarios (scripts/mjnav/eval_nav.py).
# Over the MJX walker (job 110): the MuJoCo-trained actors (jobs 130 to 132), the Isaac attempt 4 actor, and the
# straight and A* baselines. Over the Isaac walker (job 030): the attempt 4 actor and the baselines.
set -euo pipefail
source "$(dirname "$0")/common.sh"
activate_venv
setup_mujoco
ctl=()
for run in 130-nav-mujoco-scratch:nav_mj_scratch 131-nav-mujoco-finetune:nav_mj_finetune 132-nav-mujoco-ent003:nav_mj_ent003; do
    job="${run%%:*}"; name="${run##*:}"
    a="$DATA_ROOT/$job/$name/nav_actor.pt"
    if [ -f "$a" ]; then ctl+=(--controller "$name=actor:$a"); else echo "missing $a"; fi
done
A4=results/phase2/nav_policy/nav_actor.pt
python scripts/mjnav/eval_nav.py --model_dir "$MJ_DIR" --walk_policy "$DATA_ROOT/110-walk-mjx-s0/mjx_s0/policy.pt" \
    "${ctl[@]}" --controller "isaac_attempt4=actor:$A4" --controller straight=straight --controller astar=astar \
    --out "$RESULTS_DIR/mjx_walker"
python scripts/mjnav/eval_nav.py --model_dir "$MJ_DIR" --walk_policy "$DATA_ROOT/030-walk-full/walk/exported/policy.pt" \
    --controller "isaac_attempt4=actor:$A4" --controller straight=straight --controller astar=astar \
    --out "$RESULTS_DIR/isaac_walker"
stamp "done"

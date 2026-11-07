#!/usr/bin/env bash
# Job 150: videos, all physics in CPU MuJoCo on the full go2.xml (needs the X display for rendering).
# - walk_progress.mp4: the seed 1 run's training snapshots (job 111), one clip each: the dog learning to walk
# - walk_demo_<walker>.mp4: a 33 s course (speeds, turns, sideways, backward, side kicks) for the MJX and Isaac walkers
# - nav_<scenario>.mp4: real-time navigation runs (scripts/robot/run_nav_policy.py, MuJoCo backend) with the MJX
#   walker and the MuJoCo-trained actor with the best success in job 140, plus the digital twin of the first one
set -euo pipefail
source "$(dirname "$0")/common.sh"
activate_venv
setup_mujoco
setup_mjx
PY="$MJX_VENV/bin/python"
snaps=$(ls "$DATA_ROOT/111-walk-mjx-s1/mjx_s1/snapshots"/policy_*.pt)
labels=()
for s in $snaps; do n=$(basename "$s" .pt); n=$((10#${n#policy_})); labels+=(--label "seed 1, after $(printf "%'d" "$n") environment steps"); done
# shellcheck disable=SC2086
"$PY" scripts/mjx/record_walk.py --model_dir "$MJ_DIR" $(for s in $snaps; do echo --policy "$s"; done) "${labels[@]}" \
    --progress --out "$RESULTS_DIR/walk_progress.mp4" 2>&1 | grep -v "import warp\|mujoco_warp"
"$PY" scripts/mjx/record_walk.py --model_dir "$MJ_DIR" --policy "$DATA_ROOT/110-walk-mjx-s0/mjx_s0/policy.pt" \
    --label "MJX walker (seed 0, 200M steps), full go2.xml" --demo --out "$RESULTS_DIR/walk_demo_mjx_s0.mp4" 2>&1 | grep -v "import warp\|mujoco_warp"
"$PY" scripts/mjx/record_walk.py --model_dir "$MJ_DIR" --policy "$DATA_ROOT/030-walk-full/walk/exported/policy.pt" \
    --label "Isaac Lab walker (job 030), full go2.xml" --demo --out "$RESULTS_DIR/walk_demo_isaac.mp4" 2>&1 | grep -v "import warp\|mujoco_warp"
E=$(ls -d "$DATA_ROOT/140-nav-eval-mujoco"/output-attempt* | tail -1)/mjx_walker
best=$(python - "$E" <<'PY'
import glob, json, sys
rows = [json.load(open(f))["summary"] for f in glob.glob(sys.argv[1] + "/nav_eval_nav_mj_*.json")]
b = max(rows, key=lambda s: s["success_rate"]["mean"])
print(b["source"])
PY
)
echo "navigation actor for the videos: $best"
for sc in 2 23 101 7 40 77 150 190; do
    python scripts/robot/run_nav_policy.py --backend mujoco --scenario "$sc" --record --model_dir "$MJ_DIR" \
        --walk-policy "$DATA_ROOT/110-walk-mjx-s0/mjx_s0/policy.pt" --actor "$best" \
        --limits safety_limits_nav_sim_training --data-dir "$DATA_DIR/nav_runs/$sc" | tail -3
    run=$(ls -d "$DATA_DIR/nav_runs/$sc"/*/ | tail -1)
    cp "$run/sim_video.mp4" "$RESULTS_DIR/nav_scenario$(printf %03d "$sc").mp4"
    cp "$run/run.json" "$RESULTS_DIR/nav_scenario$(printf %03d "$sc").json"
done
run=$(ls -d "$DATA_DIR/nav_runs/23"/*/ | tail -1)
python scripts/robot/render_twin.py "$run" --model_dir "$MJ_DIR" && cp "$run/twin.mp4" "$RESULTS_DIR/twin_scenario023.mp4" || echo "twin failed"
echo "$best" > "$RESULTS_DIR/nav_actor_used.txt"
stamp "done"

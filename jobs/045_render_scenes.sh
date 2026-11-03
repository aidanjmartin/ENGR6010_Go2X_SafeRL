#!/usr/bin/env bash
# Job 045: render the fixed camera for 60 random atrium scenes with ground truth (Task 3 rendered check).
# MuJoCo draws the images (Isaac Sim's RTX renderer crashes on the Dell's driver, job 041).
set -euo pipefail
source "$(dirname "$0")/common.sh"
activate_venv
setup_mujoco
python scripts/render_scenes_mujoco.py --model_dir "$MJ_DIR" --out "$DATA_DIR/scenes"
mkdir -p "$RESULTS_DIR/scenes"
cp "$DATA_DIR/scenes/index.json" "$RESULTS_DIR/scenes/"
for i in 000 001 002 003; do cp "$DATA_DIR/scenes/scene_$i".{png,json} "$DATA_DIR/scenes/scene_${i}_topdown.png" "$RESULTS_DIR/scenes/"; done
python scripts/atrium/contact_sheet.py --scenes "$DATA_DIR/scenes" --out "$RESULTS_DIR/scenes_contact_sheet.jpg"
stamp "done"

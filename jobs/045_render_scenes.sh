#!/usr/bin/env bash
# Job 045: render the fixed camera for 60 random atrium scenes with ground truth (Task 3 rendered check).
set -euo pipefail
source "$(dirname "$0")/common.sh"
activate_venv
WALK_POLICY="$DATA_ROOT/030-walk-full/walk/exported/policy.pt"
python scripts/isaac/render_scenes.py --walk_policy "$WALK_POLICY" --out "$DATA_DIR/scenes"
mkdir -p "$RESULTS_DIR/scenes"
cp "$DATA_DIR/scenes/index.json" "$RESULTS_DIR/scenes/"
for i in 000 001 002 003; do cp "$DATA_DIR/scenes/scene_$i".{png,json} "$DATA_DIR/scenes/scene_${i}_topdown.png" "$RESULTS_DIR/scenes/"; done
python scripts/atrium/contact_sheet.py --scenes "$DATA_DIR/scenes" --out "$RESULTS_DIR/scenes_contact_sheet.jpg"
stamp "done"

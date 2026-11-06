#!/usr/bin/env bash
# Fetch Unitree's MuJoCo Go2 model at the pinned commit (config/mujoco.yaml) into third_party/ (gitignored).
# Used by the mujoco backend and by render_twin.py. Run from the repository root.
set -euo pipefail
COMMIT=1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d
DEST=third_party/unitree_mujoco
if [ ! -d "$DEST/.git" ]; then
    git clone --filter=blob:none --no-checkout https://github.com/unitreerobotics/unitree_mujoco.git "$DEST"
    git -C "$DEST" sparse-checkout set unitree_robots/go2
fi
git -C "$DEST" fetch -q origin "$COMMIT"
git -C "$DEST" checkout -q "$COMMIT"
ls "$DEST/unitree_robots/go2/go2.xml"
echo "unitree_mujoco at $(git -C "$DEST" rev-parse --short HEAD) in $DEST"

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

# uvc-util (github.com/jtfrey/uvc-util, pinned commit): sets the webcam's exposure, which OpenCV cannot on macOS
# (go2x/vision/uvc.py). macOS only; needs the Xcode command line tools (clang).
if [ "$(uname)" = "Darwin" ]; then
    UVC_COMMIT=8110da7025c95eea3096a7181af9a46c0cc7ac37
    UVC=third_party/uvc-util
    [ -d "$UVC/.git" ] || git clone -q https://github.com/jtfrey/uvc-util.git "$UVC"
    git -C "$UVC" fetch -q origin "$UVC_COMMIT"
    git -C "$UVC" checkout -q "$UVC_COMMIT"
    (cd "$UVC/src" && clang -o ../uvc-util -framework IOKit -framework Foundation uvc-util.m UVCController.m UVCType.m UVCValue.m)
    echo "uvc-util at $(git -C "$UVC" rev-parse --short HEAD) in $UVC"
fi

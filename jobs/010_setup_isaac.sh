#!/usr/bin/env bash
# Job 010: install Isaac Sim 5.1.0, Isaac Lab v2.3.2 and unitree_rl_lab into $JOB_CACHE,
# then report the GPU, Python, CUDA and package versions. Safe to rerun: finished steps are skipped.
set -euo pipefail
source "$(dirname "$0")/common.sh"
mkdir -p "$JOB_CACHE/uv" "$SRC_DIR"

stamp "nvidia-smi"
nvidia-smi | tee "$RESULTS_DIR/nvidia-smi.txt"
ldd --version | sed -n 1p

# 1. uv, used only to download a standalone CPython 3.11 (Ubuntu 24.04 ships 3.12).
if [ ! -x "$UV_BIN" ]; then
    stamp "download uv $UV_VERSION"
    tmp=$(mktemp -d)
    curl -sSL -o "$tmp/uv.tar.gz" \
        "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-x86_64-unknown-linux-gnu.tar.gz"
    echo "$UV_SHA256  $tmp/uv.tar.gz" | sha256sum -c -
    tar xzf "$tmp/uv.tar.gz" -C "$tmp"
    cp "$tmp/uv-x86_64-unknown-linux-gnu/uv" "$UV_BIN"
    rm -rf "$tmp"
fi
"$UV_BIN" --version

# 2. Python and a standard venv (pip inside it, as in the Isaac Lab venv instructions).
if [ ! -x "$VENV/bin/python" ]; then
    stamp "install CPython $PYTHON_VERSION and create the venv"
    "$UV_BIN" python install "$PYTHON_VERSION"
    PY=$("$UV_BIN" python find "$PYTHON_VERSION")
    "$PY" -m venv "$VENV"
fi
activate_venv
python --version
python -m pip install -q --upgrade pip

# 3. Isaac Sim and PyTorch (Isaac Lab v2.3.2 docs/source/setup/installation/pip_installation.rst).
if ! python -c "import importlib.metadata as m; assert m.version('isaacsim') == '$ISAACSIM_VERSION'" 2>/dev/null; then
    stamp "pip install isaacsim $ISAACSIM_VERSION"
    python -m pip install "isaacsim[all,extscache]==$ISAACSIM_VERSION" --extra-index-url https://pypi.nvidia.com
fi
python -m pip install -q -U "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" --index-url "$TORCH_INDEX"

# 4. Pinned source checkouts.
clone_at() {  # url dir commit [sparse paths...]
    local url=$1 dir=$2 commit=$3; shift 3
    if [ ! -d "$dir/.git" ]; then
        git clone -q --filter=blob:none --no-checkout "$url" "$dir"
        if [ $# -gt 0 ]; then git -C "$dir" sparse-checkout set "$@"; fi
    fi
    git -C "$dir" fetch -q origin "$commit" || true
    git -C "$dir" checkout -q --detach "$commit"
    echo "$dir at $(git -C "$dir" rev-parse HEAD)"
}
clone_at "$ISAACLAB_URL" "$ISAACLAB_PATH" "$ISAACLAB_COMMIT"
clone_at "$UNITREE_RL_LAB_URL" "$UNITREE_RL_LAB_PATH" "$UNITREE_RL_LAB_COMMIT"
clone_at "$UNITREE_ROS_URL" "$UNITREE_ROS_DIR" "$UNITREE_ROS_COMMIT" robots/go2_description

# 5. Isaac Lab extensions plus RSL-RL only, then unitree_rl_lab (its README's `pip install -e` step;
#    unitree_rl_lab.sh -i would also edit shell startup files, so it is not used).
# flatdict 4.0.1 (pinned by isaaclab) builds with pkg_resources, which pip's isolated build environment no
# longer has; build it against the venv's own setuptools instead.
python -m pip install -q --no-build-isolation "flatdict==4.0.1"
if ! python -c "import isaaclab, rsl_rl" 2>/dev/null; then
    stamp "isaaclab.sh --install rsl_rl"
    (cd "$ISAACLAB_PATH" && ./isaaclab.sh --install rsl_rl)
fi
python -m pip install -q -e "$UNITREE_RL_LAB_PATH/source/unitree_rl_lab"
python -m pip install -q pyyaml scipy matplotlib imageio imageio-ffmpeg pytest
# put back the Isaac Sim pins if an earlier attempt upgraded them
python -m pip install -q "packaging==23.0" "psutil==5.9.8"
python -m pip check || echo "pip check reported problems (see above)"
python -m pip freeze > "$RESULTS_DIR/pip-freeze.txt"

# 6. Report, then a headless Isaac Sim start to prove the EULA setting and the GPU work.
stamp "report"
python scripts/isaac/report_env.py --out "$RESULTS_DIR/env_report.json"
stamp "done"

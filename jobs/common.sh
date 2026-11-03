# Shared setup for Phase 2 Dell jobs. Source it from a job script after `set -euo pipefail`.
# Everything installed lives in $JOB_CACHE, which persists between jobs of this repository.
JOBS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$JOBS_DIR/pins.sh"

# The runner's login shell sources ROS 2 Jazzy, and jobs inherit its Python 3.12 PYTHONPATH and
# library paths. Isaac Sim runs on Python 3.11 and must not see them.
unset PYTHONPATH LD_LIBRARY_PATH AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH
unset GZ_SIM_RESOURCE_PATH GZ_SIM_SYSTEM_PLUGIN_PATH GZ_CONFIG_PATH
export PATH="$(echo "$PATH" | tr ':' '\n' | grep -v '^/opt/ros/' | paste -sd: -)"

: "${JOB_CACHE:?run this through the Dell runner, or set JOB_CACHE}"
: "${RESULTS_DIR:?}"
: "${DATA_DIR:?}"

export UV_BIN="$JOB_CACHE/uv/uv"
export UV_CACHE_DIR="$JOB_CACHE/uv-cache"
export UV_PYTHON_INSTALL_DIR="$JOB_CACHE/python"
export PIP_CACHE_DIR="$JOB_CACHE/pip-cache"
export PIP_CONSTRAINT="$JOBS_DIR/constraints.txt"
export VENV="$JOB_CACHE/venv-isaac"
export SRC_DIR="$JOB_CACHE/src"
export ISAACLAB_PATH="$SRC_DIR/IsaacLab"
export UNITREE_RL_LAB_PATH="$SRC_DIR/unitree_rl_lab"
export UNITREE_ROS_DIR="$SRC_DIR/unitree_ros"
# Isaac Sim 5.1.0 headless runs accept the EULA through this variable (Isaac Sim pip install docs).
export OMNI_KIT_ACCEPT_EULA=YES
# Data of earlier jobs: $DATA_ROOT/<job-id>.
export DATA_ROOT="$(dirname "$DATA_DIR")"
mkdir -p "$DATA_DIR" "$RESULTS_DIR"

activate_venv() {
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    export PYTHONPATH="$JOB_CHECKOUT"
}

stamp() { echo "[$(date -Is)] $*"; }

# MuJoCo and Unitree's Go2 model (config/mujoco.yaml). Used by job 080 (Task 6) and, because Isaac Sim 5.1's RTX
# renderer crashes on the Dell's 595 driver (LOG.md, job 041), for every rendered image and video (MuJoCo draws
# through the X display the runner passes in DISPLAY).
setup_mujoco() {
    python -m pip install -q "mujoco==$(python -c "from go2x.config import load_config; print(load_config('mujoco')['mujoco_version'])")"
    export MJ_DIR="$SRC_DIR/unitree_mujoco"
    local commit
    commit=$(python -c "from go2x.config import load_config; print(load_config('mujoco')['model']['commit'])")
    if [ ! -d "$MJ_DIR/.git" ]; then
        git clone -q --filter=blob:none --no-checkout https://github.com/unitreerobotics/unitree_mujoco.git "$MJ_DIR"
        git -C "$MJ_DIR" sparse-checkout set unitree_robots/go2
    fi
    git -C "$MJ_DIR" checkout -q --detach "$commit"
    echo "unitree_mujoco at $(git -C "$MJ_DIR" rev-parse HEAD), mujoco $(python -c 'import mujoco; print(mujoco.__version__)')"
    export MUJOCO_GL=glfw
    : "${DISPLAY:?MuJoCo rendering needs the X display}"
}

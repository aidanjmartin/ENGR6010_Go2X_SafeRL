# Shared setup for Phase 2 Dell jobs. Source it from a job script after `set -euo pipefail`.
# Everything installed lives in $JOB_CACHE, which persists between jobs of this repository.
JOBS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$JOBS_DIR/pins.sh"

: "${JOB_CACHE:?run this through the Dell runner, or set JOB_CACHE}"
: "${RESULTS_DIR:?}"
: "${DATA_DIR:?}"

export UV_BIN="$JOB_CACHE/uv/uv"
export UV_CACHE_DIR="$JOB_CACHE/uv-cache"
export UV_PYTHON_INSTALL_DIR="$JOB_CACHE/python"
export PIP_CACHE_DIR="$JOB_CACHE/pip-cache"
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
    export PYTHONPATH="$JOB_CHECKOUT${PYTHONPATH:+:$PYTHONPATH}"
}

stamp() { echo "[$(date -Is)] $*"; }

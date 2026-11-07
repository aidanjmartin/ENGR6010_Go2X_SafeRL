# Pinned versions for every Phase 2 Dell job. Sources are recorded in LOG.md.
UV_VERSION=0.12.24
UV_SHA256=b4dfaef47d491a7296981f8374a4595f55dbf84e8937c8ecd2983574d8bb3da6
PYTHON_VERSION=3.11.17
ISAACSIM_VERSION=5.1.0
TORCH_VERSION=2.7.0
TORCHVISION_VERSION=0.22.0
TORCH_INDEX=https://download.pytorch.org/whl/cu128
ISAACLAB_URL=https://github.com/isaac-sim/IsaacLab.git
ISAACLAB_COMMIT=37ddf626871758333d6ed89cf64ad702aef127d0          # tag v2.3.2
UNITREE_RL_LAB_URL=https://github.com/unitreerobotics/unitree_rl_lab.git
UNITREE_RL_LAB_COMMIT=4960b84732b0c2ec593dccbfe963fda1bcd7b1e3    # main, 2025-11-19
UNITREE_ROS_URL=https://github.com/unitreerobotics/unitree_ros.git
UNITREE_ROS_COMMIT=3ae893d905d11795ddf1075825ee5010364a1109       # master, 2026-10-30
# MuJoCo MJX training (jobs 100 to 130): Python 3.12, CPU torch for the TorchScript export.
# brax 0.14.2 calls jax.device_put_replicated, removed in jax 0.11, so jax stays at 0.9.2 (2026-03-18).
MJX_PYTHON_VERSION=3.12.12
MJX_PINS="jax[cuda12]==0.9.2 jaxlib==0.9.2 mujoco==3.16.0 mujoco-mjx==3.16.0 brax==0.14.2 flax==0.12.6 playground==0.2.0 pyyaml==6.0.3 imageio==2.37.0 imageio-ffmpeg==0.6.0"
MJX_TORCH="torch==2.7.0"

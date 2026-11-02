"""Train the Task 1 walking policy with RSL-RL PPO from a random initialization.

Adapted from unitree_rl_lab 4960b84 `scripts/rsl_rl/train.py`: no Hydra, our env config, a wall-clock
limit, and a checkpoint at least every `max_checkpoint_minutes`. No checkpoint is ever loaded.

    python scripts/isaac/train_walk.py --out $DATA_DIR/walk --minutes 15
    python scripts/isaac/train_walk.py --out $DATA_DIR/walk --iterations 3000
"""
import argparse
import json
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True, help="log and checkpoint folder")
parser.add_argument("--iterations", type=int, default=None, help="default: training.full_iterations")
parser.add_argument("--minutes", type=float, default=None, help="stop after this much training time")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--chunk", type=int, default=100, help="iterations per learn() call; RSL-RL saves at the end of each")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg  # noqa: E402

from go2x.isaac.agents import WalkPPORunnerCfg  # noqa: E402
from go2x.isaac.walk_env_cfg import WALK, WalkEnvCfg  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

tr = WALK["training"]
env_cfg = WalkEnvCfg()
env_cfg.scene.num_envs = args.num_envs or tr["num_envs"]
env_cfg.seed = tr["seed"]
agent_cfg = WalkPPORunnerCfg()
agent_cfg.seed = tr["seed"]
agent_cfg.num_steps_per_env = tr["num_steps_per_env"]
agent_cfg.save_interval = tr["save_interval"]
total = args.iterations or tr["full_iterations"]

os.makedirs(args.out, exist_ok=True)
env = ManagerBasedRLEnv(cfg=env_cfg)
robot = env.scene["robot"]
info = {
    "joint_names_sim_order": robot.joint_names,
    "body_names": robot.body_names,
    "default_joint_pos": robot.data.default_joint_pos[0].tolist(),
    "soft_joint_pos_limits": robot.data.soft_joint_pos_limits[0].tolist(),
    "joint_pos_limits": robot.data.joint_pos_limits[0].tolist(),
    "num_envs": env.num_envs,
    "policy_dt_s": env.step_dt,
    "physics_dt_s": env.physics_dt,
}
print(json.dumps(info, indent=1), flush=True)
with open(os.path.join(args.out, "robot_info.json"), "w") as f:
    json.dump(info, f, indent=1)

wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=args.out, device=agent_cfg.device)
dump_yaml(os.path.join(args.out, "params", "env.yaml"), env_cfg)
dump_yaml(os.path.join(args.out, "params", "agent.yaml"), agent_cfg)
export_deploy_cfg(env, args.out)

t0 = last_save = time.time()
done, first = 0, True
timing = []
while done < total:
    n = min(args.chunk, total - done)
    t = time.time()
    runner.learn(num_learning_iterations=n, init_at_random_ep_len=first)
    first = False
    done += n
    timing.append((done, time.time() - t0))
    print(f"[train_walk] {done}/{total} iterations, {time.time() - t0:.0f} s, {(time.time() - t) / n:.2f} s/iteration",
          flush=True)
    if time.time() - last_save > 60 * tr["max_checkpoint_minutes"]:
        runner.save(os.path.join(args.out, f"model_{runner.current_learning_iteration}.pt"))
        last_save = time.time()
    if args.minutes and time.time() - t0 > 60 * args.minutes:
        print("[train_walk] time limit reached", flush=True)
        break
runner.save(os.path.join(args.out, "model_final.pt"))
with open(os.path.join(args.out, "timing.json"), "w") as f:
    json.dump({"iterations_done": done, "seconds": time.time() - t0, "per_chunk": timing}, f, indent=1)

# Export the actor as TorchScript for the navigation task and the MuJoCo check.
try:
    from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx

    # as in Isaac Lab v2.3.2 scripts/reinforcement_learning/rsl_rl/play.py
    policy_nn = runner.alg.policy
    normalizer = getattr(policy_nn, "actor_obs_normalizer", None)
    export_dir = os.path.join(args.out, "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_dir, filename="policy.onnx")
    print("[train_walk] exported", os.path.join(args.out, "exported", "policy.pt"), flush=True)
except Exception as exc:  # report, keep the checkpoint
    print(f"[train_walk] export failed: {exc}", flush=True)
    from go2x.isaac.shutdown import finish

    finish(3)
from go2x.isaac.shutdown import finish  # noqa: E402

finish(0)

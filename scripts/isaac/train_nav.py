"""Train the Task 4 navigation policy with RSL-RL PPO over the frozen walking policy.

    python scripts/isaac/train_nav.py --walk_policy <policy.pt> --out $DATA_DIR/nav [--minutes 15] [--iterations N]
"""
import argparse
import json
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--walk_policy", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--iterations", type=int, default=None)
parser.add_argument("--minutes", type=float, default=None)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--chunk", type=int, default=None,
                    help="iterations per learn() call (default: all in one call). RSL-RL restarts its episode reward and length statistics at every call, so chunked runs log a sawtooth; use chunks only with --minutes")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
os.environ["GO2X_WALK_POLICY"] = os.path.abspath(args.walk_policy)
app = AppLauncher(args).app

import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from go2x.isaac.nav_env_cfg import NAV, AtriumNavEnvCfg, nav_agent_dict  # noqa: E402
from go2x.isaac.shutdown import finish  # noqa: E402
from go2x.nav_policy import register_with_rsl_rl  # noqa: E402

tr = NAV["training"]
register_with_rsl_rl()
env_cfg = AtriumNavEnvCfg()
env_cfg.scene.num_envs = args.num_envs or tr["num_envs"]
env_cfg.seed = tr["seed"]
agent = RslRlOnPolicyRunnerCfg(
    seed=tr["seed"], device="cuda:0", num_steps_per_env=tr["num_steps_per_env"], max_iterations=tr["iterations"],
    save_interval=tr["save_interval"], experiment_name="go2x_nav", empirical_normalization=False,
    policy=RslRlPpoActorCriticCfg(init_noise_std=0.5, actor_hidden_dims=[8], critic_hidden_dims=[8], activation="elu"),
    algorithm=RslRlPpoAlgorithmCfg(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
                                   entropy_coef=tr["entropy_coef"], num_learning_epochs=tr["num_learning_epochs"],
                                   num_mini_batches=tr["num_mini_batches"], learning_rate=tr["learning_rate"],
                                   schedule="adaptive", gamma=tr["gamma"], lam=tr["lam"], desired_kl=0.01, max_grad_norm=1.0),
)
total = args.iterations or tr["iterations"]
os.makedirs(args.out, exist_ok=True)
env = ManagerBasedRLEnv(cfg=env_cfg)
wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
agent_dict = nav_agent_dict(agent.to_dict())
with open(os.path.join(args.out, "agent.json"), "w") as f:
    json.dump(agent_dict, f, indent=1, default=str)
runner = OnPolicyRunner(wrapped, agent_dict, log_dir=args.out, device=agent.device)

t0 = last_save = time.time()
done, first = 0, True
while done < total:
    n = min(args.chunk or total, total - done)
    t = time.time()
    runner.learn(num_learning_iterations=n, init_at_random_ep_len=first)
    first = False
    done += n
    print(f"[train_nav] {done}/{total} iterations, {time.time() - t0:.0f} s, {(time.time() - t) / n:.2f} s/iteration, "
          f"GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    if time.time() - last_save > 60 * tr["max_checkpoint_minutes"]:
        runner.save(os.path.join(args.out, f"model_{runner.current_learning_iteration}.pt"))
        last_save = time.time()
    if args.minutes and time.time() - t0 > 60 * args.minutes:
        print("[train_nav] time limit reached", flush=True)
        break
runner.save(os.path.join(args.out, "model_final.pt"))
with open(os.path.join(args.out, "timing.json"), "w") as f:
    json.dump({"iterations_done": done, "seconds": time.time() - t0, "num_envs": env.num_envs}, f, indent=1)
finish(0)

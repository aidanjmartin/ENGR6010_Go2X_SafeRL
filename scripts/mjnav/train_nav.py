"""Train the Task 4 navigation policy in CPU MuJoCo (go2x/mjnav/env.py) with RSL-RL 3.1.2 PPO, over a frozen
walking policy. The PPO and network settings are attempt 4's (config/nav.yaml, scripts/isaac/train_nav.py);
options change one thing at a time for the comparison runs.

Writes model_<it>.pt checkpoints, model_final.pt, nav_actor.pt (the robot runner's format,
scripts/export_nav_actor.py), agent.json, scalars.json (every iteration's logged values) and timing.json.

    python scripts/mjnav/train_nav.py --model_dir <unitree_mujoco> --walk_policy policy.pt --out DIR
        [--iterations N] [--num_envs N] [--seed S] [--entropy_coef C] [--init CHECKPOINT]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from go2x.config import load_config  # noqa: E402
from go2x.mjnav.env import MujocoNavEnv  # noqa: E402
from go2x.nav_policy import register_with_rsl_rl  # noqa: E402


def agent_dict(NAV, TD, seed, entropy_coef, iterations):
    tr, P = NAV["training"], NAV["policy"]
    return {
        "seed": seed, "device": "cuda:0", "num_steps_per_env": tr["num_steps_per_env"], "max_iterations": iterations,
        "empirical_normalization": False, "obs_groups": {"policy": ["policy"], "critic": ["policy", "critic"]},
        "clip_actions": None, "save_interval": tr["save_interval"], "experiment_name": "go2x_nav_mujoco",
        "run_name": "", "logger": "tensorboard", "resume": False, "class_name": "OnPolicyRunner",
        "policy": {"class_name": "CNNActorCritic", "init_noise_std": 0.5, "noise_std_type": "scalar",
                   "state_dependent_std": False, "actor_obs_normalization": {}, "critic_obs_normalization": {},
                   "actor_hidden_dims": [8], "critic_hidden_dims": [8], "activation": "elu",
                   "image_shape": [2 * 4, TD["size"], TD["size"]], "cnn_channels": P["cnn_channels"],
                   "cnn_out": P["cnn_out"], "mlp": P["mlp"]},
        "algorithm": {"class_name": "PPO", "num_learning_epochs": tr["num_learning_epochs"],
                      "num_mini_batches": tr["num_mini_batches"], "learning_rate": tr["learning_rate"],
                      "schedule": "adaptive", "gamma": tr["gamma"], "lam": tr["lam"], "entropy_coef": entropy_coef,
                      "desired_kl": 0.01, "max_grad_norm": 1.0, "value_loss_coef": 1.0, "use_clipped_value_loss": True,
                      "clip_param": 0.2, "normalize_advantage_per_mini_batch": False, "rnd_cfg": None,
                      "symmetry_cfg": None},
    }


class ScalarLog:
    """Collects what the runner sends to its writer, for scalars.json."""

    def __init__(self, writer):
        self.w, self.rows = writer, {}

    def __getattr__(self, name):
        return getattr(self.w, name)

    def add_scalar(self, tag, value, step=None, *a, **k):
        self.rows.setdefault(tag, []).append([int(step or 0), float(value)])
        return self.w.add_scalar(tag, value, step, *a, **k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--walk_policy", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--num_envs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--entropy_coef", type=float, default=None)
    ap.add_argument("--init", default=None, help="start from this RSL-RL checkpoint (fine-tuning)")
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=50, help="iterations per learn() call, for checkpoints and progress")
    args = ap.parse_args()
    NAV, TD = load_config("nav"), load_config("topdown")
    tr = NAV["training"]
    seed = tr["seed"] if args.seed is None else args.seed
    ent = tr["entropy_coef"] if args.entropy_coef is None else args.entropy_coef
    total = args.iterations or tr["iterations"]
    B = args.num_envs or tr["num_envs"]
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    register_with_rsl_rl()
    env = MujocoNavEnv(B, args.model_dir, args.walk_policy, seed=seed, device="cuda:0", threads=args.threads)
    agent = agent_dict(NAV, TD, seed, ent, total)
    (out / "agent.json").write_text(json.dumps(agent, indent=1))
    (out / "run.json").write_text(json.dumps({"walk_policy": os.path.abspath(args.walk_policy), "num_envs": B,
                                              "seed": seed, "entropy_coef": ent, "iterations": total, "init": args.init,
                                              "image_view": env.view}, indent=1))
    runner = OnPolicyRunner(env, agent, log_dir=str(out), device="cuda:0")
    if args.init:
        runner.load(args.init, load_optimizer=False)
        runner.current_learning_iteration = 0
    from torch.utils.tensorboard import SummaryWriter
    runner.logger_type = "tensorboard"
    log = runner.writer = ScalarLog(SummaryWriter(log_dir=str(out), flush_secs=10))
    t0, done, first = time.time(), 0, True
    while done < total:
        n = min(args.chunk, total - done)
        t = time.time()
        runner.learn(num_learning_iterations=n, init_at_random_ep_len=first)
        runner.current_learning_iteration += 1     # RSL-RL 3.1.2 leaves it at the last iteration run, not the next
        first = False
        done += n
        runner.save(str(out / f"model_{done}.pt"))
        (out / "scalars.json").write_text(json.dumps(log.rows))
        print(f"[train_nav_mujoco] {done}/{total} iterations, {time.time() - t0:.0f} s, {(time.time() - t) / n:.2f} s/iteration",
              flush=True)
    runner.save(str(out / "model_final.pt"))
    (out / "timing.json").write_text(json.dumps({"iterations_done": done, "seconds": time.time() - t0, "num_envs": B}, indent=1))
    subprocess.run([sys.executable, str(ROOT / "scripts/export_nav_actor.py"), "--checkpoint", str(out / "model_final.pt"),
                    "--out", str(out / "nav_actor.pt"), "--note", f"MuJoCo navigation training, walker {args.walk_policy}"],
                   check=True, env={**os.environ, "PYTHONPATH": str(ROOT)})
    env.close()


if __name__ == "__main__":
    main()

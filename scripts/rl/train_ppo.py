"""Train PPO (Stable-Baselines3) on Go2NavEnv with vectorized environments.

Simulation only, with unfitted dynamics (config/sim.yaml).

Usage:
    python scripts/rl/train_ppo.py --timesteps 200000 --name smoke
    python scripts/rl/train_ppo.py --timesteps 5000000 --n-envs 16 --name full
Writes runs/<name>/monitor.csv (one row per episode), runs/<name>/progress.csv and
models/<name>.zip, with checkpoints in models/<name>_ckpt/.
"""
import argparse
import json
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from go2x.sim.nav_env import Go2NavEnv

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default="smoke")
    p.add_argument("--device", default="cpu", help="cpu is usually fastest for small MLPs")
    args = p.parse_args()

    run_dir, model_dir = ROOT / "runs" / args.name, ROOT / "models"
    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(exist_ok=True)

    env = make_vec_env(Go2NavEnv, n_envs=args.n_envs, seed=args.seed, vec_env_cls=SubprocVecEnv)
    env = VecMonitor(env, str(run_dir / "monitor.csv"), info_keywords=("is_success", "collision"))

    model = PPO("MlpPolicy", env, n_steps=256, batch_size=256, n_epochs=10, learning_rate=3e-4,
                gamma=0.99, gae_lambda=0.95, ent_coef=0.0, clip_range=0.2,
                policy_kwargs=dict(net_arch=[256, 256]), seed=args.seed, device=args.device, verbose=0)
    model.set_logger(configure(str(run_dir), ["csv", "stdout"]))
    ckpt = CheckpointCallback(max(500_000 // args.n_envs, 1), str(model_dir / f"{args.name}_ckpt"), name_prefix="ppo")

    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=ckpt)
    wall = time.time() - t0
    model.save(model_dir / f"{args.name}.zip")
    env.close()

    summary = {"timesteps": args.timesteps, "n_envs": args.n_envs, "seed": args.seed,
               "wall_s": round(wall, 1), "steps_per_s": round(args.timesteps / wall)}
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(summary)


if __name__ == "__main__":
    main()

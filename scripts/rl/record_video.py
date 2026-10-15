"""Record rollouts of a trained PPO policy as an MP4 (simulation only).

Usage:
    python scripts/rl/record_video.py --model models/full.zip --episodes 3
Writes results/rollout.mp4. Uses the evaluation seeds (10000 onward).
"""
import argparse
from pathlib import Path

import imageio.v2 as imageio
from stable_baselines3 import PPO

from go2x.sim.nav_env import Go2NavEnv

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default=str(ROOT / "models" / "full.zip"))
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--out", default=str(ROOT / "results" / "rollout.mp4"))
    args = p.parse_args()

    model = PPO.load(args.model, device="cpu")
    env = Go2NavEnv(render_mode="rgb_array")
    writer = imageio.get_writer(args.out, fps=env.metadata["render_fps"], codec="libx264", quality=6,
                                macro_block_size=1)
    for i in range(args.episodes):
        obs, _ = env.reset(seed=10_000 + i)
        writer.append_data(env.render())
        while True:
            obs, _, term, trunc, info = env.step(model.predict(obs, deterministic=True)[0])
            writer.append_data(env.render())
            if term or trunc:
                break
        outcome = "goal" if info["is_success"] else "collision" if info["collision"] else "timeout"
        print(f"episode {i}: {outcome} at {info['time_s']:.1f} s")
        for _ in range(10):  # hold the last frame for 0.5 s
            writer.append_data(env.render())
    writer.close()
    env.close()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

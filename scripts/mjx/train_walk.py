"""Train the Go2 walking policy in MuJoCo MJX with Brax PPO (config/walk_mjx.yaml, go2x/mjx/walk_env.py).

Writes to --out: params.pkl (Brax normalizer, policy and value parameters), policy.pt (TorchScript with the
Isaac walker's interface: 45-value observation in, 12 joint actions out, target = default + 0.25 * action),
progress.json (every evaluation: steps, wall time, reward and its terms), config.json, timing.json.

    python scripts/mjx/train_walk.py --model_dir <unitree_mujoco> --out DIR [--seed 0] [--timesteps N]
        [--no-randomize] [--num_envs N]
"""
import argparse
import functools
import json
import os
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402
import numpy as np  # noqa: E402
from brax.training.agents.ppo import networks as ppo_networks  # noqa: E402
from brax.training.agents.ppo import train as ppo  # noqa: E402
from mujoco_playground._src import wrapper  # noqa: E402

from go2x.config import load_config  # noqa: E402
from go2x.mjx.export import export_torchscript  # noqa: E402
from go2x.mjx.walk_env import Go2Walk, domain_randomize  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--num_envs", type=int, default=None)
    p.add_argument("--num_evals", type=int, default=None)
    p.add_argument("--no-randomize", action="store_true", help="ablation: no domain randomization, no noise, no pushes")
    p.add_argument("--symmetric", action="store_true", help="ablation: the critic sees the policy observation only")
    p.add_argument("--push_velocity", type=float, default=None,
                   help="variant: base velocity kicks up to this (m/s); legged_gym (Rudin et al., CoRL 2022) uses 1.0")
    p.add_argument("--small", action="store_true", help="CPU smoke test sizes")
    args = p.parse_args()
    cfg = load_config("walk_mjx")
    if args.push_velocity is not None:
        cfg["randomization"]["push_velocity_xy"] = args.push_velocity
    P = dict(cfg["ppo"])
    if args.timesteps:
        P["num_timesteps"] = args.timesteps
    if args.num_envs:
        P["num_envs"] = args.num_envs
    if args.num_evals:
        P["num_evals"] = args.num_evals
    if args.small:
        P.update(num_envs=8, batch_size=8, num_minibatches=2, unroll_length=4, num_evals=2)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    randomize = not args.no_randomize
    env = Go2Walk(args.model_dir, cfg, randomize=randomize)
    eval_env = Go2Walk(args.model_dir, cfg, randomize=randomize)
    run_cfg = {"ppo": P, "seed": args.seed, "randomize": randomize, "symmetric_critic": args.symmetric,
               "devices": [str(d) for d in jax.devices()], "jax": jax.__version__}
    (out / "config.json").write_text(json.dumps({**run_cfg, "walk_mjx": cfg}, indent=1))
    print(json.dumps(run_cfg), flush=True)

    net = functools.partial(ppo_networks.make_ppo_networks, policy_hidden_layer_sizes=tuple(P["policy_hidden"]),
                            value_hidden_layer_sizes=tuple(P["value_hidden"]), policy_obs_key="state",
                            value_obs_key="state" if args.symmetric else "privileged_state")
    rand_fn = functools.partial(domain_randomize, mj_model=env.mj_model, cfg=cfg) if randomize else None
    progress, t0 = [], time.time()
    times = [t0]

    def progress_fn(step, metrics):
        times.append(time.time())
        row = {"step": int(step), "wall_s": times[-1] - t0}
        row.update({k: float(v) for k, v in metrics.items() if np.ndim(v) == 0})
        progress.append(row)
        (out / "progress.json").write_text(json.dumps(progress, indent=0))
        print(f"[walk_mjx] step {step:>11,}  {row['wall_s']:7.0f} s  eval reward "
              f"{row.get('eval/episode_reward', float('nan')):8.3f}  len {row.get('eval/avg_episode_length', float('nan')):6.1f}",
              flush=True)

    snaps = out / "snapshots"
    snaps.mkdir(exist_ok=True)

    def policy_params_fn(step, make_policy, params):
        # TorchScript snapshot at every evaluation, for the learning-progress video
        try:
            export_torchscript(jax.device_get(params), P["policy_hidden"], cfg, snaps / f"policy_{int(step):011d}.pt")
        except Exception as exc:          # a snapshot must never stop training
            print(f"[walk_mjx] snapshot at {step} failed: {exc}", flush=True)

    train_fn = functools.partial(
        ppo.train, num_timesteps=P["num_timesteps"], num_evals=P["num_evals"], reward_scaling=P["reward_scaling"],
        episode_length=cfg["timing"]["episode_steps"], normalize_observations=P["normalize_observations"],
        action_repeat=1, unroll_length=P["unroll_length"], num_minibatches=P["num_minibatches"],
        num_updates_per_batch=P["num_updates_per_batch"], discounting=P["discounting"],
        learning_rate=P["learning_rate"], entropy_cost=P["entropy_cost"], num_envs=P["num_envs"],
        batch_size=P["batch_size"], max_grad_norm=P["max_grad_norm"], network_factory=net, seed=args.seed,
        randomization_fn=rand_fn, wrap_env_fn=wrapper.wrap_for_brax_training,
        num_eval_envs=min(128, P["num_envs"]), progress_fn=progress_fn, policy_params_fn=policy_params_fn)
    make_inference_fn, params, _ = train_fn(environment=env, eval_env=eval_env)
    t_train = time.time() - t0
    params = jax.device_get(params)
    with open(out / "params.pkl", "wb") as f:
        pickle.dump(params, f)
    export_torchscript(params, P["policy_hidden"], cfg, out / "policy.pt")
    # check: the TorchScript policy and Brax's deterministic inference agree on a few observations
    import torch
    infer = make_inference_fn(params, deterministic=True)
    rng = jax.random.PRNGKey(0)
    obs = eval_env.reset(rng).obs
    a_brax = np.asarray(infer(obs, rng)[0]) * cfg["actuator"]["action_scale"] / cfg["actuator"]["interface_scale"]
    ts = torch.jit.load(str(out / "policy.pt"))
    with torch.no_grad():
        a_ts = ts(torch.from_numpy(np.array(obs["state"], np.float32))[None])[0].numpy()
    err = float(np.abs(a_brax - a_ts).max())
    print(f"[walk_mjx] TorchScript vs Brax inference max difference {err:.2e}")
    assert err < 1e-4, err
    (out / "timing.json").write_text(json.dumps({
        "train_s": t_train, "env_steps": P["num_timesteps"], "steps_per_s": P["num_timesteps"] / max(t_train, 1e-9),
        "first_eval_s": times[1] - t0 if len(times) > 1 else None, "export_check_max_abs_diff": err}, indent=1))
    print(f"[walk_mjx] done in {t_train / 60:.1f} min")


if __name__ == "__main__":
    main()

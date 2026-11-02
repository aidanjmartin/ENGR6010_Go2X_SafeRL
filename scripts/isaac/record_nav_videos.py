"""Rollout videos from the rendered camera: one success and one failure per controller (Task 4).

Reads the evaluation results (nav_eval_<controller>.json), replays the chosen scenarios in the rendered
variant of the environment (building boxes and the fixed camera), and writes mp4 files at the navigation rate.
Each frame shows the camera image with the drawn top-down input (current frame, enlarged) beside it.

    python scripts/isaac/record_nav_videos.py --walk_policy policy.pt --checkpoint model_final.pt \
        --evals DIR_WITH_JSONS --out DIR
"""
import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--walk_policy", required=True)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--evals", nargs="+", required=True, help="nav_eval_<controller>.json files")
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
os.environ["GO2X_WALK_POLICY"] = os.path.abspath(args.walk_policy)
app = AppLauncher(args).app

import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

from go2x.atrium.baselines import AStarController, StraightController  # noqa: E402
from go2x.atrium.rewards import normalize_action  # noqa: E402
from go2x.config import CONFIG_DIR  # noqa: E402
from go2x.isaac.nav_env_cfg import NAV, TD, AtriumNavEnvCfg, add_render_parts, get_runtime, nav_agent_dict  # noqa: E402
from go2x.isaac.shutdown import finish  # noqa: E402

scenarios = json.load(open(CONFIG_DIR / "nav_eval_scenarios.json"))["scenarios"]
jobs = []
for path in args.evals:
    d = json.load(open(path))
    ctl = d["summary"]["controller"]
    eps = d["episodes"]
    ok = [e["scenario"] for e in eps if e["success"]]
    bad = [e["scenario"] for e in eps if not e["success"]]
    for label, ids in (("success", ok), ("failure", bad)):
        for sid in ids[:5]:            # a replay may end differently (camera noise, randomization): try up to 5
            jobs.append((ctl, label, sid))
print("[videos]", jobs, flush=True)

cfg = add_render_parts(AtriumNavEnvCfg())
cfg.scene.num_envs = 1
cfg.go2x_scenarios = [scenarios[s] for _, _, s in jobs]
env = ManagerBasedRLEnv(cfg=cfg)
wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
cam = env.scene["top_camera"]
S = TD["size"]
policy = None
if args.checkpoint:
    from rsl_rl.runners import OnPolicyRunner

    from go2x.nav_policy import register_with_rsl_rl

    register_with_rsl_rl()
    agent = json.load(open(os.path.join(os.path.dirname(args.checkpoint), "agent.json")))
    runner = OnPolicyRunner(wrapped, nav_agent_dict(agent), log_dir=None, device=str(env.device))
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.device)

os.makedirs(args.out, exist_ok=True)
obs = wrapped.get_observations()   # the wrapper's reset loaded jobs[0]
summary, done_labels = [], set()
for n, (ctl, label, sid) in enumerate(jobs):
    if (ctl, label) in done_labels:
        continue
    rt = get_runtime(env)
    # an episode end auto-resets into the next scenario; reset further only until scenario n is loaded
    while int(rt.batch.scenario_id[0]) < n:
        env.reset()
        obs = wrapped.get_observations()
    assert int(rt.batch.scenario_id[0]) == n, "scenario order"
    c = {"straight": StraightController, "astar": AStarController}.get(ctl)
    c = c() if c else None
    frames = []
    with torch.inference_mode():
        for t in range(int(env.max_episode_length)):
            img = obs["policy"][:, : 4 * S * S].reshape(1, 4, S, S)
            if ctl == "policy":
                if policy is None:
                    break
                act = policy(obs)
            else:
                act = normalize_action(torch.tensor(c.act(img[0].cpu().numpy()), dtype=torch.float32,
                                                    device=env.device)[None], NAV["action"])
            rgb = cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
            top = img[0, [0, 1, 2]].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            top = top + img[0, 3, :, :, None].cpu().numpy() * np.array([0.0, 0.0, 0.0])
            top[..., 1] = np.maximum(top[..., 1], img[0, 3].cpu().numpy() * 0.8)   # target in green
            panel = np.kron((top * 255).astype(np.uint8), np.ones((7, 7, 1), np.uint8))
            pad = np.full((rgb.shape[0], panel.shape[1], 3), 255, np.uint8)
            pad[: panel.shape[0]] = panel[: rgb.shape[0]]
            frames.append(np.concatenate([rgb, pad], 1))
            obs, _, dones, _ = wrapped.step(act)
            if bool(dones[0]):
                break
    outcome = "success" if bool(env.termination_manager.get_term("success")[0]) else "failure"
    reasons = [k for k in ("success", "collision_footprint", "collision_force", "fall", "out_of_square", "time_out")
               if bool(env.termination_manager.get_term(k)[0])]
    if outcome != label or not frames:
        print(f"[videos] {ctl} scenario {sid}: replay ended in {reasons}, wanted {label}; trying the next", flush=True)
        continue
    path = os.path.join(args.out, f"nav_{ctl}_{label}_scenario{sid:03d}.mp4")
    imageio.mimwrite(path, frames, fps=int(NAV["timing"]["nav_hz"]), quality=6)
    done_labels.add((ctl, label))
    summary.append({"controller": ctl, "label": label, "scenario": sid, "frames": len(frames), "ended_by": reasons,
                    "file": os.path.basename(path)})
    print("[videos]", summary[-1], flush=True)
json.dump(summary, open(os.path.join(args.out, "nav_videos.json"), "w"), indent=1)
finish(0)

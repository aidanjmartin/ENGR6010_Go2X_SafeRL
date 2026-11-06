"""Navigation policy deployment (go2x/deploy, scripts/robot/run_nav_policy.py) on the CPU, without a robot."""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from go2x.config import load_config
from go2x.deploy.nav_runtime import NavPolicy, OdomToSquare, SquareScene, load_scenario, wrap
from go2x.nav_net import ImageMLP

ROOT = Path(__file__).resolve().parents[1]


def test_odometry_mapping_anchors_and_follows_rigid_motion():
    start = np.array([0.7, -1.1, -2.3])
    first = np.array([5.0, 2.0, 0.4])                     # odometry frame: arbitrary origin and heading
    m = OdomToSquare(start, first)
    assert np.allclose(m(first), start)
    # 1 m forward and 0.5 m left in the robot's start frame, then turned 0.3 rad
    c, s = math.cos(first[2]), math.sin(first[2])
    odom = np.array([first[0] + c * 1.0 - s * 0.5, first[1] + s * 1.0 + c * 0.5, first[2] + 0.3])
    c2, s2 = math.cos(start[2]), math.sin(start[2])
    expect = np.array([start[0] + c2 * 1.0 - s2 * 0.5, start[1] + s2 * 1.0 + c2 * 0.5, wrap(start[2] + 0.3)])
    assert np.allclose(m(odom), expect, atol=1e-9)


def _export(tmp_path, view="robot"):
    NAV, TD = load_config("nav"), load_config("topdown")
    P = NAV["policy"]
    torch.manual_seed(3)
    net = ImageMLP((8, 64, 64), 3, 3, P["cnn_channels"], P["cnn_out"], P["mlp"])
    meta = {"image_shape": [8, 64, 64], "extra_dim": 3, "cnn_channels": P["cnn_channels"], "cnn_out": P["cnn_out"],
            "mlp": P["mlp"], "image_view": view, "topdown": TD, "action": NAV["action"], "nav_hz": 10,
            "success": NAV["success"]}
    path = tmp_path / "actor.pt"
    torch.save({"state_dict": net.state_dict(), "meta": meta}, path)
    return net, path


def test_nav_policy_matches_training_input_layout(tmp_path):
    net, path = _export(tmp_path)
    pol = NavPolicy(path)
    f1, f2 = torch.rand(4, 64, 64), torch.rand(4, 64, 64)
    a1, cmd1 = pol.act(f1)
    x1 = torch.cat([torch.cat([f1, f1], 0).flatten(), torch.zeros(3)])[None]   # first step: previous frame = current
    assert np.allclose(a1, net(x1)[0].clamp(-1, 1).detach().numpy(), atol=1e-6)
    a2, _ = pol.act(f2)
    x2 = torch.cat([torch.cat([f2, f1], 0).flatten(), torch.as_tensor(a1)])[None]
    assert np.allclose(a2, net(x2)[0].clamp(-1, 1).detach().numpy(), atol=1e-6)
    lim = load_config("nav")["action"]
    for v, k in zip(cmd1, ("vx", "vy", "wz")):
        assert lim[k][0] - 1e-6 <= v <= lim[k][1] + 1e-6


def test_exported_actor_in_repository_loads():
    p = ROOT / "results/phase2/nav_policy/nav_actor.pt"
    if not p.exists():
        return
    pol = NavPolicy(p)
    assert pol.view == "robot"
    a, cmd = pol.act(torch.zeros(4, 64, 64))
    assert a.shape == (3,) and cmd.shape == (3,)


def test_square_scene_start_is_clear_and_inside():
    scen, _ = load_scenario(0)
    sc = SquareScene(scen)
    assert np.allclose(sc.start_pose, scen["robot_start"], atol=1e-6)
    chk = sc.checks(sc.start_pose, 0.0)
    assert not chk["collision_footprint"] and not chk["out_of_square"] and not chk["at_target"]
    img = sc.draw(sc.start_pose, NavPolicy.__init__.__globals__["view_config"](load_config("topdown"), "robot"))
    assert img.shape == (4, 64, 64) and img[3].sum() > 0


def test_runner_end_to_end_on_mock(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    r = subprocess.run([sys.executable, str(ROOT / "scripts/robot/run_nav_policy.py"), "--backend", "mock", "--scenario", "0",
                        "--max-policy-s", "1.0", "--data-dir", str(tmp_path)], capture_output=True, text=True, env=env,
                       timeout=120, stdin=subprocess.DEVNULL)
    assert r.returncode == 0, r.stdout + r.stderr
    run_dir = next(tmp_path.iterdir())
    for f in ("header.json", "state.jsonl", "commands.jsonl", "events.jsonl", "run.json", "policy.npz"):
        assert (run_dir / f).exists(), f
    run = json.loads((run_dir / "run.json").read_text())
    assert run["policy_steps"] == 10
    assert np.allclose(run["mapping"]["start_pose_square"], run["scenario"]["robot_start"], atol=1e-6)
    d = np.load(run_dir / "policy.npz")
    assert d["frame"].shape == (10, 4, 64, 64)
    sent = np.abs(d["cmd_sent"]).max(0)
    lim = load_config("safety_limits_nav")
    assert sent[0] <= lim["max_vx"] + 1e-9 and sent[2] <= lim["max_wz"] + 1e-9
    events = [json.loads(line)["event"] for line in open(run_dir / "events.jsonl")]
    assert "policy_start" in events and "stop_move_sent" in events


def test_runner_refuses_walking_people_on_the_robot():
    sc = json.load(open(ROOT / "config/nav_eval_scenarios.json"))["scenarios"]
    i = next(k for k, s in enumerate(sc) if any(p["walking"] for p in s["people"]))
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    r = subprocess.run([sys.executable, str(ROOT / "scripts/robot/run_nav_policy.py"), "--backend", "webrtc",
                        "--scenario", str(i)], capture_output=True, text=True, env=env, timeout=60, stdin=subprocess.DEVNULL)
    assert r.returncode != 0 and "walking people" in (r.stdout + r.stderr)

"""Task 3 rendered check: render the fixed camera's RGB image for random atrium scenes, with ground truth.

For each scene: the layout is drawn from the scene sampler (seed config/camera.yaml rendering.seed), the robot
stands for `settle_steps` navigation steps under a zero velocity target, then the camera image is saved with
the scene's ground truth (object classes, colors, footprints, robot pose, camera model) and the drawn top-down
image of the true state (no camera noise).

    python scripts/isaac/render_scenes.py --walk_policy policy.pt --out $DATA_DIR/scenes
"""
import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--walk_policy", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--settle_steps", type=int, default=10)
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

from go2x.atrium.camera import Camera  # noqa: E402
from go2x.atrium.rewards import normalize_action  # noqa: E402
from go2x.atrium.scene import layout_to_dict, sample_layout  # noqa: E402
from go2x.atrium.topdown import draw  # noqa: E402
from go2x.isaac.nav_env_cfg import ATRIUM, CAM, NAV, TD, AtriumNavEnvCfg, add_render_parts, get_runtime  # noqa: E402
from go2x.isaac.shutdown import finish  # noqa: E402

rc = CAM["rendering"]
rng = np.random.default_rng(rc["seed"])
layouts = [sample_layout(rng, ATRIUM)[0] for _ in range(rc["num_scenes"])]
cfg = add_render_parts(AtriumNavEnvCfg())
cfg.scene.num_envs = 1
cfg.go2x_scenarios = [layout_to_dict(L) for L in layouts]
cfg.terminations.time_out = None
env = ManagerBasedRLEnv(cfg=cfg)
cam_sensor = env.scene["top_camera"]
cam = Camera.from_config(CAM)
os.makedirs(args.out, exist_ok=True)
zero = normalize_action(torch.zeros(1, 3, device=env.device), NAV["action"])
index = []

obs, _ = env.reset()
for i in range(rc["num_scenes"]):
    if i > 0:
        obs, _ = env.reset()
    rt = get_runtime(env)
    sid = int(rt.batch.scenario_id[0])
    for _ in range(args.settle_steps):
        obs, _, term, trunc, _ = env.step(zero)
    rgb = cam_sensor.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
    pose = rt.robot_pose()
    st = rt.batch.perception_state(pose)
    top = draw(pose, tuple(ATRIUM["robot"]["footprint"]), st["rects"], st["rect_mask"], st["circles"],
               st["circle_mask"], st["target_rect"], st["target_circle"], st["target_is_circle"], TD)[0].cpu().numpy()
    L = rt.batch.layouts[0]
    objs = []
    for o in L.objects():
        d = {"id": o["id"], "class": o["kind"], "color": o["color"]}
        if "rect" in o:
            d["footprint_rect"] = [float(v) for v in o["rect"]]
        else:
            k = int(o["id"].split("_")[1])
            d["footprint_circle"] = [float(v) for v in rt.batch.circles[0, k].tolist()]
            d["walking"] = o["walking"]
        objs.append(d)
    name = f"scene_{i:03d}"
    imageio.imwrite(os.path.join(args.out, name + ".png"), rgb)
    np.save(os.path.join(args.out, name + "_topdown.npy"), top.astype(np.float32))
    imageio.imwrite(os.path.join(args.out, name + "_topdown.png"),
                    (np.clip(top[[0, 1, 2]].transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8))
    gt = {"scene": i, "scenario_id": sid, "objects": objs, "target": L.target_id,
          "robot_pose_xy_yaw": [float(v) for v in pose[0].tolist()],
          "robot_fell_or_ended": bool(term[0] or trunc[0]),
          "camera": {"K": cam.K.tolist(), "R_wc": cam.R_wc.tolist(), "C": cam.C.tolist(),
                     "H_floor_to_image": cam.floor_homography().tolist()},
          "frame": "square frame = aeb_level1 world frame (origin at the square center)"}
    with open(os.path.join(args.out, name + ".json"), "w") as f:
        json.dump(gt, f, indent=1)
    index.append({"scene": i, "image": name + ".png", "objects": len(objs), "ended_early": gt["robot_fell_or_ended"]})
    print(f"[render_scenes] {name}: {len(objs)} objects, rgb mean {rgb.mean():.1f}", flush=True)
with open(os.path.join(args.out, "index.json"), "w") as f:
    json.dump(index, f, indent=1)
finish(0)

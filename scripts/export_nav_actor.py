"""Export the navigation actor from an RSL-RL checkpoint to a small file the robot runner loads on any CPU.

The file holds the actor's weights (go2x.nav_policy.ImageMLP), its input settings (image shape, image view,
drawing settings) and the action limits, so scripts/robot/run_nav_policy.py needs neither Isaac Lab nor RSL-RL.

    python scripts/export_nav_actor.py --checkpoint model_final.pt --out results/phase2/nav_policy/nav_actor.pt
"""
import argparse
import hashlib
import os

import torch

from go2x.config import load_config

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--note", default="")
args = parser.parse_args()

NAV, TD = load_config("nav"), load_config("topdown")
sd = torch.load(args.checkpoint, map_location="cpu")["model_state_dict"]
actor = {k[len("actor."):]: v for k, v in sd.items() if k.startswith("actor.")}
P = NAV["policy"]
meta = {
    "image_shape": [2 * 4, TD["size"], TD["size"]],
    "extra_dim": 3,
    "cnn_channels": list(P["cnn_channels"]), "cnn_out": P["cnn_out"], "mlp": list(P["mlp"]),
    "image_view": P["image_view"],
    "topdown": TD,
    "action": NAV["action"],
    "nav_hz": NAV["timing"]["nav_hz"],
    "success": NAV["success"],
    "source_checkpoint": os.path.abspath(args.checkpoint),
    "source_sha256": hashlib.sha256(open(args.checkpoint, "rb").read()).hexdigest(),
    "note": args.note,
}
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
torch.save({"state_dict": actor, "meta": meta}, args.out)
print(f"wrote {args.out}: {sum(v.numel() for v in actor.values())} actor values, view {meta['image_view']}")

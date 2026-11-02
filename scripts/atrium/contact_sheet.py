"""Tile rendered scene images (with their drawn top-down images) into one JPEG for review."""
import argparse
import glob
import os

import imageio
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--scenes", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--n", type=int, default=12)
args = parser.parse_args()
rows = []
for path in sorted(glob.glob(os.path.join(args.scenes, "scene_[0-9][0-9][0-9].png")))[: args.n]:
    rgb = imageio.imread(path)[::2, ::2, :3]
    top = imageio.imread(path.replace(".png", "_topdown.png"))[..., :3]
    top = np.kron(top, np.ones((3, 3, 1), dtype=np.uint8))[: rgb.shape[0]]
    pad = np.full((rgb.shape[0], top.shape[1], 3), 255, np.uint8)
    pad[: top.shape[0]] = top
    rows.append(np.concatenate([rgb, pad], 1))
pairs = [np.concatenate(rows[i:i + 2], 1) for i in range(0, len(rows) - 1, 2)]
imageio.imwrite(args.out, np.concatenate(pairs, 0), quality=85)
print("wrote", args.out)

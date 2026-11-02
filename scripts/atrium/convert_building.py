"""Task 2: check the atrium square against aeb_level1.sdf and export the nearby building boxes.

Reads the world straight from a tourbot clone at the pinned commit (read-only; `git show`), so no box is
copied by hand. Writes config/building_level1_near_square.json (boxes for Isaac Lab) and
results/phase2/square_check.json.

    python scripts/atrium/convert_building.py --tourbot ~/weekend/tourbot
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from go2x.atrium.building import boxes_to_json, parse_boxes, square_report
from go2x.config import CONFIG_DIR, load_config

SDF_PATH = "ros2_ws/src/tourbot_sim/worlds/aeb_level1.sdf"

parser = argparse.ArgumentParser()
parser.add_argument("--tourbot", required=True, help="a clone of aidanjmartin/tourbot")
args = parser.parse_args()

cfg = load_config("atrium")
commit = subprocess.run(["git", "-C", args.tourbot, "rev-parse", cfg["building"]["sdf_commit"]],
                        check=True, capture_output=True, text=True).stdout.strip()
text = subprocess.run(["git", "-C", args.tourbot, "show", f"{commit}:{SDF_PATH}"],
                      check=True, capture_output=True, text=True).stdout
boxes = [b for b in parse_boxes(text) if b.kind in cfg["building"]["kinds"]]
sq = cfg["square"]
report, near = square_report(boxes, sq["center"], sq["side"], cfg["building"]["margin"])
source = {"repo": "aidanjmartin/tourbot", "commit": commit, "path": SDF_PATH,
          "sha256": hashlib.sha256(text.encode()).hexdigest()}
report = {"source": source, "square_center": sq["center"], "square_side_m": sq["side"],
          "margin_m": cfg["building"]["margin"], **report}

root = Path(__file__).resolve().parents[2]
out_boxes = CONFIG_DIR / "building_level1_near_square.json"
with open(out_boxes, "w") as f:
    json.dump({"generated_by": "scripts/atrium/convert_building.py", "source": source,
               "frame": "aeb_level1 world, meters, origin = Central Lobby (100)", "boxes": boxes_to_json(near)}, f, indent=1)
(root / "results" / "phase2").mkdir(parents=True, exist_ok=True)
with open(root / "results" / "phase2" / "square_check.json", "w") as f:
    json.dump(report, f, indent=1)
print(json.dumps(report, indent=1))
print("wrote", out_boxes)

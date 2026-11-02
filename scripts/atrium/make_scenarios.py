"""Write the fixed navigation evaluation scenarios (Task 4), shared by every controller.

    python scripts/atrium/make_scenarios.py      # writes config/nav_eval_scenarios.json
"""
import collections
import json

import numpy as np

from go2x.atrium.scene import layout_to_dict, sample_layout
from go2x.config import CONFIG_DIR, load_config

nav, atrium = load_config("nav"), load_config("atrium")
ev = nav["evaluation"]
rng = np.random.default_rng(ev["seed"])
scenarios = [layout_to_dict(sample_layout(rng, atrium)[0]) for _ in range(ev["num_scenarios"])]
kinds = collections.Counter(s["target"][0] for s in scenarios)
walking = sum(p["walking"] for s in scenarios for p in s["people"])
summary = {"num_scenarios": len(scenarios), "seed": ev["seed"], "targets": dict(kinds),
           "people": sum(len(s["people"]) for s in scenarios), "walking_people": walking,
           "chairs": sum(len(s["chairs"]) for s in scenarios), "tables": sum(len(s["table"]) for s in scenarios)}
with open(CONFIG_DIR / "nav_eval_scenarios.json", "w") as f:
    json.dump({"generated_by": "scripts/atrium/make_scenarios.py", "summary": summary, "scenarios": scenarios}, f)
print(json.dumps(summary, indent=1))

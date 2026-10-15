"""Load the YAML files in config/."""
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def load_config(name):
    """Load config/<name>.yaml as a dict."""
    with open(CONFIG_DIR / f"{name}.yaml") as f:
        return yaml.safe_load(f)

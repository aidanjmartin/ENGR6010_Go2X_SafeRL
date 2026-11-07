"""Fixed camera exposure through uvc-util (github.com/jtfrey/uvc-util, built by scripts/robot/setup_laptop.sh).

OpenCV on macOS cannot change a webcam's exposure. On auto exposure the Brio 100 brightens the picture for the
dark atrium floor, which washes out the white paper and turns the tags' black to grey; with the lights dimmed,
corner tags 2 and 3 were read in 13 and 30 of 60 frames (2026-11-07). Locked at 16.7 ms and gain 96, every tag
on the floor was read in every frame; 8.3 ms with gain 160 read them equally well (robot tag 59 of 60) and is
used since the forward limit went to 0.8 m/s: the robot moves 7 mm during an exposure, a quarter of a tag cell.
8.3 ms is one cycle of 60 Hz lighting (no flicker bands). The settings stay until the camera loses power, so the launcher
applies them before every run. They are kept in config/camera_real.yaml under `controls`.
"""
import shutil
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
UVC_UTIL = ROOT / "third_party" / "uvc-util" / "uvc-util"
ORDER = ("auto-exposure-mode", "backlight-compensation", "exposure-time-abs", "gain")   # mode first: manual


def load_controls(calib_path):
    d = yaml.safe_load(open(calib_path)) or {}
    return d.get("controls")


def apply_controls(controls, exe=None):
    """Set the controls on the named device; returns {control: value read back}. Raises RuntimeError."""
    exe = Path(exe or UVC_UTIL)
    if not exe.exists() and not shutil.which(str(exe)):
        raise RuntimeError(f"{exe} missing: run scripts/robot/setup_laptop.sh (builds uvc-util)")
    dev = controls["device"]
    names = [k for k in ORDER if k in controls] + [k for k in controls if k not in ORDER and k != "device"]
    args = [str(exe), "-N", dev]
    for k in names:
        args += ["-s", f"{k}={controls[k]}"]
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"uvc-util could not set the controls on '{dev}': {(r.stderr or r.stdout).strip()}")
    args = [str(exe), "-N", dev]
    for k in names:
        args += ["-o", k]
    r = subprocess.run(args, capture_output=True, text=True)
    vals = r.stdout.split()
    return dict(zip(names, vals))

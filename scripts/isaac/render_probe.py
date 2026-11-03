"""Check that the RTX renderer starts and renders one camera frame (jobs 040, 045 and 065 crashed at start).

    python scripts/isaac/render_probe.py --out DIR [--kit_args "..."]

Prints "render_probe OK <mean pixel>" and writes probe.png on success.
"""
import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
app = AppLauncher(args).app
print("render_probe app started", flush=True)

import imageio  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402

from go2x.isaac.shutdown import finish  # noqa: E402

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device="cuda:0"))
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
sim_utils.DomeLightCfg(intensity=2000.0).func("/World/light", sim_utils.DomeLightCfg(intensity=2000.0))
sim_utils.CuboidCfg(size=(0.5, 0.5, 0.5), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))
                    ).func("/World/box", sim_utils.CuboidCfg(size=(0.5, 0.5, 0.5)), translation=(0.0, 0.0, 0.25))
cam = Camera(CameraCfg(prim_path="/World/cam", height=120, width=160, data_types=["rgb"],
                       spawn=sim_utils.PinholeCameraCfg(focal_length=24.0),
                       offset=CameraCfg.OffsetCfg(pos=(3.0, 0.0, 4.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world")))
sim.reset()
cam.set_world_poses_from_view(eyes=__import__("torch").tensor([[3.0, 0.0, 4.0]], device="cuda:0"),
                              targets=__import__("torch").tensor([[0.0, 0.0, 0.0]], device="cuda:0"))
for _ in range(10):
    sim.step()
    cam.update(0.005)
rgb = cam.data.output["rgb"][0].cpu().numpy()
os.makedirs(args.out, exist_ok=True)
imageio.imwrite(os.path.join(args.out, "probe.png"), rgb[..., :3])
print(f"render_probe OK {rgb[..., :3].mean():.2f}", flush=True)
finish(0)

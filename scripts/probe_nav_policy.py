"""Probe a trained navigation actor (CPU): its mean velocity target as the target's bearing changes.

For 512 sampled starts, the robot is turned so the target sits at each bearing from -180 to 180 degrees (left
positive), the drawn image is made without camera noise (current frame = previous frame, previous action 0),
and the actor's mean output is scaled to a velocity target. A policy that steers by the image turns left for
targets on the left and right for targets on the right. Used for job 050 attempt 3 (LOG.md step 10).

    python scripts/probe_nav_policy.py model_final.pt [world|robot]
"""
import sys, math, torch, numpy as np
from go2x.config import load_config
from go2x.atrium.state import AtriumBatch
from go2x.atrium.topdown import draw, view_config
from go2x.atrium.rewards import scale_action, front_point
from go2x.nav_policy import ImageMLP

ck = sys.argv[1]
A, TD, NAV = load_config("atrium"), load_config("topdown"), load_config("nav")
TD = view_config(TD, sys.argv[2] if len(sys.argv) > 2 else "world")
sd = torch.load(ck, map_location="cpu")["model_state_dict"]
P = NAV["policy"]
net = ImageMLP((8, 64, 64), 3, 3, P["cnn_channels"], P["cnn_out"], P["mlp"])
net.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith("actor.")})
net.eval()
B = 512
b = AtriumBatch(B, "cpu", seed=11, atrium_cfg=A)
pose0 = b.reset(list(range(B)))
center = torch.where(b.target_is_circle[:, None], b.target_circle[:, :2], b.target_rect[:, :2])
d = center - pose0[:, :2]
to_target = torch.atan2(d[:, 1], d[:, 0])
print("bearing (deg), mean and standard deviation over starts of vx (m/s) and wz (rad/s)")
rows = []
for bdeg in range(-180, 181, 30):
    yaw = to_target - math.radians(bdeg)       # target at +bdeg (left positive) from heading
    pose = torch.stack([pose0[:, 0], pose0[:, 1], yaw], -1)
    st = b.perception_state(pose)
    img = draw(pose, tuple(A["robot"]["footprint"]), st["rects"], st["rect_mask"], st["circles"], st["circle_mask"],
               st["target_rect"], st["target_circle"], st["target_is_circle"], TD)
    x = torch.cat([img, img], 1).flatten(1)
    with torch.no_grad():
        a = net(torch.cat([x, torch.zeros(B, 3)], -1))
    cmd = scale_action(a, NAV["action"])
    rows.append((bdeg, cmd[:, 0].mean().item(), cmd[:, 2].mean().item(), cmd[:, 0].std().item(), cmd[:, 2].std().item()))
    print(f"{bdeg:5d}   vx {rows[-1][1]:.3f}±{rows[-1][3]:.3f}  wz {rows[-1][2]:+.3f}±{rows[-1][4]:.3f}")

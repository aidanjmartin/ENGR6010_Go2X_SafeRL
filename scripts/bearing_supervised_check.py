"""Can the navigation network (go2x.nav_policy.ImageMLP) read the target's bearing from the drawn image? (CPU)

Supervised regression of (cos, sin) of the target bearing in the robot frame from noise-free drawn images of
sampled starts with random headings, 128 images per step, Adam 3e-4; prints the held-out (512 images) angle
error. Compares the world view with the robot view (LOG.md step 10).

    python scripts/bearing_supervised_check.py STEPS world|robot
"""
import sys, math, time, torch
from go2x.config import load_config
from go2x.atrium.state import AtriumBatch
from go2x.atrium.topdown import draw, view_config
from go2x.nav_policy import ImageMLP
torch.manual_seed(0); torch.set_num_threads(8)
A, TD, NAV = load_config("atrium"), load_config("topdown"), load_config("nav")
P = NAV["policy"]
steps = int(sys.argv[1]); TD = view_config(TD, sys.argv[2])
def batch(B, seed):
    b = AtriumBatch(B, "cpu", seed=seed, atrium_cfg=A)
    pose = b.reset(list(range(B)))
    pose[:, 2] = (torch.rand(B) * 2 - 1) * math.pi            # any heading
    st = b.perception_state(pose)
    img = draw(pose, tuple(A["robot"]["footprint"]), st["rects"], st["rect_mask"], st["circles"], st["circle_mask"],
               st["target_rect"], st["target_circle"], st["target_is_circle"], TD)
    c = torch.where(b.target_is_circle[:, None], b.target_circle[:, :2], b.target_rect[:, :2])
    d = c - pose[:, :2]
    beta = torch.atan2(d[:, 1], d[:, 0]) - pose[:, 2]
    x = torch.cat([img, img], 1).flatten(1)
    return torch.cat([x, torch.zeros(B, 3)], -1), torch.stack([torch.cos(beta), torch.sin(beta)], -1)
net = ImageMLP((8, 64, 64), 3, 2, P["cnn_channels"], P["cnn_out"], P["mlp"])
opt = torch.optim.Adam(net.parameters(), 3e-4)
xt, yt = batch(512, 999)
t0 = time.time()
for s in range(steps + 1):
    x, y = batch(128, s)
    loss = ((net(x) - y) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if s % 250 == 0:
        with torch.no_grad():
            p = net(xt)
            err = torch.rad2deg(torch.atan2(p[:, 1], p[:, 0]) - torch.atan2(yt[:, 1], yt[:, 0]))
            err = (err + 180) % 360 - 180
        print(f"step {s:5d}  loss {loss.item():.4f}  held-out bearing error median {err.abs().median():.1f} deg, "
              f"mean {err.abs().mean():.1f} deg  ({time.time() - t0:.0f} s)", flush=True)

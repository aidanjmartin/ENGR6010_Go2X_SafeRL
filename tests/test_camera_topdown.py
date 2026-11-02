"""Camera projection, floor homography, coverage, and the drawn top-down image with camera realism."""
import copy

import numpy as np
import torch

from go2x.atrium.camera import Camera, coverage_check, rot_to_quat
from go2x.atrium.topdown import CameraFeed, draw, pixel_centers, world_to_pixel
from go2x.config import load_config

TD = load_config("topdown")


def test_coverage_of_square_and_robot():
    r = coverage_check()
    assert r["all_inside"], r
    assert r["robot_top_inside"] == r["robot_top_points"]


def test_coverage_fails_when_camera_looks_away():
    cfg = copy.deepcopy(load_config("camera"))
    cfg["pose"]["look_at"] = [-3.0, 0.0, 0.0]
    assert not coverage_check(cam_cfg=cfg)["all_inside"]


def test_homography_matches_projection_and_inverts():
    cam = Camera.from_config()
    rng = np.random.default_rng(0)
    xy = rng.uniform(-1.8, 1.8, (200, 2))
    uv, depth = cam.project(np.column_stack([xy, np.zeros(200)]))
    assert np.all(depth > 0)
    H = cam.floor_homography()
    p = np.column_stack([xy, np.ones(200)]) @ H.T
    assert np.allclose(p[:, :2] / p[:, 2:], uv, atol=1e-8)
    assert np.allclose(cam.pixel_to_floor(uv), xy, atol=1e-8)


def test_camera_axes_and_quaternion():
    cam = Camera.from_config()
    R = cam.R_wc
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-12) and np.isclose(np.linalg.det(R), 1)
    # the look-at point projects to the image center
    uv, _ = cam.project(np.array(load_config("camera")["pose"]["look_at"], float))
    assert np.allclose(uv[0], [cam.width / 2, cam.height / 2])
    q = rot_to_quat(R)
    w, x, y, z = q
    Rq = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                   [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                   [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    assert np.allclose(Rq, R, atol=1e-9)


def _state(B=1):
    return {
        "robot_pose": torch.tensor([[0.0, 0.0, 0.0]] * B),
        "rects": torch.tensor([[[1.0, 1.0, 0.0, 0.25, 0.25]]] * B),
        "rect_mask": torch.tensor([[True]] * B),
        "circles": torch.tensor([[[-1.0, -1.0, 0.25]]] * B),
        "circle_mask": torch.tensor([[True]] * B),
        "target_rect": torch.tensor([[1.0, 1.0, 0.0, 0.25, 0.25]] * B),
        "target_circle": torch.zeros(B, 3),
        "target_is_circle": torch.tensor([False] * B),
    }


def test_drawn_image_channels_and_positions():
    s = _state()
    img = draw(s["robot_pose"], (0.70, 0.31), s["rects"], s["rect_mask"], s["circles"], s["circle_mask"],
               s["target_rect"], s["target_circle"], s["target_is_circle"], TD)
    assert img.shape == (1, 4, TD["size"], TD["size"])
    res = 2 * TD["half_extent"] / TD["size"]
    # areas match the footprints within the pixel discretization
    assert abs(img[0, 0].sum().item() * res ** 2 - 0.25) < 0.05
    assert abs(img[0, 1].sum().item() * res ** 2 - np.pi * 0.25 ** 2) < 0.05
    assert torch.equal(img[0, 3], img[0, 0])           # target is the chair
    # chair at (+1, +1) is in the top-right quadrant: small row, large column
    rows, cols = torch.nonzero(img[0, 0], as_tuple=True)
    r0, c0 = world_to_pixel(torch.tensor([1.0, 1.0]), TD)
    assert abs(rows.float().mean() - r0) < 1 and abs(cols.float().mean() - c0) < 1
    # heading marker at the robot's front (+x), body value elsewhere
    robot = img[0, 2]
    hr, hc = torch.nonzero(robot == TD["heading_value"], as_tuple=True)
    br, bc = torch.nonzero(robot == TD["robot_value"], as_tuple=True)
    assert hc.float().mean() > bc.float().mean()
    # turned 90 degrees, the marker moves up (smaller rows)
    s2 = _state()
    s2["robot_pose"][0, 2] = np.pi / 2
    img2 = draw(s2["robot_pose"], (0.70, 0.31), s2["rects"], s2["rect_mask"], s2["circles"], s2["circle_mask"],
                s2["target_rect"], s2["target_circle"], s2["target_is_circle"], TD)
    hr2, _ = torch.nonzero(img2[0, 2] == TD["heading_value"], as_tuple=True)
    br2, _ = torch.nonzero(img2[0, 2] == TD["robot_value"], as_tuple=True)
    assert hr2.float().mean() < br2.float().mean()


def test_masked_obstacles_are_not_drawn_and_circle_targets():
    s = _state()
    s["rect_mask"][:] = False
    s["target_is_circle"][:] = True
    s["target_circle"] = s["circles"][:, 0].clone()
    img = draw(s["robot_pose"], (0.70, 0.31), s["rects"], s["rect_mask"], s["circles"], s["circle_mask"],
               s["target_rect"], s["target_circle"], s["target_is_circle"], TD)
    assert img[0, 0].sum() == 0
    assert torch.equal(img[0, 3], img[0, 1])


def test_pixel_grid_spacing():
    pix = pixel_centers(TD)
    res = 2 * TD["half_extent"] / TD["size"]
    assert torch.allclose(pix[0, 1, 0] - pix[0, 0, 0], torch.tensor(res))
    assert pix[0, 0, 1] > pix[1, 0, 1]                 # row 0 is +y


def test_camera_feed_rate_delay_and_noise():
    cfg = copy.deepcopy(TD)
    cfg["realism"].update(delay_s=[0.15, 0.15], miss_probability=0.0, robot_pos_noise_m=0.0, robot_yaw_noise_rad=0.0,
                          obstacle_pos_noise_m=0.0)
    feed = CameraFeed(1, cfg)
    feed.reset(torch.tensor([0]))
    s = _state()
    captures = set()
    for k in range(60):                                 # 50 Hz control for 1.2 s
        t = torch.tensor([k * 0.02])
        s["robot_pose"][0, 0] = -1.0 + 0.02 * k         # robot moves right 1 m/s
        img = feed.observe(t, s)
        captures.add(int(feed.last_idx[0]))
        if k * 0.02 >= 0.3:
            ct = feed.delivered_capture_time(t).item()
            assert 0.15 - 1e-6 <= k * 0.02 - ct < 0.15 + 1 / 15 + 1e-6
            # the drawn robot is where it was at the capture time
            cols = torch.nonzero(img[0, 2])[:, 1].float().mean()
            _, c_exp = world_to_pixel(torch.tensor([-1.0 + ct, 0.0]), cfg)
            assert abs(cols - c_exp) < 1.5
    assert len(captures) == 18                          # 15 Hz over 1.2 s: frames 0..17


def test_camera_feed_misses():
    cfg = copy.deepcopy(TD)
    cfg["realism"].update(miss_probability=1.0)
    feed = CameraFeed(4, cfg, generator=torch.Generator().manual_seed(0))
    feed.reset(torch.arange(4))
    img = feed.observe(torch.zeros(4), _state(4))
    assert img[:, 0].sum() == 0 and img[:, 1].sum() == 0
    assert img[:, 3].sum() > 0                          # the target mask is kept

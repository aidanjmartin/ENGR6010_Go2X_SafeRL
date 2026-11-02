"""Navigation reward terms, action scaling, and the two image-based baselines."""
import numpy as np
import torch

from go2x.atrium.baselines import AStarController, ImageGeometry, StraightController, astar
from go2x.atrium.rewards import TERMS, front_point, normalize_action, reward_terms, scale_action, total
from go2x.atrium.topdown import draw
from go2x.config import load_config

NAV = load_config("nav")
TD = load_config("topdown")
W = NAV["reward"]


def _terms(**kw):
    B = 3
    base = dict(prev_dist=torch.tensor([1.0, 1.0, 1.0]), dist=torch.tensor([0.9, 1.0, 1.1]),
                success=torch.tensor([False, True, False]), collision=torch.tensor([False, False, True]),
                person_gap=torch.tensor([1.0, 0.3, 0.6]), fall=torch.tensor([False, False, True]),
                action=torch.zeros(B, 3), prev_action=torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [1.0, 1.0, 0]]))
    base.update(kw)
    return reward_terms(**base, w=W, person_near_distance=0.5)


def test_each_term():
    t = _terms()
    assert set(t) == set(TERMS)
    assert torch.allclose(t["progress"], W["progress"] * torch.tensor([0.1, 0.0, -0.1]))
    assert t["success"].tolist() == [0.0, W["success"], 0.0]
    assert t["collision"].tolist() == [0.0, 0.0, W["collision"]]
    assert torch.allclose(t["person_near"], torch.tensor([0.0, W["person_near"], 0.0]))
    assert t["fall"].tolist() == [0.0, 0.0, W["fall"]]
    assert torch.allclose(t["time"], torch.full((3,), W["time"]))
    assert torch.allclose(t["smoothness"], W["smoothness"] * torch.tensor([0.0, 1.0, 2.0]))
    assert torch.allclose(total(t), sum(t[k] for k in TERMS))


def test_signs():
    assert W["progress"] > 0 and W["success"] > 0
    for k in ("collision", "person_near", "fall", "time", "smoothness"):
        assert W[k] < 0, k


def test_action_scaling_roundtrip_and_limits():
    lim = NAV["action"]
    a = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [3.0, -3.0, 0.5]])
    cmd = scale_action(a, lim)
    assert torch.allclose(cmd[0], torch.tensor([0.0, -0.3, -1.0]))
    assert torch.allclose(cmd[1], torch.tensor([0.8, 0.3, 1.0]))
    assert torch.allclose(cmd[3, :2], torch.tensor([0.8, -0.3]))       # clipped
    assert torch.allclose(normalize_action(cmd[:3], lim), a[:3], atol=1e-6)


def test_front_point():
    p = front_point(torch.tensor([[1.0, 2.0, np.pi / 2]]), 0.35)
    assert torch.allclose(p, torch.tensor([[1.0, 2.35]]), atol=1e-6)


def _image(robot_pose, rects=(), circles=(), target=("rect", 0)):
    rects_t = torch.tensor([list(rects)], dtype=torch.float32).reshape(1, -1, 5)
    circ_t = torch.tensor([list(circles)], dtype=torch.float32).reshape(1, -1, 3)
    t_rect = rects_t[:, target[1]] if target[0] == "rect" else torch.zeros(1, 5)
    t_circ = circ_t[:, target[1]] if target[0] == "circle" else torch.zeros(1, 3)
    img = draw(torch.tensor([robot_pose], dtype=torch.float32), (0.70, 0.31), rects_t,
               torch.ones(rects_t.shape[:2], dtype=torch.bool), circ_t, torch.ones(circ_t.shape[:2], dtype=torch.bool),
               t_rect, t_circ, torch.tensor([target[0] == "circle"]), TD)
    return img[0].numpy()


def test_pose_from_image():
    geo = ImageGeometry(TD)
    for pose in ([0.3, -0.4, 0.0], [-0.5, 0.6, 2.0], [0.0, 0.0, -1.2]):
        est = geo.pose(_image(pose, rects=[(1.2, 1.2, 0.0, 0.25, 0.25)]))
        assert np.hypot(*(est[:2] - pose[:2])) < 0.05
        assert abs((est[2] - pose[2] + np.pi) % (2 * np.pi) - np.pi) < 0.25


def test_straight_turns_toward_target_and_stops():
    ctl = StraightController(NAV)
    # target chair straight to the left of the robot: turn left, no forward speed yet
    cmd = ctl.act(_image([0.0, -1.0, 0.0], rects=[(0.0, 1.0, 0.0, 0.25, 0.25)]))
    assert cmd[2] > 0.5 and cmd[0] < 0.05
    # facing the target from afar: forward
    cmd = ctl.act(_image([0.0, -1.0, np.pi / 2], rects=[(0.0, 1.0, 0.0, 0.25, 0.25)]))
    assert cmd[0] > 0.3 and abs(cmd[2]) < 0.2
    # facing it and close: stop
    cmd = ctl.act(_image([0.0, 0.25, np.pi / 2], rects=[(0.0, 1.0, 0.0, 0.25, 0.25)]))
    assert np.allclose(cmd, 0)


def test_astar_grid_detour_and_blocked():
    free = np.ones((10, 10), bool)
    free[5, 1:10] = False
    goal = np.zeros_like(free)
    goal[9, 9] = True
    path = astar(free, (0, 9), goal)
    assert path[0] == (0, 9) and path[-1] == (9, 9)
    assert all(free[p] for p in path) and any(p[1] == 0 for p in path)   # goes around through column 0
    free[5, 0] = False
    assert astar(free, (0, 9), goal) is None


def test_astar_controller_goes_around_table():
    ctl = AStarController(NAV)
    # table between the robot and the target chair
    img = _image([-1.0, 0.0, 0.0], rects=[(0.0, 0.0, np.pi / 2, 0.6, 0.375), (1.1, 0.0, 0.0, 0.25, 0.25)],
                 target=("rect", 1))
    cmd = ctl.act(img)
    assert ctl.path is not None and ctl.failed_plans == 0
    geo = ImageGeometry(TD)
    # the planned path never enters the table's footprint
    for x, y in ctl.path:
        assert not (abs(x) <= 0.375 and abs(y) <= 0.6)
    assert np.all(np.abs(cmd) <= np.array([0.8, 0.3, 1.0]) + 1e-9)
    assert geo.pose(img) is not None

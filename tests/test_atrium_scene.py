"""Scene sampler: placement rules, reachability, people paths, determinism, JSON round trip."""
import numpy as np

from go2x.atrium import geometry as G
from go2x.atrium.scene import Layout, Person, layout_from_dict, layout_to_dict, reachable, sample_layout
from go2x.config import load_config

CFG = load_config("atrium")
HALF = CFG["square"]["side"] / 2


def test_sampled_layouts_follow_the_rules():
    rng = np.random.default_rng(3)
    ob, rb = CFG["obstacles"], CFG["robot"]
    for _ in range(150):
        L, _ = sample_layout(rng, CFG)
        rects = L.static_rects()
        assert 0 <= len(L.chairs) <= ob["chair"]["count"][1]
        assert len(L.table) <= 1 and len(L.people) <= ob["person"]["count"][1]
        for i, r in enumerate(rects):
            assert G.rect_inside_square(r, HALF)
            for o in rects[i + 1:]:
                assert not G.rects_overlap(r, o, ob["gap"])
        rr = G.robot_rect(L.robot_start, rb["footprint"])
        assert all(not G.rects_overlap(rr, o, rb["start_clearance"]) for o in rects)
        assert all(not G.rect_circle_overlap(rr, c, rb["start_clearance"]) for c in L.person_circles())
        assert np.all(np.abs(L.robot_start[:2]) <= HALF - rb["start_margin"])
        assert reachable(L)[0]
        for p in L.people:
            traj = p.trajectory(0.02, 1000)
            assert np.all(np.abs(traj) <= HALF + 1e-9)          # people stay in the square
            if p.walking:
                d, e = p.b - p.a, traj - p.a
                assert np.allclose(d[0] * e[:, 1] - d[1] * e[:, 0], 0, atol=1e-6)  # on the straight path


def test_blocked_target_is_unreachable():
    # robot boxed in by four walls that touch each other; target chair outside, out of reach through the wall
    tables = np.array([[0.0, 0.5, 0.0], [0.0, -0.5, 0.0], [0.5, 0.0, np.pi / 2], [-0.5, 0.0, np.pi / 2]])
    cfg = dict(CFG)
    layout = Layout(np.array([0.0, 0.0, 0.0]), np.array([[1.3, 1.3, 0.0, 0]]), np.array([[0.0, -1.0, 0.0]]), [],
                    ("chair", 0), cfg)
    assert reachable(layout)[0]                       # one table does not block
    big = dict(cfg)
    big["obstacles"] = dict(cfg["obstacles"])
    big["obstacles"]["table"] = dict(cfg["obstacles"]["table"], footprint=[2.2, 0.3])
    walled = Layout(np.array([0.0, 0.0, 0.0]), np.array([[1.3, 1.3, 0.0, 0]]), tables, [], ("chair", 0), big)
    assert not walled.cfg is cfg
    # four 1.3 m walls around the robot
    walled.table_rects = lambda: [(x, y, yaw, 0.65, 0.15) for x, y, yaw in tables]
    ok, info = reachable(walled, big)
    assert not ok and info["reason"] == "no path"


def test_walking_person_speed_and_pauses():
    p = Person(np.array([-1.0, 0.0]), True, np.array([-1.0, 0.0]), np.array([1.0, 0.0]), [(1.0, 1.0), (1.0, 0.0), (5.0, 0.5)])
    traj = p.trajectory(0.01, 400)
    assert np.isclose(traj[100, 0], 0.0, atol=0.02)    # 1 m at 1 m/s
    assert np.allclose(traj[110:200, 0], traj[110, 0], atol=1e-9)  # paused
    assert np.isclose(traj[300, 0] - traj[200, 0], 0.5, atol=0.02)
    back = p.trajectory(0.01, 1000)
    assert back[:, 0].max() <= 1.0 + 1e-9                # turns around at the end


def test_determinism_and_roundtrip():
    a, _ = sample_layout(np.random.default_rng(7), CFG)
    b, _ = sample_layout(np.random.default_rng(7), CFG)
    assert np.allclose(a.robot_start, b.robot_start) and a.target == b.target
    c = layout_from_dict(layout_to_dict(a), CFG)
    assert np.allclose(c.robot_start, a.robot_start) and np.allclose(c.chairs, a.chairs)
    assert [p.walking for p in c.people] == [p.walking for p in a.people]
    for p, q in zip(a.people, c.people):
        assert np.allclose(p.trajectory(0.02, 50), q.trajectory(0.02, 50))


def test_target_kind_filter():
    rng = np.random.default_rng(11)
    for kind in ("chair", "table", "person"):
        L, _ = sample_layout(rng, CFG, target_kind=kind)
        assert L.target[0] == kind

"""Batched atrium state: slots, parking, people motion, metrics and simulator part poses."""
import numpy as np
import torch

from go2x.atrium.scene import layout_to_dict, sample_layout
from go2x.atrium.state import AtriumBatch, object_parts
from go2x.config import load_config

CFG = load_config("atrium")


def test_reset_fills_slots_and_parks_the_rest():
    ab = AtriumBatch(8, seed=1)
    ab.reset(torch.arange(8))
    half = CFG["square"]["side"] / 2
    for e in range(8):
        L = ab.layouts[e]
        assert int(ab.rect_mask[e, : ab.n_chair_slots].sum()) == len(L.chairs)
        assert bool(ab.rect_mask[e, ab.table_slot]) == (len(L.table) == 1)
        assert int(ab.circle_mask[e].sum()) == len(L.people)
        used = ab.rect_mask[e].tolist() + ab.circle_mask[e].tolist()
        for s, u in enumerate(used):
            xy = ab.slot_pose[e, s, :2]
            if u:
                assert (xy.abs() <= half + 1e-6).all()
            else:
                assert torch.allclose(xy, ab.park_xy[s]) and xy[0] > 3.0      # behind the camera
        # chair colors map to their own slots
        for (x, y, yaw, c), name in zip(L.chairs, L.color_names()):
            s = [i for i in range(ab.n_chair_slots) if ab.rect_mask[e, i] and torch.allclose(ab.rects[e, i, :2], torch.tensor([x, y], dtype=torch.float32))][0]
            assert s // ab.cpc == int(c) and ab.colors[int(c)] == name
        kind, idx = L.target
        assert bool(ab.target_is_circle[e]) == (kind == "person")


def test_parked_slots_do_not_collide():
    ab = AtriumBatch(4, seed=2)
    ab.reset(torch.arange(4))
    # a robot standing on a park position touches nothing: parked slots are masked out
    pose = torch.cat([ab.park_xy[:1].repeat(4, 1), torch.zeros(4, 1)], -1)
    m = ab.metrics(pose, torch.zeros(4))
    assert not m["collision_footprint"].any()
    assert m["out_of_square"].all()


def test_people_follow_trajectories_and_target_moves():
    rng = np.random.default_rng(5)
    while True:
        L, _ = sample_layout(rng, CFG, target_kind="person")
        if L.people[L.target[1]].walking:
            break
    ab = AtriumBatch(1, scenarios=[layout_to_dict(L)])
    ab.reset([0])
    assert ab.scenario_id[0] == 0
    traj = L.people[L.target[1]].trajectory(ab.dt, ab.T)
    for k in (0, 100, 400):
        ab.move_people(torch.tensor([k]))
        assert np.allclose(ab.circles[0, L.target[1], :2].numpy(), traj[k], atol=1e-5)
        assert torch.allclose(ab.target_circle[0], ab.circles[0, L.target[1]])


def test_metrics_success_and_distance():
    ab = AtriumBatch(1, seed=0)
    ab.reset([0])
    # place the target as a chair at the origin, facing it from 0.3 m at the front
    ab.target_is_circle[:] = False
    ab.target_rect[0] = torch.tensor([0.0, 0.0, 0.0, 0.25, 0.25])
    ab.rect_mask[:] = False
    ab.circle_mask[:] = False
    front_offset = CFG["robot"]["front_offset"]
    pose = torch.tensor([[-(0.25 + 0.3 + front_offset), 0.0, 0.0]])
    m = ab.metrics(pose, torch.tensor([0.05]))
    assert torch.allclose(m["target_distance"], torch.tensor([0.3]), atol=1e-5)
    assert m["at_target"].item()
    assert not ab.metrics(pose, torch.tensor([0.2]))["at_target"].item()      # too fast
    far = pose.clone()
    far[0, 0] -= 0.3
    assert not ab.metrics(far, torch.tensor([0.0]))["at_target"].item()      # 0.6 m away


def test_part_poses_rotate_offsets():
    ab = AtriumBatch(1, seed=0)
    ab.reset([0])
    parts = ab.parts
    j = [i for i, p in enumerate(parts) if p["name"] == "chair_red_0_back"][0]
    s = parts[j]["slot"]
    ab.slot_pose[0, s] = torch.tensor([1.0, 0.5, np.pi / 2])
    P = ab.part_poses()[0, j]
    ox = parts[j]["offset"][0]                                  # negative: the back is behind the seat center
    assert torch.allclose(P[:2], torch.tensor([1.0, 0.5 + ox]), atol=1e-6)
    assert torch.allclose(P[3:], torch.tensor([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)], dtype=torch.float32), atol=1e-6)
    names = [p["name"] for p in object_parts(CFG)]
    assert len(names) == len(set(names))
    assert len(ab.person_part_ids()) == CFG["obstacles"]["person"]["count"][1]

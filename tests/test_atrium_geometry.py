"""Collision checks: numpy and torch footprint tests on hand-built cases and against each other."""
import numpy as np
import torch

from go2x.atrium import batch as Bt
from go2x.atrium import geometry as G
from go2x.atrium.building import polygon_distance


def test_point_rect_distance_cases():
    r = (0.0, 0.0, 0.0, 1.0, 0.5)
    d = G.point_rect_distance([[0, 0], [2, 0], [0, 1.5], [2, 1.5]], r)
    assert np.allclose(d, [0, 1, 1, np.sqrt(2)])
    # rotated 90 degrees: the long side now runs along y
    r90 = (0.0, 0.0, np.pi / 2, 1.0, 0.5)
    assert np.allclose(G.point_rect_distance([[0, 1.5], [1.5, 0]], r90), [0.5, 1.0])


def test_rect_overlap_cases():
    a = (0, 0, 0, 0.5, 0.5)
    assert G.rects_overlap(a, (0.9, 0, 0, 0.5, 0.5))
    assert not G.rects_overlap(a, (1.1, 0, 0, 0.5, 0.5))
    assert G.rects_overlap(a, (1.1, 0, 0, 0.5, 0.5), gap=0.2)          # within the gap
    # a 45-degree square whose corner reaches across x = 0.5 by 0.05
    reach = 0.5 * np.sqrt(2)
    assert G.rects_overlap(a, (0.5 + reach - 0.05, 0, np.pi / 4, 0.5, 0.5))
    assert not G.rects_overlap(a, (0.5 + reach + 0.05, 0, np.pi / 4, 0.5, 0.5))
    # thin cross: no corner inside the other rectangle, still overlapping
    assert G.rects_overlap((0, 0, 0, 1.0, 0.05), (0, 0, np.pi / 2, 1.0, 0.05))


def test_torch_matches_numpy_random():
    rng = np.random.default_rng(1)
    A = np.column_stack([rng.uniform(-1, 1, (400, 2)), rng.uniform(-np.pi, np.pi, 400), rng.uniform(0.05, 0.6, (400, 2))])
    Bb = np.column_stack([rng.uniform(-1, 1, (400, 2)), rng.uniform(-np.pi, np.pi, 400), rng.uniform(0.05, 0.6, (400, 2))])
    np_overlap = np.array([G.rects_overlap(a, b) for a, b in zip(A, Bb)])
    t_overlap = Bt.rects_overlap(torch.tensor(A), torch.tensor(Bb)).numpy()
    assert np.array_equal(np_overlap, t_overlap)
    # gap against an independent polygon distance
    d_t = Bt.rect_rect_distance(torch.tensor(A), torch.tensor(Bb)).numpy()
    d_ref = np.array([polygon_distance(G.rect_corners(a), G.rect_corners(b)) for a, b in zip(A, Bb)])
    assert np.allclose(d_t, d_ref, atol=1e-9)
    assert np.mean(np_overlap) > 0.1 and np.mean(~np_overlap) > 0.1  # both cases exercised


def test_robot_obstacle_overlap_masks():
    robot = torch.tensor([[0.0, 0.0, 0.0, 0.35, 0.155]])
    rects = torch.tensor([[[0.6, 0.0, 0.0, 0.3, 0.3], [5.0, 5.0, 0.0, 0.3, 0.3]]])
    circles = torch.tensor([[[0.0, 0.5, 0.25], [0.0, -0.39, 0.25]]])
    hit, hr, hc = Bt.robot_obstacle_overlap(robot, rects, torch.tensor([[False, True]]), circles,
                                            torch.tensor([[True, True]]))
    # the overlapping chair is masked (parked); circle 0 is 0.345 from the body edge minus 0.25 -> clear; circle 1 overlaps
    assert hr.tolist() == [[False, False]]
    assert hc.tolist() == [[False, True]]
    assert hit.tolist() == [True]
    gap = Bt.robot_circle_gap(robot, circles[:, :1], torch.tensor([[True]]))
    assert torch.allclose(gap, torch.tensor([0.5 - 0.155 - 0.25]))

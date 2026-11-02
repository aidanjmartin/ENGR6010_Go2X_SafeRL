"""Prompt generator, floor projection of a detection, and the target mask."""
import numpy as np

from go2x.atrium.camera import Camera
from go2x.grounding import box_floor_point, disk_mask, footprint_distance, iou, make_prompts
from go2x.config import load_config

OBJS = [
    {"id": "chair_0", "class": "chair", "color": "red", "footprint_rect": [1.0, 1.0, 0.0, 0.25, 0.25]},
    {"id": "chair_1", "class": "chair", "color": "red", "footprint_rect": [-1.0, 1.0, 0.0, 0.25, 0.25]},
    {"id": "chair_2", "class": "chair", "color": "blue", "footprint_rect": [0.0, -1.0, 0.0, 0.25, 0.25]},
    {"id": "table_0", "class": "table", "color": None, "footprint_rect": [0.0, 0.5, 0.0, 0.6, 0.375]},
    {"id": "person_0", "class": "person", "color": None, "footprint_circle": [-1.0, -1.0, 0.25]},
]


def test_prompts_skip_ambiguous():
    prompts, skipped = make_prompts(OBJS)
    by_prompt = {p["prompt"]: p for p in prompts}
    assert by_prompt["go to the blue chair"]["target_id"] == "chair_2"
    assert by_prompt["go to the table"]["target_id"] == "table_0"
    assert by_prompt["go to the person"]["target_id"] == "person_0"
    assert "go to the red chair" not in by_prompt and "go to the chair" not in by_prompt
    sk = {s["prompt"]: s["matches"] for s in skipped}
    assert sk == {"go to the red chair": 2, "go to the chair": 3}
    assert by_prompt["go to the blue chair"]["query"] == "a blue chair"


def test_single_chair_and_missing_classes():
    prompts, skipped = make_prompts([OBJS[2]])
    assert {p["prompt"] for p in prompts} == {"go to the blue chair", "go to the chair"}
    assert {s["prompt"]: s["matches"] for s in skipped} == {"go to the table": 0, "go to the person": 0}
    assert make_prompts([]) == ([], [{"prompt": t, "query": q, "template": k, "matches": 0} for k, t, q in
                                     [("chair", "go to the chair", "a chair"), ("table", "go to the table", "a table"),
                                      ("person", "go to the person", "a person")]])


def test_box_bottom_center_projects_to_floor():
    cam = Camera.from_config()
    H = cam.floor_homography()
    floor = np.array([0.4, -0.7, 0.0])
    uv, _ = cam.project(floor)
    u, v = uv[0]
    box = (u - 20, v - 60, u + 20, v)          # a box whose bottom center sits on that floor point
    assert np.allclose(box_floor_point(box, H), floor[:2], atol=1e-9)


def test_footprint_distance_and_mask_iou():
    assert footprint_distance([1.0, 1.0], OBJS[0]) == 0.0
    assert np.isclose(footprint_distance([1.55, 1.0], OBJS[0]), 0.3)
    assert np.isclose(footprint_distance([-1.0, -0.5], OBJS[4]), 0.25)
    td = load_config("topdown")
    m1 = disk_mask([(0.0, 0.0)], [0.5], td)
    m2 = disk_mask([(0.0, 0.0)], [0.5], td)
    m3 = disk_mask([(1.0, 0.0)], [0.4], td)
    assert iou(m1, m2) == 1.0 and iou(m1, m3) == 0.0
    res = 2 * td["half_extent"] / td["size"]
    assert abs(m1.sum() * res ** 2 - np.pi * 0.25) < 0.03

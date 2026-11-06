"""Overhead camera pipeline on synthetic frames (go2x/vision): printed tags drawn with the camera model, then
detected; the recovered camera pose and robot pose are compared with the truth."""
import copy
import math

import numpy as np

from go2x.atrium.camera import Camera
from go2x.config import load_config
from go2x.vision.geometry import RealCamera, TagDetector, tag_center_px, tag_pose_on_plane
from go2x.vision.synth import overlay_tag

T = load_config("tags")


def _frame(cam, robot=None, h=0.38):
    img = np.full((cam.height, cam.width, 3), 150, np.uint8)
    for tid, (x, y) in zip(T["corners"]["ids"], T["corners"]["xy"]):
        overlay_tag(img, cam, tid, x, y, 0.0, T["tag_size_m"], 0.0)
    if robot is not None:
        overlay_tag(img, cam, T["robot"]["id"], *robot, T["tag_size_m"], h)
    return img


def test_extrinsics_from_corner_tags():
    truth = RealCamera.from_atrium_camera(Camera.from_config(load_config("camera")), scale=2.0)
    found = TagDetector().detect(_frame(truth))
    assert set(T["corners"]["ids"]) <= set(found)
    est = copy.deepcopy(truth)
    est.R_wc, est.C = None, None
    uv = [tag_center_px(found[i]) for i in T["corners"]["ids"]]
    X = [[x, y, 0.0] for x, y in T["corners"]["xy"]]
    mean, mx = est.solve_extrinsics(uv, X)
    assert mx < 1.0
    assert np.linalg.norm(est.C - truth.C) < 0.05                       # camera position within 5 cm
    ang = math.degrees(math.acos(np.clip((np.trace(est.R_wc.T @ truth.R_wc) - 1) / 2, -1, 1)))
    assert ang < 1.0


def test_robot_tag_pose():
    cam = RealCamera.from_atrium_camera(Camera.from_config(load_config("camera")), scale=2.0)
    det = TagDetector()
    for x, y, yaw in ((0.3, -0.4, 0.7), (-1.0, 0.9, -2.5), (1.1, 1.0, 3.0)):
        found = det.detect(_frame(cam, (x, y, yaw), h=0.38))
        pose, edge = tag_pose_on_plane(cam, found[T["robot"]["id"]], 0.38)
        assert np.hypot(pose[0] - x, pose[1] - y) < 0.02
        assert abs((pose[2] - yaw + math.pi) % (2 * math.pi) - math.pi) < math.radians(3)
        assert abs(edge - T["tag_size_m"]) < 0.01


def test_wrong_tag_height_shows_in_position():
    """A wrong height for the robot tag shifts its position away from the camera (why tags.yaml says MEASURE)."""
    cam = RealCamera.from_atrium_camera(Camera.from_config(load_config("camera")), scale=2.0)
    found = TagDetector().detect(_frame(cam, (0.0, 0.0, 0.0), h=0.38))
    good, _ = tag_pose_on_plane(cam, found[T["robot"]["id"]], 0.38)
    bad, _ = tag_pose_on_plane(cam, found[T["robot"]["id"]], 0.20)
    assert np.hypot(*good[:2]) < 0.02 and np.hypot(*bad[:2]) > 0.05


def _perc(target, owl_objs=(), tag_objs=(), hint=None):
    from go2x.vision.perception import Perception
    cam = RealCamera.from_atrium_camera(Camera.from_config(load_config("camera")), scale=2.0)
    p = Perception(cam, None, target, "both")
    p.owl_objects, p.tag_obstacles, p.target_hint = list(owl_objs), list(tag_objs), hint
    return p


def _o(kind, x, y, score=0.5, color=None, source="owl"):
    return {"kind": kind, "color": color, "x": x, "y": y, "yaw": 0.0, "source": source, "score": score}


def test_layout_from_detections_and_target_choice():
    objs = [_o("chair", 0.5, 0.5, 0.4), _o("chair", -0.5, 0.2, 0.6), _o("table", 1.0, -1.0), _o("person", -1.0, -1.0)]
    scen, _ = _perc("chair", objs).to_scenario(np.zeros(3))
    assert len(scen["chairs"]) == 2 and len(scen["table"]) == 1 and len(scen["people"]) == 1
    k, i = scen["target"]
    assert k == "chair" and np.allclose(scen["chairs"][i][:2], [-0.5, 0.2])          # best score
    objs[0]["color"] = "red"
    scen, _ = _perc("go to the red chair", objs).to_scenario(np.zeros(3))
    k, i = scen["target"]
    assert np.allclose(scen["chairs"][i][:2], [0.5, 0.5])                             # the chair seen as red
    assert _perc("blue chair", objs).to_scenario(np.zeros(3))[0] is None
    scen, note = _perc("table", []).to_scenario(np.zeros(3))
    assert scen is None and "not found" in note


def test_obstacle_tag_replaces_nearby_detection_and_can_be_the_target():
    tag = _o("chair", 0.52, 0.48, 1.0, color="red", source="tag 20")
    scen, _ = _perc("tag:20", [_o("chair", 0.5, 0.5)], [tag]).to_scenario(np.zeros(3))
    assert len(scen["chairs"]) == 1 and np.allclose(scen["chairs"][0][:2], [0.52, 0.48])
    assert scen["target"] == ["chair", 0]
    scen, _ = _perc("red chair", [_o("chair", -1, 0)], [tag]).to_scenario(np.zeros(3))
    assert np.allclose(scen["chairs"][scen["target"][1]][:2], [0.52, 0.48])


def test_extra_objects_are_dropped_nearest_kept():
    people = [_o("person", x, 0.0) for x in (1.2, 0.4, -0.8)]
    scen, note = _perc("person", people).to_scenario(np.zeros(3))
    assert len(scen["people"]) == 2 and "person dropped" in note
    assert sorted(abs(p["start"][0]) for p in scen["people"]) == [0.4, 0.8]


def test_detections_persist_through_flicker():
    p = _perc("table")
    seen = p._update_tracks([_o("chair", 0.5, 0.5), _o("table", 1.0, 1.0)], 0.0)
    assert len(seen) == 2
    seen = p._update_tracks([_o("table", 1.05, 1.0)], 3.0)                  # chair missed once: kept
    assert sorted(o["kind"] for o in seen) == ["chair", "table"]
    assert len([o for o in seen if o["kind"] == "table"]) == 1                # updated, not duplicated
    seen = p._update_tracks([_o("table", 1.0, 1.0)], 7.0)                   # chair unseen for 7 s: forgotten
    assert [o["kind"] for o in seen] == ["table"]


def test_best_color_wins_when_one_chair_is_seen_twice():
    p = _perc("red chair")
    seen = p._update_tracks([_o("chair", 0.5, 0.5, 0.25, color="blue"), _o("chair", 0.52, 0.5, 0.34, color="red"),
                             _o("chair", 0.5, 0.49, 0.22)], 0.0)
    assert len(seen) == 1 and seen[0]["color"] == "red"

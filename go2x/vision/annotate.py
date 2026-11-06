"""Draw what the camera pipeline believes onto a camera frame: the square, tags, robot pose and obstacles."""
import math

import cv2
import numpy as np

COLORS = {"chair": (230, 60, 60), "table": (150, 90, 40), "person": (60, 200, 60), "target": (255, 215, 0),
          "robot": (40, 110, 255), "square": (255, 255, 255)}


def _poly(img, cam, pts, color, thick=2):
    uv = cam.project(np.asarray(pts, float)).astype(np.int32)
    cv2.polylines(img, [uv.reshape(-1, 1, 2)], True, color, thick, cv2.LINE_AA)


def rect_pts(x, y, yaw, L, W, z=0.0):
    c, s = math.cos(yaw), math.sin(yaw)
    return [[x + c * a - s * b, y + s * a + c * b, z] for a, b in ((L / 2, W / 2), (L / 2, -W / 2), (-L / 2, -W / 2), (-L / 2, W / 2))]


def draw(frame, cam, half, tags=None, robot_pose=None, footprint=(0.70, 0.31), scenario=None, atrium=None, text=None):
    img = frame.copy()
    _poly(img, cam, [[-half, -half, 0], [half, -half, 0], [half, half, 0], [-half, half, 0]], COLORS["square"], 2)
    o = cam.project(np.array([[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0]], float)).astype(int)
    cv2.arrowedLine(img, tuple(o[0]), tuple(o[1]), (255, 0, 0), 2, tipLength=0.2)
    cv2.arrowedLine(img, tuple(o[0]), tuple(o[2]), (0, 255, 0), 2, tipLength=0.2)
    for tid, c in (tags or {}).items():
        cv2.polylines(img, [c.astype(np.int32).reshape(-1, 1, 2)], True, (255, 0, 255), 2)
        cv2.putText(img, str(tid), tuple(c[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    if scenario is not None and atrium is not None:
        ob = atrium["obstacles"]
        cw, cd = ob["chair"]["footprint"]
        tw, td = ob["table"]["footprint"]
        kind, idx = scenario["target"]
        for i, (x, y, yaw, _) in enumerate(scenario["chairs"]):
            _poly(img, cam, rect_pts(x, y, yaw, cw, cd), COLORS["target"] if (kind, idx) == ("chair", i) else COLORS["chair"])
        for x, y, yaw in scenario["table"]:
            _poly(img, cam, rect_pts(x, y, yaw, tw, td), COLORS["target"] if kind == "table" else COLORS["table"])
        r = ob["person"]["radius"]
        for i, p in enumerate(scenario["people"]):
            x, y = p["start"]
            circ = [[x + r * math.cos(a), y + r * math.sin(a), 0] for a in np.linspace(0, 2 * math.pi, 16, endpoint=False)]
            _poly(img, cam, circ, COLORS["target"] if (kind, idx) == ("person", i) else COLORS["person"])
    if robot_pose is not None:
        x, y, yaw = robot_pose
        _poly(img, cam, rect_pts(x, y, yaw, *footprint), COLORS["robot"], 3)
        a = cam.project(np.array([[x, y, 0], [x + 0.5 * math.cos(yaw), y + 0.5 * math.sin(yaw), 0]])).astype(int)
        cv2.arrowedLine(img, tuple(a[0]), tuple(a[1]), COLORS["robot"], 3, tipLength=0.25)
    if text:
        cv2.rectangle(img, (0, 0), (img.shape[1], 34), (255, 255, 255), -1)
        cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    return img

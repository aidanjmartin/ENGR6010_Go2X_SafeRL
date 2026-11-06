"""Printed tags drawn into rendered frames, for testing the camera pipeline without a camera.

MuJoCo renders the scene from the simulated overhead camera; each tag (black ArUco pattern on a white paper
margin) is warped into the frame at its projected corners with the same camera model, as a printed tag would
appear. Nothing hides the tags (no occlusion), which a real floor does not guarantee.
"""
import cv2
import numpy as np

from go2x.vision.geometry import ARUCO_DICTS, tag_world_corners

PAPER = 1.4          # paper edge / black square edge


def tag_image(tag_id, dictionary="DICT_4X4_50", px=240):
    """Tag on paper: the black square spans the middle 1/PAPER of the image."""
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[dictionary])
    inner = cv2.aruco.generateImageMarker(d, tag_id, px)
    pad = int(round(px * (PAPER - 1) / 2))
    return cv2.copyMakeBorder(inner, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)


def overlay_tag(frame, cam, tag_id, x, y, yaw, size, z, dictionary="DICT_4X4_50", cache={}):  # noqa: B006
    key = (tag_id, dictionary)
    if key not in cache:
        cache[key] = tag_image(tag_id, dictionary)
    img = cache[key]
    world = tag_world_corners(x, y, yaw, size * PAPER, z)          # paper corners
    uv = cam.project(world).astype(np.float32)
    n = img.shape[0]
    src = np.float32([[0, 0], [n - 1, 0], [n - 1, n - 1], [0, n - 1]])
    M = cv2.getPerspectiveTransform(src, uv)
    h, w = frame.shape[:2]
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    mask = cv2.warpPerspective(np.full_like(img, 255), M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    a = (mask.astype(np.float32) / 255.0)[..., None]
    out = frame.astype(np.float32) * (1 - a) + warped[..., None].astype(np.float32) * a
    frame[:] = out.astype(np.uint8)
    return frame

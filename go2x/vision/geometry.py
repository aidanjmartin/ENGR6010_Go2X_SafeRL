"""Calibrated camera (OpenCV model, with lens distortion) and the tag geometry on top of it.

Conventions as go2x/atrium/camera.py: optical frame x right, y down, z forward; R_wc holds the camera axes as
columns in the square frame, C is the camera center; a square-frame point X maps to X_c = R_wc^T (X - C).
"""
import math
from dataclasses import dataclass

import cv2
import numpy as np
import yaml

ARUCO_DICTS = {"DICT_4X4_50": cv2.aruco.DICT_4X4_50, "DICT_5X5_100": cv2.aruco.DICT_5X5_100}


@dataclass
class RealCamera:
    K: np.ndarray
    dist: np.ndarray
    width: int
    height: int
    R_wc: np.ndarray = None
    C: np.ndarray = None

    # -- files -----------------------------------------------------------------------
    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = yaml.safe_load(f)
        ex = d.get("extrinsics") or {}
        return cls(np.array(d["intrinsics"]["K"], float), np.array(d["intrinsics"]["dist"], float),
                   int(d["intrinsics"]["width"]), int(d["intrinsics"]["height"]),
                   np.array(ex["R_wc"], float) if ex else None, np.array(ex["C"], float) if ex else None)

    def to_dict(self):
        d = {"intrinsics": {"K": self.K.tolist(), "dist": np.asarray(self.dist).ravel().tolist(),
                            "width": self.width, "height": self.height}}
        if self.R_wc is not None:
            d["extrinsics"] = {"R_wc": self.R_wc.tolist(), "C": self.C.tolist()}
        return d

    @classmethod
    def from_atrium_camera(cls, cam, scale=1.0):
        """The simulated camera (go2x.atrium.camera.Camera) at `scale` times its resolution, no distortion."""
        K = cam.K.copy()
        K[:2] *= scale
        return cls(K, np.zeros(5), int(cam.width * scale), int(cam.height * scale), cam.R_wc.copy(), cam.C.copy())

    # -- geometry --------------------------------------------------------------------
    def undistort(self, uv):
        """Pixels (N, 2) -> normalized image coordinates (N, 2), lens distortion removed."""
        pts = np.asarray(uv, np.float64).reshape(-1, 1, 2)
        return cv2.undistortPoints(pts, self.K, self.dist).reshape(-1, 2)

    def pixel_to_plane(self, uv, z):
        """Pixels (N, 2) -> points (N, 3) on the horizontal plane at height z (square frame)."""
        n = self.undistort(uv)
        rays = np.column_stack([n, np.ones(len(n))]) @ self.R_wc.T
        s = (z - self.C[2]) / rays[:, 2]
        return self.C + s[:, None] * rays

    def project(self, X):
        """Square-frame points (N, 3) -> pixels (N, 2), with distortion."""
        R_cw = self.R_wc.T
        rvec, _ = cv2.Rodrigues(R_cw)
        tvec = -R_cw @ self.C
        uv, _ = cv2.projectPoints(np.asarray(X, np.float64).reshape(-1, 1, 3), rvec, tvec, self.K, self.dist)
        return uv.reshape(-1, 2)

    def solve_extrinsics(self, uv, X):
        """Camera pose from pixel / square-frame correspondences (N >= 4, coplanar allowed). Sets R_wc and C and
        returns the reprojection error in pixels (mean, max)."""
        X = np.asarray(X, np.float64).reshape(-1, 3)
        uv = np.asarray(uv, np.float64).reshape(-1, 2)
        flag = cv2.SOLVEPNP_IPPE if np.allclose(X[:, 2], X[0, 2]) else cv2.SOLVEPNP_ITERATIVE
        ok, rvec, tvec = cv2.solvePnP(X, uv, self.K, self.dist, flags=flag)
        if not ok:
            raise RuntimeError("solvePnP failed")
        ok, rvec, tvec = cv2.solvePnP(X, uv, self.K, self.dist, rvec, tvec, useExtrinsicGuess=True,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        R_cw, _ = cv2.Rodrigues(rvec)
        self.R_wc = R_cw.T
        self.C = (-self.R_wc @ tvec).ravel()
        err = np.linalg.norm(self.project(X) - uv, axis=1)
        return float(err.mean()), float(err.max())


# -- tags ------------------------------------------------------------------------------
class TagDetector:
    """ArUco detection in two passes. The second pass runs on the image after a 5 x 5 grey-level closing, which
    bridges thin dark gaps in a tag's white margin (a cut or a seam that joins the black border to a dark
    background and so hides the border's outline from the detector); it only adds tags the first pass missed.
    2026-11-07, Brio 100 at 7.4 m: the robot tag, with a notch cut in its margin, went from 0 to 40 of 40 frames;
    on tags both passes found, the closing moved centers by 0.2 px on average (corners by up to 2.4 px); 3 ms."""

    def __init__(self, dictionary="DICT_4X4_50", fill_gaps=True):
        self.dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[dictionary])
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.det = cv2.aruco.ArucoDetector(self.dict, params)
        self.fill_gaps = fill_gaps
        self._kernel = np.ones((5, 5), np.uint8)

    def _detect(self, img):
        corners, ids, _ = self.det.detectMarkers(img)
        if ids is None:
            return {}
        return {int(i): c.reshape(4, 2) for i, c in zip(ids.ravel(), corners)}

    def detect(self, image_rgb_or_gray):
        """{id: corners (4, 2) in pixels}, corners in the tag's own order: top-left, top-right, bottom-right,
        bottom-left of the printed tag (top = the side the FRONT arrow points to)."""
        img = image_rgb_or_gray if image_rgb_or_gray.ndim == 2 else cv2.cvtColor(image_rgb_or_gray, cv2.COLOR_RGB2GRAY)
        found = self._detect(img)
        if self.fill_gaps:
            for i, c in self._detect(cv2.morphologyEx(img, cv2.MORPH_CLOSE, self._kernel)).items():
                found.setdefault(i, c)
        return found


def tag_center_px(corners):
    """Center of a tag in the image: intersection of its diagonals (exact under perspective)."""
    p0, p1, p2, p3 = np.asarray(corners, float)
    d1, d2 = p2 - p0, p3 - p1
    A = np.array([d1, -d2]).T
    s = np.linalg.solve(A, p1 - p0)[0]
    return p0 + s * d1


def tag_pose_on_plane(cam, corners, z):
    """A flat tag at height z: (x, y, yaw) of its center and its 'up' direction (top edge, FRONT arrow) in the
    square frame, plus the tag's apparent edge length in meters (a sanity check against the printed size)."""
    P = cam.pixel_to_plane(np.vstack([corners, tag_center_px(corners)[None]]), z)
    tl, tr, br, bl, c = P
    up = (tl + tr) / 2 - (bl + br) / 2
    yaw = math.atan2(up[1], up[0])
    edge = float(np.mean([np.linalg.norm(tr - tl), np.linalg.norm(br - tr), np.linalg.norm(bl - br), np.linalg.norm(tl - bl)]))
    return np.array([c[0], c[1], yaw]), edge


def tag_world_corners(x, y, yaw, size, z=0.0):
    """Square-frame corners (4, 3) of a flat tag centered at (x, y, z) whose top points along yaw, in the
    detector's order (top-left, top-right, bottom-right, bottom-left), seen from above."""
    h = size / 2
    local = np.array([[h, h], [h, -h], [-h, -h], [-h, h]])     # (forward, left): top-left = front-left
    c, s = math.cos(yaw), math.sin(yaw)
    pts = np.array([[x + c * f - s * l, y + s * f + c * l, z] for f, l in local])
    return pts

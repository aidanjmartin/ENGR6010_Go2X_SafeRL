"""Fixed pinhole camera over the atrium: projection, floor homography and coverage check.

Camera frame is the optical (ROS) convention: x right, y down, z forward. R_wc holds the camera axes as
columns in world coordinates, so a world point X maps to X_c = R_wc^T (X - C). Pixel coordinates are
continuous, with (0, 0) at the top-left corner of the top-left pixel.
"""
from dataclasses import dataclass

import numpy as np

from go2x.config import load_config


@dataclass
class Camera:
    K: np.ndarray        # 3 x 3 intrinsics
    R_wc: np.ndarray     # 3 x 3, camera axes in world
    C: np.ndarray        # camera center in world
    width: int
    height: int

    @classmethod
    def from_config(cls, cfg=None):
        cfg = cfg or load_config("camera")
        w, h = cfg["intrinsics"]["width"], cfg["intrinsics"]["height"]
        f = (w / 2) / np.tan(np.radians(cfg["intrinsics"]["hfov_deg"]) / 2)
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
        C = np.asarray(cfg["pose"]["position"], float)
        fwd = np.asarray(cfg["pose"]["look_at"], float) - C
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, np.asarray(cfg["pose"]["up"], float))
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        return cls(K, np.stack([right, down, fwd], axis=1), C, w, h)

    def to_camera(self, X):
        return (np.atleast_2d(X) - self.C) @ self.R_wc

    def project(self, X):
        """World points (N, 3) to pixels (N, 2) and depth (N,)."""
        Xc = self.to_camera(X)
        uvw = Xc @ self.K.T
        return uvw[:, :2] / uvw[:, 2:3], Xc[:, 2]

    def floor_homography(self):
        """H with [u, v, 1]^T ~ H [x, y, 1]^T for floor points (z = 0)."""
        R_cw = self.R_wc.T
        t = -R_cw @ self.C
        H = self.K @ np.column_stack([R_cw[:, 0], R_cw[:, 1], t])
        return H / H[2, 2]

    def pixel_to_floor(self, uv):
        Hi = np.linalg.inv(self.floor_homography())
        p = np.column_stack([np.atleast_2d(uv), np.ones(len(np.atleast_2d(uv)))]) @ Hi.T
        return p[:, :2] / p[:, 2:3]

    def quat_wxyz_ros(self):
        """Orientation of the optical frame in the world, (w, x, y, z), for Isaac Lab convention="ros"."""
        return rot_to_quat(self.R_wc)

    def in_image(self, uv, depth, edge=0.0):
        return ((depth > 0) & (uv[:, 0] >= edge) & (uv[:, 0] <= self.width - edge)
                & (uv[:, 1] >= edge) & (uv[:, 1] <= self.height - edge))


def rot_to_quat(R):
    """Rotation matrix to unit quaternion (w, x, y, z), w >= 0."""
    m = R
    tr = np.trace(m)
    if tr > 0:
        s = 2 * np.sqrt(tr + 1)
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * np.sqrt(1 + m[i, i] - m[j, j] - m[k, k])
        q = np.zeros(4)
        q[0] = (m[k, j] - m[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (m[j, i] + m[i, j]) / s
        q[1 + k] = (m[k, i] + m[i, k]) / s
    q = np.asarray(q, float)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def coverage_check(cam=None, cam_cfg=None, atrium_cfg=None, n_grid=25):
    """Do the square plus margin (floor) and the robot top (0.4 m) anywhere in the square project inside the image?"""
    cam_cfg = cam_cfg or load_config("camera")
    atrium_cfg = atrium_cfg or load_config("atrium")
    cam = cam or Camera.from_config(cam_cfg)
    cx, cy = atrium_cfg["square"]["center"]
    h = atrium_cfg["square"]["side"] / 2
    e = h + cam_cfg["coverage"]["margin"]
    edge = cam_cfg["coverage"]["edge_pixels"]
    corners = np.array([[cx - e, cy - e, 0], [cx + e, cy - e, 0], [cx + e, cy + e, 0], [cx - e, cy + e, 0]], float)
    uv_c, d_c = cam.project(corners)
    g = np.linspace(-h, h, n_grid)
    gx, gy = np.meshgrid(cx + g, cy + g)
    top = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, cam_cfg["coverage"]["robot_height"])])
    uv_t, d_t = cam.project(top)
    ok_c = cam.in_image(uv_c, d_c, edge)
    ok_t = cam.in_image(uv_t, d_t, edge)
    return {
        "corners_world": corners.tolist(),
        "corners_pixels": np.round(uv_c, 2).tolist(),
        "corners_inside": ok_c.tolist(),
        "robot_top_points": int(len(top)),
        "robot_top_inside": int(ok_t.sum()),
        "robot_top_pixel_range": {"u": [float(uv_t[:, 0].min()), float(uv_t[:, 0].max())],
                                  "v": [float(uv_t[:, 1].min()), float(uv_t[:, 1].max())]},
        "all_inside": bool(ok_c.all() and ok_t.all()),
    }

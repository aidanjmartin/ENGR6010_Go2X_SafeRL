"""The MuJoCo camera used for renders matches go2x.atrium.camera.Camera (no OpenGL needed)."""
import mujoco
import numpy as np

from go2x.atrium.camera import Camera
from go2x.config import load_config
from go2x.mj_scene import add_box, add_fixed_camera


def test_fixed_camera_matches_pinhole_model():
    cam = Camera.from_config(load_config("camera"))
    spec = mujoco.MjSpec()
    add_box(spec, "b", (0.5, 0.2, 0.1), (0.2, 0.2, 0.2), 0.3, (1, 0, 0))
    add_fixed_camera(spec, cam)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    fy = (cam.height / 2) / np.tan(np.radians(model.cam_fovy[0]) / 2)
    assert np.isclose(fy, cam.K[1, 1])
    pos, R = data.cam_xpos[0], data.cam_xmat[0].reshape(3, 3)
    for p in ([0, 0, 0], [1.5, -1.2, 0], [-1.5, 1.5, 0.4], [0.3, 0.7, 0.0]):
        pc = R.T @ (np.asarray(p, float) - pos)
        u = cam.width / 2 + fy * pc[0] / -pc[2]
        v = cam.height / 2 - fy * pc[1] / -pc[2]
        assert np.allclose([u, v], cam.project(np.array([p], float))[0], atol=1e-6)
        if p[2] == 0:
            h = cam.floor_homography() @ np.array([p[0], p[1], 1.0])
            assert np.allclose([u, v], h[:2] / h[2], atol=1e-6)

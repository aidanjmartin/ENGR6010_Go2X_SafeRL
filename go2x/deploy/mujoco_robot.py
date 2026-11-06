"""MuJoCo stand-in for the Go2 X behind the same interface as the WebRTC backend (go2x.robot.interface).

Velocity commands (Move) drive the Task 1 walking policy, which runs at 50 Hz on Unitree's go2.xml with PD at every
5 ms physics step (the conventions of scripts/mujoco_check.py). Unitree's own walking controller runs on the real
robot instead, so this backend tests the software path and shows the behavior, not the real robot's tracking.

- Real time: the physics thread keeps simulated time in step with the wall clock.
- State, as the robot sends it over WebRTC: sport-mode state at 20 Hz (odometry position and velocity in the
  body frame, IMU roll, pitch and yaw, body height) and low state at 20 Hz (12 leg motor angles in SDK order).
  The odometry frame is offset from the square frame by `odom_offset` (x, y, yaw), as on the robot, whose
  odometry starts wherever it booted; the runner's mapping (go2x.deploy.nav_runtime.OdomToSquare) must undo it.
- Before stand_up() the robot holds the default pose with PD and ignores Move, like the mock backend.
- The scenario's obstacles are drawn (no contacts: the runner's footprint check stops a run first), people move
  along their scripts from the moment the runner calls start_scene_clock().
- Optional live viewer (needs a display) and an mp4 recording: fixed camera (config/camera.yaml) and a chase
  camera side by side, plus the policy input and a status line the runner passes to annotate().
"""
import threading
import time

import mujoco
import numpy as np
import torch

from go2x.atrium.camera import Camera
from go2x.atrium.state import object_parts
from go2x.config import CONFIG_DIR, load_config
from go2x.mj_scene import BUILDING_COLORS, add_building, add_fixed_camera, add_moving_part, go2_spec
from go2x.robot.interface import RobotInterface, state_from_dict
from go2x.sim2sim import POLICY_JOINTS, SDK_JOINTS, joint_targets, observation, pd_torque, quat_rotate_inverse

SPORT_TOPIC = "mujoco/sportmodestate"
LOW_TOPIC = "mujoco/lowstate"
CHANNEL_RGB = np.array([[0.9, 0.2, 0.2], [0.2, 0.8, 0.2], [0.2, 0.4, 1.0], [1.0, 0.85, 0.0]])


def frame_to_rgb(frame, scale=5):
    """(4, S, S) drawn image -> RGB uint8 panel (static red, people green, robot blue, target yellow)."""
    img = np.asarray(frame, np.float32).clip(0, 1)
    top = np.ones((*img.shape[1:], 3))
    for c in range(4):
        top = top * (1 - img[c, :, :, None]) + img[c, :, :, None] * CHANNEL_RGB[c]
    return np.kron((top * 255).astype(np.uint8), np.ones((scale, scale, 1), np.uint8))


class MujocoRobot(RobotInterface):
    name = "mujoco"

    def __init__(self, model_dir, walk_policy, scene=None, odom_offset=(1.7, -0.6, 0.9), viewer=False,
                 record=None, fps=25, size=(640, 480)):
        super().__init__()
        self.model_dir, self.walk_policy_path = model_dir, walk_policy
        self.scene, self.odom_offset = scene, np.asarray(odom_offset, float)
        self.viewer_on, self.record_path, self.fps, self.size = viewer, record, fps, size
        self.MJ, self.WALK = load_config("mujoco"), load_config("walk")
        self.cmd = np.zeros(3)
        self.standing = False
        self.low_listeners = []
        self.overlay = None
        self.scene_t0 = None
        self.n_frames = 0
        self._want_record = record is not None
        self.writer = None
        self.fell = False
        self._run = threading.Event()
        self._thread = None
        self._ready = threading.Event()
        self._error = None

    # -- interface -------------------------------------------------------------------
    def add_low_state_listener(self, fn):
        self.low_listeners.append(fn)

    def connect(self):
        self._run.set()
        self._thread = threading.Thread(target=self._main, daemon=True)
        self._thread.start()
        if not self._ready.wait(60):
            raise RuntimeError("MuJoCo backend did not start")
        if self._error:
            raise self._error

    def close(self):
        self._run.clear()
        if self._thread:
            self._thread.join(timeout=10)

    def send_velocity(self, vx, vy, wz):
        if self.standing:
            self.cmd = np.array([vx, vy, wz], float)

    def stand_up(self):
        self.standing = True

    def stop_move(self):
        self.cmd = np.zeros(3)

    def start_scene_clock(self):
        self.scene_t0 = self.sim_time

    def scene_time(self):
        return None if self.scene_t0 is None else self.sim_time - self.scene_t0

    def start_recording(self):
        """Start writing self.record_path (the renderer is made in the physics thread, which owns its context)."""
        self._want_record = True

    def annotate(self, frame, text):
        self.overlay = (frame_to_rgb(frame), text)

    def info(self):
        return {"backend": self.name, "model": self.MJ["model"], "mujoco": mujoco.__version__,
                "walk_policy": str(self.walk_policy_path), "odom_offset": self.odom_offset.tolist(),
                "note": "the Task 1 walking policy stands in for Unitree's walking controller"}

    # -- simulation ------------------------------------------------------------------
    def _build(self):
        cam = Camera.from_config(load_config("camera"))
        self.cam = cam
        spec = go2_spec(self.model_dir, offscreen=self.size, checker=True)
        import json
        with open(CONFIG_DIR / "building_level1_near_square.json") as f:
            add_building(spec, json.load(f)["boxes"], BUILDING_COLORS)
        self.parts = object_parts(load_config("atrium"))
        for p in self.parts:
            add_moving_part(spec, p)
        add_fixed_camera(spec, cam)
        m = spec.compile()
        m.opt.timestep = self.MJ["timestep"]
        self.model, self.data = m, mujoco.MjData(m)
        self.mocap = [m.body("mocap_" + p["name"]).mocapid[0] for p in self.parts]
        self.qadr = np.array([m.joint(n).qposadr[0] for n in POLICY_JOINTS])
        self.vadr = np.array([m.joint(n).dofadr[0] for n in POLICY_JOINTS])
        self.sdk_qadr = np.array([m.joint(n).qposadr[0] for n in SDK_JOINTS])
        self.sdk_vadr = np.array([m.joint(n).dofadr[0] for n in SDK_JOINTS])
        self.act = np.array([m.actuator(n.replace("_joint", "")).id for n in POLICY_JOINTS])
        self.limit = m.actuator_ctrlrange[self.act, 1]
        self.gyro = m.sensor("imu_gyro").adr[0]
        self.quat = m.sensor("imu_quat").adr[0]
        self.base_id = m.body("base_link").id if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link") >= 0 \
            else m.body("base").id
        self.policy = torch.jit.load(self.walk_policy_path, map_location="cpu").eval()
        a = self.WALK["actuator"]
        self.kp, self.kd, self.scale = a["kp"], a["kd"], a["action_scale"]

    def _reset(self):
        d = self.data
        mujoco.mj_resetData(self.model, d)
        x, y, yaw = self.scene.start_pose if self.scene is not None else (0.0, 0.0, 0.0)
        d.qpos[0:3] = [x, y, self.MJ["start_height"]]
        d.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        d.qpos[self.qadr] = joint_targets(np.zeros(12), self.scale)
        self._set_parts()
        mujoco.mj_forward(self.model, d)
        self.last = np.zeros(12, np.float32)
        self.sim_time = 0.0

    def _set_parts(self):
        if self.scene is None:
            return
        poses = self.scene.part_poses()
        for j, mid in enumerate(self.mocap):
            self.data.mocap_pos[mid] = poses[j, :3]
            self.data.mocap_quat[mid] = poses[j, 3:7]

    def _policy_step(self):
        d = self.data
        if self.standing:
            obs = observation(d.sensordata[self.gyro:self.gyro + 3], d.sensordata[self.quat:self.quat + 4].copy(),
                              self.cmd.astype(np.float32), d.qpos[self.qadr], d.qvel[self.vadr], self.last)
            with torch.no_grad():
                self.last = self.policy(torch.from_numpy(obs)[None])[0].numpy().astype(np.float32)
            target = joint_targets(self.last, self.scale)
        else:
            target = joint_targets(np.zeros(12), self.scale)
        for _ in range(self.MJ["pd_substeps"]):
            d.ctrl[self.act] = pd_torque(target, d.qpos[self.qadr], d.qvel[self.vadr], self.kp, self.kd, self.limit)
            mujoco.mj_step(self.model, d)
        self.sim_time += self.MJ["timestep"] * self.MJ["pd_substeps"]

    def _rpy(self):
        w, x, y, z = self.data.qpos[3:7]
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return np.array([roll, pitch, yaw])

    def _publish(self):
        d = self.data
        ox, oy, oyaw = self.odom_offset
        c, s = np.cos(oyaw), np.sin(oyaw)
        px, py = d.qpos[0], d.qpos[1]
        rpy = self._rpy()
        v_b = quat_rotate_inverse(d.qpos[3:7], d.qvel[0:3])
        now = time.time_ns()
        raw = {"stamp": {"sec": now // 1_000_000_000, "nanosec": now % 1_000_000_000},
               "mode": 1 if self.standing else 0,
               "position": [float(ox + c * px - s * py), float(oy + s * px + c * py), float(d.qpos[2])],
               "velocity": [float(v_b[0]), float(v_b[1]), float(v_b[2])],
               "yaw_speed": float(d.qvel[5]),
               "imu_state": {"rpy": [float(rpy[0]), float(rpy[1]), float((rpy[2] + oyaw + np.pi) % (2 * np.pi) - np.pi)]},
               "body_height": float(d.qpos[2]),
               "sim_time": self.sim_time}
        self._on_state(state_from_dict(raw, SPORT_TOPIC))
        low = {"motor_state": [{"q": float(q), "dq": float(dq)} for q, dq in zip(d.qpos[self.sdk_qadr], d.qvel[self.sdk_vadr])],
               "imu_state": {"rpy": raw["imu_state"]["rpy"]}, "sim_time": self.sim_time}
        for fn in self.low_listeners:
            fn(low)

    def _render(self, renderer, chase):
        d = self.data
        renderer.update_scene(d, camera="top")
        a = renderer.render().copy()
        chase.lookat[:] = [d.qpos[0], d.qpos[1], 0.25]
        renderer.update_scene(d, camera=chase)
        b = renderer.render().copy()
        frame = np.concatenate([a, b], 1)
        from PIL import Image, ImageDraw
        panel, text = self.overlay if self.overlay is not None else (np.full((320, 320, 3), 255, np.uint8), "standing up")
        h = min(panel.shape[0], frame.shape[0])
        pad = np.full((frame.shape[0], 320, 3), 255, np.uint8)
        pad[:h, : panel.shape[1]] = panel[:h, :320]
        img = Image.fromarray(np.concatenate([frame, pad], 1))
        dr = ImageDraw.Draw(img)
        dr.rectangle((0, 0, 2 * self.size[0], 22), fill=(255, 255, 255))
        dr.text((8, 6), f"MuJoCo  t {self.sim_time:5.1f} s   {text}", fill=(0, 0, 0))
        dr.text((2 * self.size[0] + 6, h + 6), "policy input (robot view)", fill=(0, 0, 0))
        frame = np.asarray(img)
        self.writer.append_data(frame)
        self.n_frames += 1

    def _main(self):
        try:
            self._build()
            self._reset()
        except BaseException as e:   # report to connect()
            self._error = e
            self._ready.set()
            return
        viewer = None
        if self.viewer_on:
            from mujoco import viewer as mj_viewer
            viewer = mj_viewer.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False)
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = self.base_id
            viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 3.0, -30.0, 135.0
        renderer = chase = None
        dt = self.MJ["timestep"] * self.MJ["pd_substeps"]
        next_pub, next_rec = 0.0, 0.0
        sync_every = 2
        t_wall0 = time.monotonic()
        k = 0
        self._ready.set()
        try:
            while self._run.is_set():
                if self.scene is not None and self.scene_t0 is not None:
                    self.scene.set_time(self.sim_time - self.scene_t0)
                    self._set_parts()
                if viewer is not None:
                    with viewer.lock():
                        self._policy_step()
                else:
                    self._policy_step()
                if np.abs(self._rpy()[:2]).max() > self.MJ["fall"]["tilt_rad"] or self.data.qpos[2] < self.MJ["fall"]["base_height_m"]:
                    self.fell = True
                if self.sim_time >= next_pub - 1e-9:
                    self._publish()
                    next_pub += 1 / 20
                if self._want_record and renderer is None and self.record_path:
                    import imageio
                    renderer = mujoco.Renderer(self.model, self.size[1], self.size[0])
                    self.writer = imageio.get_writer(self.record_path, fps=self.fps, quality=7, macro_block_size=8)
                    chase = mujoco.MjvCamera()
                    chase.type = mujoco.mjtCamera.mjCAMERA_FREE
                    chase.distance, chase.azimuth, chase.elevation = 2.4, 135.0, -22.0
                    next_rec = self.sim_time
                if renderer is not None and self.sim_time >= next_rec - 1e-9:
                    self._render(renderer, chase)
                    next_rec += 1 / self.fps
                if viewer is not None and k % sync_every == 0:
                    if not viewer.is_running():
                        viewer = None
                    else:
                        viewer.sync()
                k += 1
                lag = t_wall0 + k * dt - time.monotonic()
                if lag > 0:
                    time.sleep(lag)
        finally:
            if viewer is not None:
                viewer.close()
            if renderer is not None:
                renderer.close()
                self.writer.close()
                print(f"[mujoco] recording: {self.record_path} ({self.n_frames} frames, "
                      f"{self.n_frames / self.fps:.1f} s)", flush=True)

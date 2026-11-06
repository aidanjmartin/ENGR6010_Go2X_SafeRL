"""Wireless WebRTC backend for the Go2 X: the Phase 1 backend (go2x/robot/webrtc.py, tested over the Ethernet
tether on 2026-10-17) with the connection method selectable and the low state (motor angles) subscribed.

UNTESTED OVER WI-FI. Only the connection method and one extra subscription differ from the tested backend.

From legion1581/unitree_webrtc_connect at commit e0abc57 (the pinned commit):
- WebRTCConnectionMethod.LocalSTA (2): the robot and this computer on the same Wi-Fi network, robot reached at
  its Wi-Fi address; LocalAP (1): this computer joined to the robot's own hotspot, no address needed
  (unitree_webrtc_connect/constants.py:21-24; examples/go2/data_channel/lowstate/lowstate.py:56-59).
- RTC_TOPIC["LOW_STATE"] = "rt/lf/lowstate" (constants.py:63), message["data"] with motor_state[i]["q"],
  imu_state["rpy"], foot_force (examples/go2/data_channel/lowstate/lowstate.py:14-40). Motor order is the SDK
  order (FR, FL, RR, RL; hip, thigh, calf), as in docs/LOWLEVEL_INTERFACE.md.
"""
import asyncio
import json
import threading

from go2x.robot.webrtc import WebRTCRobot


class WirelessWebRTCRobot(WebRTCRobot):
    name = "webrtc"

    def __init__(self, method="sta", ip=None, **kwargs):
        if method not in ("sta", "ap"):
            raise ValueError("method must be 'sta' (same Wi-Fi network) or 'ap' (robot hotspot)")
        self.method = method
        if method == "ap":
            ip = ip or "unused"           # LocalAP needs no address; WebRTCRobot insists on one
        super().__init__(ip=ip, **kwargs)
        self.low_listeners = []

    def add_low_state_listener(self, fn):
        self.low_listeners.append(fn)

    def connect(self):
        from unitree_webrtc_connect.constants import RTC_TOPIC
        from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection, WebRTCConnectionMethod
        self._topics = RTC_TOPIC
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True).start()
        if self.method == "ap":
            self._conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalAP, aes_128_key=self.aes_key)
        else:
            self._conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=self.ip, aes_128_key=self.aes_key)
        self._run(self._conn.connect(), self.connect_timeout_s)
        sub = self._conn.datachannel.pub_sub.subscribe
        self._loop.call_soon_threadsafe(sub, RTC_TOPIC["LF_SPORT_MOD_STATE"], self._handler)
        self._loop.call_soon_threadsafe(sub, RTC_TOPIC["LOW_STATE"], self._low_handler)
        try:
            resp = self._request(RTC_TOPIC["MOTION_SWITCHER"], {"api_id": 1001})
            self.motion_mode = json.loads(resp["data"]["data"]).get("name")
        except Exception as e:  # diagnosis only
            self.motion_mode = f"query failed: {e!r}"
        print(f"motion switcher reports: {self.motion_mode}", flush=True)

    def _low_handler(self, message):
        data = message.get("data") or {}
        for fn in self.low_listeners:
            fn(data)

    def info(self):
        d = super().info()
        d.update(method={"sta": "LocalSTA (same Wi-Fi network)", "ap": "LocalAP (robot hotspot)"}[self.method],
                 topics=["rt/lf/sportmodestate", "rt/lf/lowstate"],
                 untested_over_wifi=True)
        d.pop("untested_on_hardware", None)
        return d

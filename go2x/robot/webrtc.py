"""WebRTC backend: unitree_webrtc_connect sport requests over the Ethernet tether.

UNTESTED ON HARDWARE. Exam 1 connected to this Go2 X over WebRTC read-only (video, state);
no sport request has been sent to it yet.

Every library call below is taken from legion1581/unitree_webrtc_connect at commit e0abc57
(pinned in ENGR6010_Exam1 q3_apps/requirements.txt). Paths are relative to that repository:
- UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=..., aes_128_key=...),
  .connect(), .disconnect(): unitree_webrtc_connect/webrtc_driver.py; used the same way in
  ENGR6010_Exam1 q3_apps/webrtc_read_state.py (commit 90150da).
- conn.datachannel.pub_sub.subscribe(RTC_TOPIC["LF_SPORT_MOD_STATE"], callback), with the
  message fields in message["data"]: msgs/pub_sub.py (subscribe); constants.py:77
  ("rt/lf/sportmodestate"); same use in Exam 1 webrtc_read_state.py, which measured 20 Hz.
- Move as a no-reply request on RTC_TOPIC["SPORT_MOD"] ("rt/api/sport/request",
  constants.py:75) through pub_sub.publish_without_callback, with header.policy.noreply = True:
  copied from examples/go2/data_channel/sportmode_mcf/sportmode_mcf.py:109-126 (call_no_reply).
- StandUp, BalanceStand, StopMove as request/response through pub_sub.publish_request_new:
  sportmode_mcf.py:96-106 (call). API ids from constants.py SPORT_CMD (Move 1008, StopMove 1003,
  StandUp 1004, BalanceStand 1002), identical in SPORT_CMD_MCF.
- BalanceStand after StandUp before any Move: sportmode_mcf.py:301-303.
- Read-only motion-switcher query (api_id 1001 on RTC_TOPIC["MOTION_SWITCHER"],
  constants.py:119), logged for diagnosis only: examples/go2/data_channel/sportmode/sportmode.py:29-37.
  This backend never switches modes; Exam 1's firmware (1.1.11 to 1.1.14) is past 1.1.7, where
  the MCF example says no motion-switcher handshake is needed (sportmode_mcf.py:3-5).

The library is asyncio-based. It runs on its own event loop in a background thread; the
blocking methods below hand work to that loop.
"""
import asyncio
import json
import random
import threading
import time

from go2x.robot.interface import RobotInterface, read_env, state_from_dict


class WebRTCRobot(RobotInterface):
    name = "webrtc"

    def __init__(self, ip=None, aes_key=None, request_timeout_s=5.0, connect_timeout_s=20.0):
        super().__init__()
        env = read_env()
        self.ip = ip or env.get("GO2_IP")
        if not self.ip:
            raise SystemExit("Set GO2_IP in .env, or pass ip")
        self.aes_key = aes_key or env.get("GO2_AES_KEY") or None
        self.request_timeout_s = request_timeout_s
        self.connect_timeout_s = connect_timeout_s
        self.motion_mode = None
        self._loop = None
        self._conn = None

    # -- event loop plumbing --------------------------------------------------------
    def _run(self, coro, timeout):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    def connect(self):
        from unitree_webrtc_connect.constants import RTC_TOPIC
        from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection, WebRTCConnectionMethod
        self._topics = RTC_TOPIC

        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True).start()
        self._conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=self.ip, aes_128_key=self.aes_key)
        self._run(self._conn.connect(), self.connect_timeout_s)
        self._loop.call_soon_threadsafe(
            self._conn.datachannel.pub_sub.subscribe, RTC_TOPIC["LF_SPORT_MOD_STATE"], self._handler)
        try:
            resp = self._request(RTC_TOPIC["MOTION_SWITCHER"], {"api_id": 1001})
            self.motion_mode = json.loads(resp["data"]["data"]).get("name")
        except Exception as e:  # diagnosis only
            self.motion_mode = f"query failed: {e!r}"
        print(f"motion switcher reports: {self.motion_mode}", flush=True)

    def _handler(self, message):
        self._on_state(state_from_dict(message.get("data") or {}, self._topics["LF_SPORT_MOD_STATE"]))

    def close(self):
        if self._conn is not None:
            try:
                self._run(self._conn.disconnect(), 5.0)
            finally:
                self._conn = None
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop = None

    # -- requests -------------------------------------------------------------------
    def _request(self, topic, options):
        """Request/response; publish_request_new waits without a timeout, so add one."""
        coro = asyncio.wait_for(self._conn.datachannel.pub_sub.publish_request_new(topic, options),
                                self.request_timeout_s)
        return self._run(coro, self.request_timeout_s + 1.0)

    def _sport_call(self, name):
        from unitree_webrtc_connect.constants import SPORT_CMD
        resp = self._request(self._topics["SPORT_MOD"], {"api_id": SPORT_CMD[name]})
        code = resp.get("data", {}).get("header", {}).get("status", {}).get("code", -1)
        print(f"{name} returned {code}", flush=True)
        return code

    def send_velocity(self, vx, vy, wz):
        from unitree_webrtc_connect.constants import SPORT_CMD
        pub_sub = self._conn.datachannel.pub_sub
        # publish_without_callback silently drops the message if the channel is closed
        # (msgs/pub_sub.py), so check first.
        if pub_sub.channel.readyState != "open":
            raise RuntimeError("WebRTC data channel is not open")
        generated_id = int(time.time() * 1000) % 2147483648 + random.randint(0, 1000)
        payload = {
            "header": {"identity": {"id": generated_id, "api_id": SPORT_CMD["Move"]},
                       "policy": {"priority": 0, "noreply": True}},
            "parameter": json.dumps({"x": float(vx), "y": float(vy), "z": float(wz)}),
            "binary": [],
        }
        self._loop.call_soon_threadsafe(pub_sub.publish_without_callback, self._topics["SPORT_MOD"], payload)

    def stand_up(self):
        self._sport_call("StandUp")
        time.sleep(2.0)
        self._sport_call("BalanceStand")

    def stop_move(self):
        self._sport_call("StopMove")

    def info(self):
        return {"backend": self.name, "library": "unitree_webrtc_connect e0abc57",
                "topic": "rt/lf/sportmodestate", "motion_mode": self.motion_mode,
                "untested_on_hardware": True}

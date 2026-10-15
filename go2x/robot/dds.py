"""DDS backend: unitree_sdk2_python SportClient over the Ethernet tether.

UNTESTED ON HARDWARE. Exam 1 reached this Go2 X over DDS read-only (camera, LowState);
no sport request has been sent to it yet.

Every SDK call below is taken from unitree_sdk2_python at commit 814556d (the commit pinned in
ENGR6010_Exam1 q2_sdk/LOG.md). Paths are relative to that repository:
- ChannelFactoryInitialize(0, iface): unitree_sdk2py/core/channel.py (def ChannelFactoryInitialize);
  used as in example/go2/high_level/go2_sport_client.py:74-77.
- ChannelSubscriber(topic, SportModeState_) and .Init(handler, 10): core/channel.py
  (class ChannelSubscriber); used as in example/a2/sport/a2_sport_state.py:32-33. That file
  (line 8) is the only place in this SDK commit that names a sport-state topic:
  "rt/lf/sportmodestate". .Close(): core/channel.py (ChannelSubscriber.Close).
- SportModeState_: unitree_sdk2py/idl/unitree_go/msg/dds_/_SportModeState_.py (fields stamp,
  mode, position, velocity, yaw_speed, imu_state.rpy, body_height, ...); imported as in
  example/go2/high_level/go2_sport_client.py:5.
- SportClient(), SetTimeout(), Init(): go2_sport_client.py:83-85.
- Move(vx, vy, vyaw) (no-reply call, API 1008), StopMove() (1003), StandUp() (1004),
  BalanceStand() (1002): unitree_sdk2py/go2/sport/sport_client.py.
- BalanceStand after StandUp before any Move: hint printed by unitree_webrtc_connect
  examples/go2/data_channel/sportmode_mcf/sportmode_mcf.py:301-303 (same API ids).
"""
import dataclasses
import time

from go2x.robot.interface import RobotInterface, read_env, state_from_dict

TOPIC = "rt/lf/sportmodestate"


def _to_dict(msg):
    """IDL structs are dataclasses; arrays become lists."""
    def conv(v):
        if dataclasses.is_dataclass(v):
            return {f.name: conv(getattr(v, f.name)) for f in dataclasses.fields(v)}
        if isinstance(v, (list, tuple)):
            return [conv(x) for x in v]
        if hasattr(v, "tolist"):
            return v.tolist()
        return v
    return conv(msg)


class DDSRobot(RobotInterface):
    name = "dds"

    def __init__(self, iface=None, topic=TOPIC, rpc_timeout_s=5.0):
        super().__init__()
        self.iface = iface or read_env().get("GO2_ETH_IFACE")
        if not self.iface:
            raise SystemExit("Set GO2_ETH_IFACE in .env (the USB-C Ethernet adapter), or pass iface")
        self.topic = topic
        self.rpc_timeout_s = rpc_timeout_s
        self._sub = None
        self._client = None

    def connect(self):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.go2.sport.sport_client import SportClient
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

        ChannelFactoryInitialize(0, self.iface)
        self._sub = ChannelSubscriber(self.topic, SportModeState_)
        self._sub.Init(self._handler, 10)
        self._client = SportClient()
        self._client.SetTimeout(self.rpc_timeout_s)
        self._client.Init()

    def _handler(self, msg):
        self._on_state(state_from_dict(_to_dict(msg), self.topic))

    def close(self):
        if self._sub is not None:
            self._sub.Close()
            self._sub = None

    def send_velocity(self, vx, vy, wz):
        code = self._client.Move(float(vx), float(vy), float(wz))
        if code not in (0, None):
            raise RuntimeError(f"SportClient.Move returned {code}")

    def stand_up(self):
        code = self._client.StandUp()
        print(f"StandUp returned {code}", flush=True)
        time.sleep(2.0)
        code = self._client.BalanceStand()
        print(f"BalanceStand returned {code}", flush=True)

    def stop_move(self):
        code = self._client.StopMove()
        print(f"StopMove returned {code}", flush=True)

    def info(self):
        return {"backend": self.name, "library": "unitree_sdk2_python 814556d", "topic": self.topic,
                "iface": self.iface, "untested_on_hardware": True}

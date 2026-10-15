"""Summarize a motion session log: commanded vs measured velocity, rough delay, displacement.

Usage:
    python scripts/motion/summarize_log.py data/<session folder>

The frame of SportModeState.velocity on the Go2 X is not confirmed (LOG.md). Linear velocity
is therefore shown two ways: as reported, and rotated from world to body by the IMU yaw.
Whichever matches the commanded direction during the rotate and square tests is the frame.
"""
import json
import sys
from pathlib import Path

import numpy as np


def load(session):
    session = Path(session)
    read = lambda n: [json.loads(line) for line in open(session / n)]
    return json.loads((session / "header.json").read_text()), read("commands.jsonl"), read("state.jsonl"), read("events.jsonl")


def state_arrays(states):
    t = np.array([s["t_mono_ns"] for s in states]) / 1e9
    d = [s["data"] for s in states]
    vel = np.array([(x.get("velocity") or [np.nan] * 3)[:2] for x in d], float)
    wz = np.array([x.get("yaw_speed", np.nan) for x in d], float)
    yaw = np.array([((x.get("imu_state") or {}).get("rpy") or [np.nan] * 3)[2] for x in d], float)
    pos = np.array([(x.get("position") or [np.nan] * 3)[:2] for x in d], float)
    c, s = np.cos(yaw), np.sin(yaw)
    vel_world_to_body = np.stack([c * vel[:, 0] + s * vel[:, 1], -s * vel[:, 0] + c * vel[:, 1]], 1)
    return t, vel, vel_world_to_body, wz, yaw, pos


def first_cross(t, x, t0, level):
    idx = np.where((t >= t0) & (np.abs(x) >= abs(level)))[0]
    return t[idx[0]] - t0 if len(idx) else None


def summarize(session, out=print):
    header, cmds, states, events = load(session)
    if not states or not cmds:
        out("no state or no commands logged")
        return
    tc = np.array([c["t_mono_ns"] for c in cmds]) / 1e9
    sent = np.array([c["sent"] for c in cmds], float)
    t, v_rep, v_rot, wz, yaw, pos = state_arrays(states)
    t0 = t[0]

    gaps = np.diff(t)
    out(f"session: {Path(session).name}   backend: {header['backend'].get('backend')}")
    out(f"state: {len(t)} messages over {t[-1] - t[0]:.1f} s, {len(t) / max(t[-1] - t[0], 1e-9):.1f} per second, "
        f"largest gap {gaps.max() * 1000:.0f} ms" if len(t) > 1 else "state: one message")
    stop = [e for e in events if e["event"] == "stop"]
    if stop:
        out(f"stop reason: {stop[0].get('reason')}")

    out("")
    out("axis | commanded | measured (as reported) | measured (world->body) | t to 10% | t to 50%")
    for i, (name, unit) in enumerate((("vx", "m/s"), ("vy", "m/s"), ("wz", "rad/s"))):
        peak = sent[np.argmax(np.abs(sent[:, i])), i]
        if abs(peak) < 1e-6:
            continue
        on = np.abs(sent[:, i] - peak) < 1e-9          # commands at the plateau
        t_first = tc[np.abs(sent[:, i]) > 1e-9][0]       # first nonzero command on this axis
        # each contiguous plateau (the square has four); skip the first second of each
        edges = np.diff(np.concatenate([[0], on.astype(int), [0]]))
        starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0] - 1
        window = np.zeros(len(t), bool)
        for a, b in zip(starts, ends):
            window |= (t >= tc[a] + 1.0) & (t <= tc[b])
        if i < 2:
            m_rep = np.nanmean(v_rep[window, i]) if window.any() else np.nan
            m_rot = np.nanmean(v_rot[window, i]) if window.any() else np.nan
            meas = v_rep[:, i]
        else:
            m_rep = m_rot = np.nanmean(wz[window]) if window.any() else np.nan
            meas = wz
        t10, t50 = first_cross(t, meas, t_first, 0.1 * peak), first_cross(t, meas, t_first, 0.5 * peak)
        fmt = lambda x: "n/a" if x is None else f"{x * 1000:.0f} ms"
        rot = "same" if i == 2 else f"{m_rot:+.3f} {unit}"
        out(f"{name} | {peak:+.3f} {unit} | {m_rep:+.3f} {unit} | {rot} | {fmt(t10)} | {fmt(t50)}")

    dt_c = np.diff(tc, append=tc[-1])
    cmd_dist = np.abs(sent[:, :2] * dt_c[:, None]).sum(0)
    cmd_yaw = (sent[:, 2] * dt_c).sum()
    disp = pos[-1] - pos[0]
    dyaw = (yaw[-1] - yaw[0] + np.pi) % (2 * np.pi) - np.pi
    out("")
    out(f"commanded travel (integral of sent commands): |x| {cmd_dist[0]:.3f} m, |y| {cmd_dist[1]:.3f} m, yaw {np.degrees(cmd_yaw):+.1f} deg")
    out(f"odometry displacement (state position, frame unconfirmed): dx {disp[0]:+.3f} m, dy {disp[1]:+.3f} m, "
        f"|d| {np.hypot(*disp):.3f} m; IMU yaw change {np.degrees(dyaw):+.1f} deg")
    out("t to 10% / 50%: from the first nonzero command on that axis to the first state at 10% / 50% of the "
        "commanded value (as reported). Rough; includes the rate limit ramp and the robot's own response.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    summarize(sys.argv[1])

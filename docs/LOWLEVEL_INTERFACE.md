# Low-level interface contract: walking policy on the Go2 X

This document fixes the interface between the Task 1 walking policy and a Go2 X running in low-level mode. **No deploy or hardware script exists yet.** Everything below is the contract a later deploy script must follow. It is a plan, not yet built or tested on the robot.

## Assumption

The Go2 X accepts Unitree's low-level command with Unitree's built-in walking controller off. For each of the 12 leg joints, the command carries a position target `q`, a velocity target `dq`, gains `kp` and `kd`, and a feedforward torque `tau`. Joint positions, joint velocities, and the inertial measurement unit (IMU) readings come back.

Source for the message format: `unitreerobotics/unitree_sdk2_python` commit `814556d15970dd2ecf1c9984e845ca02ab07e206` (the commit Phase 1 pinned), `example/go2/low_level/go2_stand_example.py`:
- commands on DDS (Data Distribution Service) topic `rt/lowcmd`, type `LowCmd_`;
- state on `rt/lowstate`, type `LowState_`;
- writer period `dt = 0.002` s, so commands go out at 500 Hz;
- `head = [0xFE, 0xEF]`, `level_flag = 0xFF` (low level);
- `motor_cmd[i].mode = 0x01`;
- `motor_cmd[i].q`, `.dq`, `.kp`, `.kd`, `.tau` set per motor.

Sources for the policy side:
- `unitreerobotics/unitree_rl_lab` commit `4960b84732b0c2ec593dccbfe963fda1bcd7b1e3`: `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py` (`UNITREE_GO2_CFG`) and `tasks/locomotion/robots/go2/velocity_env_cfg.py`;
- this repository's `config/walk.yaml`;
- the joint order and limits that job 020 read from the simulated robot (`results/phase2/walk/robot_info.json`).

The robot model is `unitreerobotics/unitree_ros` commit `3ae893d905d11795ddf1075825ee5010364a1109`, `robots/go2_description/urdf/go2_description.urdf`.

## Joint order

Two orders exist, so a deploy script must map between them.

The **SDK order** is the index into `motor_cmd[]` and `motor_state[]`. Source: `unitree_sdk2_python` `814556d` `example/go2/low_level/unitree_legged_const.py` (`LegID`), and `UNITREE_GO2_CFG.joint_sdk_names`.

The **policy order** is the order of every 12-element policy input and output. It is the order Isaac Lab gives the articulation's joints, read from the simulated robot in job 020.

| policy index | joint | SDK index (`joint_ids_map`) |
| ---: | :--- | ---: |
| 0 | FL_hip_joint | 3 |
| 1 | FR_hip_joint | 0 |
| 2 | RL_hip_joint | 9 |
| 3 | RR_hip_joint | 6 |
| 4 | FL_thigh_joint | 4 |
| 5 | FR_thigh_joint | 1 |
| 6 | RL_thigh_joint | 10 |
| 7 | RR_thigh_joint | 7 |
| 8 | FL_calf_joint | 5 |
| 9 | FR_calf_joint | 2 |
| 10 | RL_calf_joint | 11 |
| 11 | RR_calf_joint | 8 |

`joint_ids_map = [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]` comes from `unitree_rl_lab` `utils/export_deploy_cfg.py`, which writes it to `deploy.yaml`. The policy's joint `i` is SDK motor `joint_ids_map[i]`. This is the same mapping Unitree's own deploy code uses (`deploy/robots/go2/src/State_RLBase.cpp`, `motor_cmd()[joint_ids_map[i]].q() = action[i]`).

## Default standing pose and joint limits

Values in radians, in policy order. The default pose comes from `UNITREE_GO2_CFG.init_state.joint_pos`. Hard limits come from the URDF `<limit>` tags. Soft limits are the hard limits narrowed to 90 % about their middle (`soft_joint_pos_limit_factor = 0.9` in `UnitreeArticulationCfg`). Job 020 read all three from the simulated robot.

| joint group | default | hard limit | soft limit |
| :--- | ---: | :--- | :--- |
| FL_hip, RL_hip | +0.1 | [-1.0472, 1.0472] | [-0.942, 0.942] |
| FR_hip, RR_hip | -0.1 | [-1.0472, 1.0472] | [-0.942, 0.942] |
| FL_thigh, FR_thigh | 0.8 | [-1.5708, 3.4907] | [-1.318, 3.238] |
| RL_thigh, RR_thigh | 1.0 | [-0.5236, 4.5379] | [-0.271, 4.285] |
| all calves | -1.5 | [-2.7227, -0.8378] | [-2.628, -0.932] |

The URDF effort limits are 23.7 N m for hips and thighs and 45.43 N m for calves. Velocity limits are 30.1 rad/s for hips and thighs and 15.70 rad/s for calves.

In simulation, `unitree_rl_lab`'s `UnitreeActuatorCfg_Go2HV` limits every joint to 20.2 N m when torque and speed have the same sign and 23.4 N m when they are opposed, falling to zero between 13.5 and 30 rad/s.

## Gains, action scale, and rates

| item | value | source |
| :--- | :--- | :--- |
| kp | 25.0 N m/rad, all joints | `config/walk.yaml` `actuator.kp` (`UNITREE_GO2_CFG`) |
| kd | 0.5 N m s/rad, all joints | `config/walk.yaml` `actuator.kd` |
| feedforward torque | 0 | the policy outputs positions only |
| velocity target `dq` | 0 | |
| action scale | 0.25 | `config/walk.yaml` `actuator.action_scale` |
| joint target | `q_target[i] = default[i] + 0.25 * action[i]` (policy order) | `JointPositionActionCfg(scale=0.25, use_default_offset=True)` |
| action clip | ±100 (no practical clip) | `velocity_env_cfg.py` |
| policy rate | 50 Hz (0.02 s) | `sim_dt` 0.005 s times `decimation` 4 |
| PD rate in simulation | 200 Hz, torque computed by the explicit actuator model each physics step | `sim_dt` 0.005 s |
| command rate on the robot | 500 Hz (0.002 s); the motor drivers close the PD loop with the sent `kp`, `kd` | SDK example `dt = 0.002` |
| action delay in training | 4 to 8 physics steps (20 to 40 ms), drawn per episode | `config/walk.yaml` `randomization.action_delay_physics_steps` |

Between policy updates, the deploy loop resends the latest joint targets at 500 Hz.

The training randomized kp and kd by ±20 % and torque by ±20 %, so the policy has seen some gain error. Whether the Go2 X motor drivers behave like the simulated PD with these gains is unmeasured.

## Observation (45 values, in this order)

All vectors are body frame. Joint vectors are in policy order. Scales are applied before the values enter the policy. Training adds uniform noise (`config/walk.yaml` `randomization.obs_noise`); a deploy script adds none.

| term | size | unit | scale | robot source |
| :--- | ---: | :--- | ---: | :--- |
| base angular velocity | 3 | rad/s | 0.2 | `lowstate.imu_state.gyroscope` (Unitree deploy: `unitree_articulation.h`) |
| projected gravity | 3 | unit vector | 1.0 | gravity (0, 0, -1) rotated into the body frame by the inverse of `imu_state.quaternion` (w, x, y, z) |
| velocity target (vx, vy, wz) | 3 | m/s, m/s, rad/s | 1.0 | the navigation policy, limited to `config/nav.yaml` `action` |
| joint position minus default | 12 | rad | 1.0 | `motor_state[joint_ids_map[i]].q - default[i]` |
| joint velocity | 12 | rad/s | 0.05 | `motor_state[joint_ids_map[i]].dq` |
| previous action | 12 | (policy output) | 1.0 | the last raw policy output, before scaling |

The exported policy (`policy.pt` TorchScript, `policy.onnx`) includes any observation normalizer, so it takes these 45 values directly. The walking task's training runs used `empirical_normalization = False`.

## Safety limits a deploy script must enforce

None of these exists in code yet. All are requirements on the future deploy script. Numbers marked "proposed" are not from any source and need Aidan's approval before robot use.

1. **Joint targets:** clamp every `q_target` to the soft limits above before sending.
2. **Torque:** with `tau = 0`, the motor torque is `kp (q_target - q) - kd dq`. Stop (see item 7) if any measured joint torque exceeds the URDF effort limit: 23.7 N m for hips and thighs, 45.43 N m for calves.
3. **Joint velocity:** stop if any joint exceeds its URDF velocity limit: 30.1 rad/s for hips and thighs, 15.70 rad/s for calves.
4. **Tilt:** stop if the angle between projected gravity and (0, 0, -1) exceeds 0.8 rad. That is the training's `bad_orientation` limit; Unitree's deploy code switches to its passive state at 1.0 rad.
5. **Velocity target:** clamp to the navigation limits in `config/nav.yaml` (vx 0 to 0.8 m/s, vy -0.3 to 0.3 m/s, wz -1.0 to 1.0 rad/s). These lie inside the walking training range.
6. **Watchdog:** stop if `rt/lowstate` is older than 3 policy periods (60 ms, proposed) or the policy misses 3 updates in a row (proposed).
7. **Stop behavior:** a damping state with `kp = 0`, `kd = 3`, `tau = 0`. That is Unitree's `Passive` state in `deploy/robots/go2/config/config.yaml`.
8. **Start:** move from the measured pose to the default pose over at least 1 s, using Unitree's `FixStand` interpolation idea (same file). Start the policy only from a standing default pose. Reject non-finite policy outputs.
9. **Bench first:** the first runs happen with the robot lifted or tethered, at reduced kp (proposed: 50 %), before any floor test.

Walking policy, Isaac Lab vs. MuJoCo 3.16.0 (unitree_mujoco Go2), same 100 episodes and velocity targets. Mean and standard error over episodes; falls per minute with a Poisson standard error.

| metric | Isaac Lab | MuJoCo |
| :--- | ---: | ---: |
| velocity tracking error, vx (m/s) | 0.0600 ± 0.0085 | 0.1248 ± 0.0034 |
| velocity tracking error, vy (m/s) | 0.0539 ± 0.0020 | 0.0883 ± 0.0026 |
| velocity tracking error, wz (rad/s) | 0.1009 ± 0.0061 | 0.1941 ± 0.0050 |
| mean joint torque magnitude (N m) | 3.5586 ± 0.0724 | 2.9886 ± 0.0162 |
| falls per minute | 0.0612 ± 0.0433 | 0.0000 ± 0.0300 |
| falls | 2 | 0 |

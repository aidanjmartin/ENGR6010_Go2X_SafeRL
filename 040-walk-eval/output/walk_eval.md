Walking policy evaluation: 100 episodes of 20 s, seed 1000, a new velocity target every 5 s from vx [-0.5, 1.0], vy [-0.5, 0.5], wz [-1.0, 1.0]. Tracking error is the mean absolute error per episode, skipping 0.5 s after each new target. Mean and standard error over episodes.

| metric | mean | standard error |
| :--- | ---: | ---: |
| velocity tracking error, vx (m/s) | 0.0600 | 0.0085 |
| velocity tracking error, vy (m/s) | 0.0539 | 0.0020 |
| velocity tracking error, wz (rad/s) | 0.1009 | 0.0061 |
| mean joint torque magnitude (N m) | 3.5586 | 0.0724 |
| fraction of steps with a joint within 0.01 rad of a limit | 0.0094 | 0.0060 |
| contact fraction, FL_foot | 0.7692 | 0.0079 |
| contact fraction, FR_foot | 0.7891 | 0.0062 |
| contact fraction, RL_foot | 0.6978 | 0.0088 |
| contact fraction, RR_foot | 0.7139 | 0.0071 |
| falls per minute (2 falls in 32.7 robot-minutes) | 0.0612 | 0.0433 |

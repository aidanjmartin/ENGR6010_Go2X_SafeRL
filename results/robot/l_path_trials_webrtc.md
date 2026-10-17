L-path trials on the Go2 X, webrtc backend: forward 0.5 m at 0.2 m/s, turn left 90.0 deg at 0.5 rad/s, forward 0.5 m; closed loop on the robot's odometry and IMU. Values are the robot's own odometry and IMU, after a 1 s settle; they are not external ground truth.

| trial | leg 1 (m) | turn (deg) | leg 2 (m) | end forward (m) | end left (m) | heading change (deg) | stop reason |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 1 | 0.412 | 82.5 | 0.450 | +0.454 | +0.477 | +82.3 | end of script |
| 2 | 0.428 | 80.5 | 0.445 | +0.489 | +0.530 | +85.3 | end of script |
| 3 | 0.341 (timeout) | 80.5 | 0.456 | +0.425 | +0.521 | +82.7 | end of script |
| 4 | 0.294 (timeout) | 85.8 | 0.062 (timeout) | +0.286 | +0.127 | +85.4 | end of script |
| 5 | n/a | n/a | n/a | +nan | +nan | +nan | connection failed; nothing sent |
| mean ± sd | 0.369 ± 0.062 | 82.3 ± 2.5 | 0.353 ± 0.194 | +0.414 ± 0.089 | +0.414 ± 0.192 | +83.9 ± 1.7 | |

Ideal: legs 0.5 m, turn 90.0 deg, end +0.5 m forward and +0.5 m left of the start, heading change +90.0 deg. Sessions: 20261017T205303Z_07_l_path_t1_webrtc, 20261017T205359Z_07_l_path_t2_webrtc, 20261017T205454Z_07_l_path_t3_webrtc, 20261017T205548Z_07_l_path_t4_webrtc, 20261017T205644Z_07_l_path_t5_webrtc (data/, not committed).

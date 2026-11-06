"""List the cameras OpenCV can open (index, resolution), and save one frame from each, to find the USB webcam.
On a MacBook, index 0 is usually the built-in camera; the first run asks for camera permission for the terminal.

    python scripts/camera/list_cameras.py
"""
import cv2

for i in range(6):
    cap = cv2.VideoCapture(i)
    if not cap.isOpened():
        continue
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    ok, frame = False, None
    for _ in range(10):
        ok, frame = cap.read()
        if ok:
            break
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if ok:
        cv2.imwrite(f"camera_{i}.png", frame)
    print(f"camera {i}: {w} x {h} {'(frame saved to camera_%d.png)' % i if ok else '(opened, but no frame)'}")

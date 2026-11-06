"""Printable tags for the overhead camera (config/tags.yaml): one PDF, one US letter page per tag, at true size.

Print at 100 % ("actual size", no "fit to page"), then measure the black square of one printed tag and the
ChArUco board's square, and put them in config/tags.yaml (tag_size_m, charuco.square_m).

Pages: corner tags 0 to 3 (one per square corner), robot tag 10 (with a FRONT arrow), obstacle tags 20 to 24,
and the ChArUco board for lens calibration.

    python scripts/camera/make_tags.py --out tags.pdf
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from go2x.config import load_config  # noqa: E402
from go2x.vision.geometry import ARUCO_DICTS  # noqa: E402

DPI = 300
PAGE = (int(8.5 * DPI), int(11 * DPI))


def px(m):
    return int(round(m / 0.0254 * DPI))


def font(size):
    for name in ("DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def tag_page(tag_id, size_m, dictionary, title, lines, arrow=False):
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[dictionary])
    n = px(size_m)
    img = Image.new("L", PAGE, 255)
    tag = Image.fromarray(cv2.aruco.generateImageMarker(d, tag_id, n))
    x0, y0 = (PAGE[0] - n) // 2, (PAGE[1] - n) // 2
    img.paste(tag, (x0, y0))
    dr = ImageDraw.Draw(img)
    dr.text((PAGE[0] // 2, y0 - 200), title, fill=0, font=font(90), anchor="mm")
    if arrow:
        cx = PAGE[0] // 2
        dr.polygon([(cx, y0 - 130), (cx - 60, y0 - 40), (cx + 60, y0 - 40)], fill=0)
    for i, line in enumerate(lines):
        dr.text((PAGE[0] // 2, y0 + n + 90 + 70 * i), line, fill=0, font=font(55), anchor="mm")
    dr.text((PAGE[0] // 2, PAGE[1] - 90), f"{dictionary} id {tag_id}, black square {size_m * 100:.1f} cm: print at 100 %, "
            "then measure", fill=0, font=font(40), anchor="mm")
    return img


def charuco_page(c):
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[c["dictionary"]])
    board = cv2.aruco.CharucoBoard((c["squares_x"], c["squares_y"]), c["square_m"], c["marker_m"], d)
    w, h = px(c["squares_x"] * c["square_m"]), px(c["squares_y"] * c["square_m"])
    b = Image.fromarray(board.generateImage((w, h), marginSize=0))
    img = Image.new("L", PAGE, 255)
    img.paste(b, ((PAGE[0] - w) // 2, (PAGE[1] - h) // 2))
    dr = ImageDraw.Draw(img)
    dr.text((PAGE[0] // 2, 120), f"lens calibration board: {c['squares_x']} x {c['squares_y']} squares of "
            f"{c['square_m'] * 100:.1f} cm", fill=0, font=font(55), anchor="mm")
    dr.text((PAGE[0] // 2, PAGE[1] - 90), "tape flat to cardboard; measure one square and set charuco.square_m",
            fill=0, font=font(40), anchor="mm")
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="tags.pdf")
    args = p.parse_args()
    T = load_config("tags")
    s, dic = T["tag_size_m"], T["dictionary"]
    names = {(-1, -1): "-x, -y", (1, -1): "+x, -y", (1, 1): "+x, +y", (-1, 1): "-x, +y"}
    pages = []
    for tid, (x, y) in zip(T["corners"]["ids"], T["corners"]["xy"]):
        pages.append(tag_page(tid, s, dic, f"CORNER {tid}: square corner ({names[(int(np.sign(x)), int(np.sign(y)))]})",
                              ["flat on the floor, tag CENTER exactly on the square's corner",
                               "any rotation; keep it flat and unwrinkled"]))
    r = T["robot"]
    pages.append(tag_page(r["id"], s, dic, "ROBOT  -  FRONT", ["flat on the robot's back, centered left-right",
                                                             "arrow toward the robot's head",
                                                             "measure its height above the floor (robot standing)"],
                          arrow=True))
    for tid, o in T["obstacles"].items():
        what = f"{o.get('color', '')} {o['kind']}".strip()
        pages.append(tag_page(int(tid), s, dic, f"OBSTACLE: {what}  -  FRONT",
                              [f"flat on the top, centered (height {o['height_m']:.2f} m)", "optional: improves position "
                               "and heading over detection"], arrow=True))
    pages.append(charuco_page(T["charuco"]))
    pages[0].save(args.out, save_all=True, append_images=pages[1:], resolution=DPI)
    print(f"wrote {args.out}: {len(pages)} pages (corners {T['corners']['ids']}, robot {r['id']}, obstacles "
          f"{list(T['obstacles'])}, ChArUco board)")


if __name__ == "__main__":
    main()

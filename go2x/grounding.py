"""Task 5: prompts with ground-truth targets, detector output to a floor point, and the target mask.

The navigation policy never sees text. A prompt becomes a detector query; the best box's bottom center is
projected to the floor with the camera's floor homography; the target mask is a disk around that point.
"""
import numpy as np

from go2x.config import load_config


def make_prompts(objects, cfg=None):
    """Prompts for one scene. objects: list of dicts with id, class (chair, table, person) and color (chairs).

    Returns a list of dicts: prompt, query, target_id, template. A prompt that matches more than one object,
    or none, is skipped; the skipped prompts are returned separately with their match counts."""
    cfg = cfg or load_config("grounding")
    t, q = cfg["templates"], cfg["detector_queries"]
    out, skipped = [], []
    candidates = []
    colors = sorted({o["color"] for o in objects if o["class"] == "chair"})
    for c in colors:
        candidates.append(("chair_color", {"color": c}, lambda o, c=c: o["class"] == "chair" and o["color"] == c))
    candidates.append(("chair", {}, lambda o: o["class"] == "chair"))
    candidates.append(("table", {}, lambda o: o["class"] == "table"))
    candidates.append(("person", {}, lambda o: o["class"] == "person"))
    for template, fill, match in candidates:
        hits = [o for o in objects if match(o)]
        item = {"prompt": t[template].format(**fill), "query": q[template].format(**fill), "template": template,
                "matches": len(hits)}
        if len(hits) == 1:
            out.append({**item, "target_id": hits[0]["id"], "target_class": hits[0]["class"]})
        else:
            skipped.append(item)
    return out, skipped


def box_floor_point(box_xyxy, H_floor_to_image):
    """Bottom center of an image box (x0, y0, x1, y1), projected to the floor plane."""
    x0, y0, x1, y1 = box_xyxy
    uv = np.array([(x0 + x1) / 2, y1, 1.0])
    p = np.linalg.inv(np.asarray(H_floor_to_image)) @ uv
    return p[:2] / p[2]


def footprint_distance(point, obj):
    """Distance from a floor point to an object's footprint (0 inside)."""
    p = np.asarray(point, float)
    if "footprint_rect" in obj:
        cx, cy, yaw, hx, hy = obj["footprint_rect"]
        c, s = np.cos(yaw), np.sin(yaw)
        dx, dy = p[0] - cx, p[1] - cy
        qx, qy = c * dx + s * dy, -s * dx + c * dy
        return float(np.hypot(max(abs(qx) - hx, 0), max(abs(qy) - hy, 0)))
    cx, cy, r = obj["footprint_circle"]
    return float(max(np.hypot(p[0] - cx, p[1] - cy) - r, 0.0))


def footprint_center(obj):
    return np.asarray(obj["footprint_rect"][:2] if "footprint_rect" in obj else obj["footprint_circle"][:2], float)


def disk_mask(points, radii, td_cfg=None):
    """Draw disks (floor points, radii) into a size x size mask in the top-down image frame."""
    td = td_cfg or load_config("topdown")
    S, E = td["size"], td["half_extent"]
    res = 2 * E / S
    c = -E + (np.arange(S) + 0.5) * res
    xs, ys = np.meshgrid(c, c[::-1])
    m = np.zeros((S, S), bool)
    for (x, y), r in zip(points, radii):
        m |= np.hypot(xs - x, ys - y) <= r
    return m


def iou(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0

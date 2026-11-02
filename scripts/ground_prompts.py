"""Task 5: ground text prompts in the rendered camera images and measure the result.

For every rendered scene (job 045): make the prompts (go2x/grounding.py), run OWLv2 with each prompt's query,
take the best box, project its bottom center to the floor with the floor homography, and draw the target mask.
Also run class queries (chair, table, person) for every object and compare the resulting obstacle masks with
the drawn top-down image of the same scene.

    python scripts/ground_prompts.py --scenes $DATA_ROOT/045-render-scenes/scenes --out $RESULTS_DIR
"""
import argparse
import glob
import json
import math
import os

import numpy as np
import torch
from PIL import Image, ImageDraw

from go2x.config import load_config
from go2x.grounding import box_floor_point, disk_mask, footprint_center, footprint_distance, iou, make_prompts

parser = argparse.ArgumentParser()
parser.add_argument("--scenes", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--examples", type=int, default=4)
args = parser.parse_args()

from transformers import Owlv2ForObjectDetection, Owlv2Processor  # noqa: E402

G, TD = load_config("grounding"), load_config("topdown")
dc = G["detector"]
dev = "cuda" if torch.cuda.is_available() else "cpu"
processor = Owlv2Processor.from_pretrained(dc["model"], revision=dc["revision"])
model = Owlv2ForObjectDetection.from_pretrained(dc["model"], revision=dc["revision"]).to(dev).eval()
CLASS_QUERIES = {"chair": "a chair", "table": "a table", "person": "a person"}
radius = G["mask"]["radius"]


def nms(boxes, scores, thr):
    keep, order = [], np.argsort(-scores)
    while len(order):
        i = order[0]
        keep.append(i)
        b = boxes[order[1:]]
        x0, y0 = np.maximum(b[:, 0], boxes[i, 0]), np.maximum(b[:, 1], boxes[i, 1])
        x1, y1 = np.minimum(b[:, 2], boxes[i, 2]), np.minimum(b[:, 3], boxes[i, 3])
        inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
        area = lambda bb: (bb[..., 2] - bb[..., 0]) * (bb[..., 3] - bb[..., 1])  # noqa: E731
        order = order[1:][inter / (area(b) + area(boxes[i]) - inter + 1e-9) < thr]
    return keep


def detect(image, queries):
    """Boxes (xyxy, original pixels), scores and query index for each query, above the score threshold."""
    inputs = processor(text=[queries], images=image, return_tensors="pt").to(dev)
    with torch.no_grad():
        outputs = model(**inputs)
    side = max(image.size)  # OWLv2 pads to a square at the bottom and right
    res = processor.post_process_object_detection(outputs, threshold=dc["score_threshold"],
                                                  target_sizes=torch.tensor([[side, side]], device=dev))[0]
    return res["boxes"].cpu().numpy(), res["scores"].cpu().numpy(), res["labels"].cpu().numpy()


rows, mask_rows, examples = [], [], 0
os.makedirs(os.path.join(args.out, "grounding_examples"), exist_ok=True)
for gt_path in sorted(glob.glob(os.path.join(args.scenes, "scene_[0-9][0-9][0-9].json"))):
    gt = json.load(open(gt_path))
    image = Image.open(gt_path.replace(".json", ".png")).convert("RGB")
    H = np.array(gt["camera"]["H_floor_to_image"])
    objs = {o["id"]: o for o in gt["objects"]}
    prompts, skipped = make_prompts(gt["objects"])
    queries = [p["query"] for p in prompts] + list(CLASS_QUERIES.values())
    boxes, scores, labels = detect(image, queries)
    draw = ImageDraw.Draw(image)
    for qi, p in enumerate(prompts):
        sel = labels == qi
        target = objs[p["target_id"]]
        row = {"scene": gt["scene"], "prompt": p["prompt"], "target_id": p["target_id"], "target_class": p["target_class"]}
        if sel.any():
            k = np.flatnonzero(sel)[np.argmax(scores[sel])]
            pt = box_floor_point(boxes[k], H)
            d_fp = footprint_distance(pt, target)
            row.update(detected=True, score=float(scores[k]), floor_point=pt.tolist(), distance_to_footprint_m=d_fp,
                       distance_to_center_m=float(np.linalg.norm(pt - footprint_center(target))),
                       correct=d_fp <= G["evaluation"]["correct_distance"])
            pred = disk_mask([pt], [radius[p["target_class"]]], TD)
            row["target_mask_iou"] = iou(pred, np.load(gt_path.replace(".json", "_topdown.npy"))[3] > 0)
            if examples < args.examples:
                draw.rectangle(boxes[k].tolist(), outline=(255, 0, 255), width=3)
                draw.text((boxes[k][0], max(boxes[k][1] - 12, 0)), p["prompt"], fill=(255, 0, 255))
        else:
            row.update(detected=False, correct=False)
        rows.append(row)
    # obstacle masks from the rendered pipeline vs the drawn image
    top = np.load(gt_path.replace(".json", "_topdown.npy"))
    base = len(prompts)
    static_pts, static_r, people_pts, people_r = [], [], [], []
    for ci, cls in enumerate(CLASS_QUERIES):
        sel = np.flatnonzero(labels == base + ci)
        for k in [sel[i] for i in nms(boxes[sel], scores[sel], dc["nms_iou"])] if len(sel) else []:
            pt = box_floor_point(boxes[k], H)
            (people_pts if cls == "person" else static_pts).append(pt)
            (people_r if cls == "person" else static_r).append(radius[cls])
    m_static, m_people = disk_mask(static_pts, static_r, TD), disk_mask(people_pts, people_r, TD)
    mask_rows.append({"scene": gt["scene"], "static_iou": iou(m_static, top[0] > 0), "people_iou": iou(m_people, top[1] > 0),
                      "detections_static": len(static_pts), "detections_people": len(people_pts),
                      "true_static": sum(o["class"] != "person" for o in gt["objects"]),
                      "true_people": sum(o["class"] == "person" for o in gt["objects"])})
    if examples < args.examples and prompts:
        image.save(os.path.join(args.out, "grounding_examples", f"scene_{gt['scene']:03d}.jpg"), quality=85)
        examples += 1
    print(f"[ground] scene {gt['scene']}: {len(prompts)} prompts, {len(skipped)} skipped", flush=True)


def stat(v):
    v = np.asarray(v, float)
    return {"mean": float(v.mean()), "se": float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else None, "n": int(len(v))} if len(v) else None


det = [r for r in rows if r["detected"]]
summary = {
    "model": dc["model"], "revision": dc["revision"], "scenes": len(mask_rows), "prompts": len(rows),
    "correct_fraction": stat([r["correct"] for r in rows]),
    "detected_fraction": stat([r["detected"] for r in rows]),
    "floor_error_to_footprint_m": stat([r["distance_to_footprint_m"] for r in det]),
    "floor_error_to_center_m": stat([r["distance_to_center_m"] for r in det]),
    "target_mask_iou": stat([r["target_mask_iou"] for r in det]),
    "by_class": {c: stat([r["correct"] for r in rows if r["target_class"] == c]) for c in ("chair", "table", "person")},
    "static_mask_iou": stat([m["static_iou"] for m in mask_rows]),
    "people_mask_iou": stat([m["people_iou"] for m in mask_rows]),
    "settings": {"score_threshold": dc["score_threshold"], "correct_distance_m": G["evaluation"]["correct_distance"],
                 "mask_radius_m": radius},
}
with open(os.path.join(args.out, "grounding_eval.json"), "w") as f:
    json.dump({"summary": summary, "prompts": rows, "masks": mask_rows}, f, indent=1)
print(json.dumps(summary, indent=1))

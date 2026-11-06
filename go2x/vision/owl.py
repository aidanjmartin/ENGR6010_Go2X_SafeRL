"""OWLv2 open-vocabulary detector (config/grounding.yaml model and revision), as in scripts/ground_prompts.py.

The first use downloads the weights (about 0.6 GB) into the Hugging Face cache: do it once while the computer
still has internet (scripts/camera/preview.py --owl does), before joining the robot's Wi-Fi hotspot.
"""
import numpy as np
import torch
from PIL import Image

from go2x.config import load_config


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class OwlDetector:
    def __init__(self, device=None):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        dc = load_config("grounding")["detector"]
        self.cfg = load_config("tags")["owl"]
        self.dev = device or pick_device()
        self.processor = Owlv2Processor.from_pretrained(dc["model"], revision=dc["revision"])
        self.model = Owlv2ForObjectDetection.from_pretrained(dc["model"], revision=dc["revision"]).to(self.dev).eval()
        self.min_score = min(self.cfg["score_threshold"].values())

    @torch.no_grad()
    def detect(self, frame_rgb, queries):
        """Boxes (N, 4) xyxy in pixels, scores (N,), query index (N,), after per-query non-maximum suppression."""
        image = Image.fromarray(frame_rgb)
        inputs = self.processor(text=[queries], images=image, return_tensors="pt").to(self.dev)
        out = self.model(**inputs)
        side = max(image.size)          # OWLv2 pads to a square at the bottom and right
        res = self.processor.post_process_object_detection(out, threshold=self.min_score,
                                                           target_sizes=torch.tensor([[side, side]], device=self.dev))[0]
        boxes, scores, labels = (res[k].float().cpu().numpy() for k in ("boxes", "scores", "labels"))
        keep = []
        for q in range(len(queries)):
            idx = np.nonzero(labels == q)[0]
            keep += [idx[i] for i in nms(boxes[idx], scores[idx], self.cfg["nms_iou"])]
        keep = np.array(sorted(keep), int)
        return boxes[keep].reshape(-1, 4), scores[keep], labels[keep].astype(int)


def nms(boxes, scores, thr):
    keep, order = [], list(np.argsort(-scores))
    while order:
        i = order.pop(0)
        keep.append(i)
        rest = []
        for j in order:
            x0, y0 = max(boxes[i, 0], boxes[j, 0]), max(boxes[i, 1], boxes[j, 1])
            x1, y1 = min(boxes[i, 2], boxes[j, 2]), min(boxes[i, 3], boxes[j, 3])
            inter = max(x1 - x0, 0) * max(y1 - y0, 0)
            area = lambda b: (b[2] - b[0]) * (b[3] - b[1])  # noqa: E731
            if inter / (area(boxes[i]) + area(boxes[j]) - inter + 1e-9) < thr:
                rest.append(j)
        order = rest
    return keep

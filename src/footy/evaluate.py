"""Model-agnostic detection evaluation on YOLO-format labels.

Different models use different class lists (COCO has `person` and `sports ball`; ours has ball, goalkeeper,
player, referee). To compare them fairly, both are mapped onto shared *groups* before computing mAP:
    group 0 = ball, group 1 = person (any of player / goalkeeper / referee)
"""
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from supervision.metrics import MeanAveragePrecision

GROUP_NAMES = ["ball", "person"]
OURS_TO_GROUP = {0: 0, 1: 1, 2: 1, 3: 1}   # ball, goalkeeper, player, referee
COCO_TO_GROUP = {32: 0, 0: 1}              # sports ball, person


def load_yolo_labels(image_path, width, height):
    image_path = Path(image_path)
    label = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"   # .../images/x.jpg -> .../labels/x.txt
    rows = np.loadtxt(label, ndmin=2) if label.exists() and label.stat().st_size else np.zeros((0, 5))
    cx, cy, w, h = rows[:, 1] * width, rows[:, 2] * height, rows[:, 3] * width, rows[:, 4] * height
    return sv.Detections(xyxy=np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1).reshape(-1, 4),
                         class_id=rows[:, 0].astype(int))


def remap(det, mapping):
    keep = np.isin(det.class_id, list(mapping))
    det = det[keep]
    det.class_id = np.array([mapping[c] for c in det.class_id], dtype=int)
    return det


def grouped_map(model, image_paths, class_to_group, imgsz=1280, conf=0.001):
    """mAP of `model` on `image_paths`, with both predictions and labels mapped to GROUP_NAMES."""
    metric = MeanAveragePrecision()
    for path in image_paths:
        img = cv2.imread(str(path))
        h, w = img.shape[:2]
        pred = sv.Detections.from_ultralytics(model(img, imgsz=imgsz, conf=conf, verbose=False)[0])
        metric.update(remap(pred, class_to_group), remap(load_yolo_labels(path, w, h), OURS_TO_GROUP))
    result = metric.compute()
    per_group = {GROUP_NAMES[c]: round(float(ap[0]), 3) for c, ap in zip(result.matched_classes, result.ap_per_class)}
    return {"mAP50": round(float(result.map50), 3), "mAP50-95": round(float(result.map50_95), 3), "AP50_per_group": per_group}

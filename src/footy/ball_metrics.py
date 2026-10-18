"""Scoring point predictions of the ball (shared by BallNet and the YOLO detector, so the comparison is fair)."""
from __future__ import annotations

import math

import numpy as np

HIT_PX = 5.0   # correct if within 5 px of the true centre at 1280x720: less than one ball width

Pred = tuple[float, float, float] | None     # (x, y, confidence) or None
Point = tuple[float, float] | None


def score(preds: list[Pred], truths: list[Point], threshold: float = 0.0, hit_px: float = HIT_PX) -> dict:
    """Precision / recall / F1 over frames.

    Per frame: a prediction within `hit_px` of a visible ball is a true positive; any other prediction is a false
    positive; a visible ball without a correct prediction is a false negative (a far-off prediction counts as both).
    """
    tp = fp = fn = 0
    errors = []
    for p, t in zip(preds, truths, strict=True):
        if p is not None and p[2] < threshold:
            p = None
        dist = math.dist(p[:2], t) if p is not None and t is not None else None
        hit = dist is not None and dist <= hit_px
        if hit:
            errors.append(dist)
        tp += hit
        fp += p is not None and not hit
        fn += t is not None and not hit
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(2 * precision * recall / max(precision + recall, 1e-9), 4),
        "hit_median_err_px": round(float(np.median(errors)), 2) if errors else None,
        "frames": len(preds),
        "frames_with_ball": sum(t is not None for t in truths),
    }


def best_threshold(preds: list[Pred], truths: list[Point], grid=np.arange(0.05, 0.96, 0.05)) -> tuple[float, dict]:
    """Confidence threshold maximising F1 (choose it on validation, then apply it unchanged to test)."""
    results = {round(float(th), 2): score(preds, truths, th) for th in grid}
    th = max(results, key=lambda k: results[k]["f1"])
    return th, results[th]

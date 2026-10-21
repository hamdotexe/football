"""Scoring pitch keypoint predictions against our SoccerNet pitch labels.

Two levels:
- keypoints: is each visible landmark found within PCK_PX pixels (at 1280x720)?
- homography: fit pitch->image from the predicted keypoints, map the true landmark pixels back onto the pitch and
  measure how many metres off they land. This is what the minimap and the in/out-of-play rules depend on.
"""
import cv2
import numpy as np

from footy.pitch import KEYPOINT_XY

PCK_PX = 10.0
N_KEYPOINTS = len(KEYPOINT_XY)


def parse_label(line: str, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """One YOLO-pose label line ('0 cx cy w h' + 33 x 'x y v', normalised) -> (33x2 pixel coords, 33 visibility)."""
    v = np.array(line.split(), dtype=np.float64)
    if v.size != 5 + 3 * N_KEYPOINTS:
        raise ValueError(f"expected {5 + 3 * N_KEYPOINTS} values in a pitch label, got {v.size}")
    kp = v[5:].reshape(N_KEYPOINTS, 3)
    return kp[:, :2] * (width, height), kp[:, 2]


def fit_homography(xy: np.ndarray, conf: np.ndarray, conf_threshold: float = 0.5, ransac_px: float = 10.0):
    """Pitch (metres) -> image (pixels) homography from confident keypoints; RANSAC threshold in pixels. None if < 4."""
    keep = conf >= conf_threshold
    if keep.sum() < 4:
        return None
    H, _ = cv2.findHomography(KEYPOINT_XY[keep], np.float64(xy)[keep], cv2.RANSAC, ransac_px)
    return H


def frame_scores(pred_xy, pred_conf, gt_xy, gt_vis, conf_threshold: float = 0.5) -> dict:
    vis = np.asarray(gt_vis) > 0
    detected = np.asarray(pred_conf) >= conf_threshold
    err = np.linalg.norm(np.float64(pred_xy) - gt_xy, axis=1)
    H = fit_homography(pred_xy, pred_conf, conf_threshold)
    pitch_err = None
    if H is not None and vis.any():
        try:
            to_pitch = cv2.perspectiveTransform(np.float64(gt_xy)[vis].reshape(-1, 1, 2), np.linalg.inv(H)).reshape(-1, 2)
            pitch_err = float(np.mean(np.linalg.norm(to_pitch - KEYPOINT_XY[vis], axis=1)))
        except np.linalg.LinAlgError:
            pass
    return {"visible": int(vis.sum()), "detected": int(detected.sum()),
            "hits": int((vis & detected & (err <= PCK_PX)).sum()),
            "errors_px": err[vis & detected].round(2).tolist(), "pitch_err_m": pitch_err}


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    pitch = np.array([r["pitch_err_m"] if r["pitch_err_m"] is not None else np.inf for r in rows])
    visible, detected, hits = (sum(r[k] for r in rows) for k in ("visible", "detected", "hits"))
    errors = [e for r in rows for e in r["errors_px"]]
    return {
        "frames": n,
        "homography_found": round(float(np.isfinite(pitch).mean()), 4),
        "pitch_error_m_median": round(float(np.median(pitch)), 2) if np.isfinite(np.median(pitch)) else None,
        "frames_within_1m": round(float((pitch <= 1).mean()), 4),
        "frames_within_2m": round(float((pitch <= 2).mean()), 4),
        "frames_within_5m": round(float((pitch <= 5).mean()), 4),
        f"keypoint_recall_at_{PCK_PX:.0f}px": round(hits / max(visible, 1), 4),
        f"keypoint_precision_at_{PCK_PX:.0f}px": round(hits / max(detected, 1), 4),
        "keypoint_error_px_median": round(float(np.median(errors)), 2) if errors else None,
    }

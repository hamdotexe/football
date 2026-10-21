"""Score a pitch keypoint model (ours, or Roboflow's pretrained one) on the SoccerNet pitch test labels.

    python scripts/eval_pitch_keypoints.py --weights football-pitch-detection.pt --mapping roboflow \
        --images .../soccernet_yolo/test/images --labels .../test/labels --out outputs/pitch_eval

Writes <out>/metrics.json, per-frame rows (frames.csv), examples.jpg and eval.log. Fail-fast: any failed check logs
the reason and exits with an error; a time budget stops the scoring early and marks the result as partial.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from footy.pitch import KEYPOINT_NAMES, pitch_polylines  # noqa: E402
from footy.pitch_metrics import fit_homography, frame_scores, parse_label, summarise  # noqa: E402

log = logging.getLogger("pitch_eval")


class RunError(RuntimeError):
    pass


def check(ok: bool, message: str) -> None:
    if not ok:
        raise RunError(message)


def predict(model, img: np.ndarray, mapping: str, imgsz: int) -> tuple[np.ndarray, np.ndarray]:
    """Best pitch instance -> (33x2 xy, 33 conf) in our keypoint layout; all-zero confidence if nothing found."""
    r = model(img, imgsz=imgsz, verbose=False)[0]
    if r.keypoints is None or len(r.boxes) == 0:
        return np.zeros((len(KEYPOINT_NAMES), 2)), np.zeros(len(KEYPOINT_NAMES))
    best = int(r.boxes.conf.argmax())
    xy, conf = r.keypoints.xy[best].cpu().numpy(), r.keypoints.conf[best].cpu().numpy()
    if mapping == "roboflow":
        from footy.roboflow_pitch import to_our_keypoints
        return to_our_keypoints(xy, conf)
    return xy, conf


def draw_example(img, pred_xy, pred_conf, gt_xy, gt_vis, conf_threshold):
    """Green = true keypoints, red = predicted, yellow = pitch lines projected through the predicted homography."""
    vis = img.copy()
    H = fit_homography(pred_xy, pred_conf, conf_threshold)
    if H is not None:
        for line in pitch_polylines():
            pts = cv2.perspectiveTransform(np.float64(line).reshape(-1, 1, 2), H).reshape(-1, 2)
            if np.all(np.isfinite(pts)) and np.abs(pts).max() < 1e5:
                cv2.polylines(vis, [pts.astype(np.int32)], False, (0, 255, 255), 2)
    for (x, y), v in zip(gt_xy, gt_vis):
        if v > 0:
            cv2.circle(vis, (int(x), int(y)), 7, (0, 200, 0), 2)
    for (x, y), c in zip(pred_xy, pred_conf):
        if c >= conf_threshold:
            cv2.drawMarker(vis, (int(x), int(y)), (0, 0, 255), cv2.MARKER_CROSS, 12, 2)
    return cv2.resize(vis, (640, 360))


def run(a) -> None:
    out = Path(a.out)
    deadline = time.time() + a.max_minutes * 60
    check(Path(a.weights).is_file(), f"weights not found: {a.weights}")
    labels = sorted(Path(a.labels).glob("*.txt"))
    check(len(labels) > 0, f"no label files in {a.labels}")
    labels = labels[::a.stride]
    images = {p.stem: p for p in Path(a.images).glob("*.jpg")}
    have = [lab for lab in labels if lab.stem in images]
    check(len(have) >= 0.95 * len(labels), f"only {len(have)} of {len(labels)} labelled frames have an image in {a.images}")
    check(len(have) >= a.min_frames, f"only {len(have)} frames to score (need {a.min_frames})")
    log.info("%d frames to score (every %d of %d labelled test frames)", len(have), a.stride, len(labels) * a.stride)

    from ultralytics import YOLO
    model = YOLO(a.weights)
    n_kp = model.model.yaml.get("kpt_shape", [0])[0]
    expected = 32 if a.mapping == "roboflow" else len(KEYPOINT_NAMES)
    check(n_kp == expected, f"model has {n_kp} keypoints, mapping '{a.mapping}' expects {expected}")

    rows, examples, t0 = [], [], time.time()
    example_every = max(1, len(have) // 6)
    for i, lab in enumerate(have):
        if time.time() > deadline:
            log.warning("time budget of %.0f min reached: scored %d of %d frames", a.max_minutes, i, len(have))
            break
        img = cv2.imread(str(images[lab.stem]))
        check(img is not None, f"unreadable image {images[lab.stem]}")
        h, w = img.shape[:2]
        check(w == 1280, f"{lab.stem}: image is {w}x{h}, pixel thresholds assume 1280 wide")
        gt_xy, gt_vis = parse_label(lab.read_text().splitlines()[0], w, h)
        pred_xy, pred_conf = predict(model, img, a.mapping, a.imgsz)
        rows.append({"frame": lab.stem, **frame_scores(pred_xy, pred_conf, gt_xy, gt_vis, a.conf)})
        if i % example_every == 0 and len(examples) < 6:
            examples.append(draw_example(img, pred_xy, pred_conf, gt_xy, gt_vis, a.conf))
        if i + 1 == 100:   # sanity: a mapping or label-format bug shows up as homographies that are never found
            found = np.mean([r["pitch_err_m"] is not None for r in rows])
            check(found >= 0.1, f"homography found in only {found:.0%} of the first 100 frames: mapping or labels wrong?")
        if (i + 1) % 100 == 0:
            rate = (i + 1) / (time.time() - t0)
            log.info("%d/%d frames | %.2f frames/s | ETA %.0f min | so far %s", i + 1, len(have), rate,
                     (len(have) - i - 1) / rate / 60, json.dumps(summarise(rows)))

    check(len(rows) >= a.min_frames, f"only {len(rows)} frames scored before the time budget ran out")
    result = {"weights": Path(a.weights).name, "mapping": a.mapping, "imgsz": a.imgsz, "conf_threshold": a.conf,
              "complete": len(rows) == len(have), **summarise(rows)}
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    with open(out / "frames.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "visible", "detected", "hits", "pitch_err_m"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    if examples:
        while len(examples) % 3:
            examples.append(np.zeros_like(examples[0]))
        grid = np.vstack([np.hstack(examples[k:k + 3]) for k in range(0, len(examples), 3)])
        cv2.imwrite(str(out / "examples.jpg"), grid)
    log.info("RESULT %s", json.dumps(result, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--mapping", choices=("roboflow", "ours"), required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=1, help="score every n-th labelled frame")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.5, help="keypoint confidence threshold")
    ap.add_argument("--max-minutes", type=float, default=30)
    ap.add_argument("--min-frames", type=int, default=50)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(out / "eval.log")):
        handler.setFormatter(fmt)
        log.addHandler(handler)
    log.setLevel(logging.INFO)
    try:
        run(a)
    except RunError as e:
        log.error("STOPPED: %s", e)
        return 1
    except Exception:
        log.exception("STOPPED: unexpected error")
        return 1
    log.info("finished OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

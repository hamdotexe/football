"""Pick the pitch detection threshold, keypoint confidence threshold and RANSAC tolerance of the homography on
VALIDATION predictions, then report the test score at that one choice (choosing on test would make the test score
look better than it is).

    python scripts/tune_pitch_conf.py --run session2 s2/val/predictions.npz s2/test/predictions.npz \
                                      --run session3 s3/val/predictions.npz s3/test/predictions.npz --out tune.json

The predictions.npz files come from eval_pitch_keypoints.py: re-scoring them needs no model and no GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from footy.pitch_metrics import frame_scores, summarise  # noqa: E402

BOX_CONFS = (0.01, 0.05, 0.1, 0.25)      # predictions must have been made with box_conf <= the lowest of these
CONFS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
RANSAC_PX = (5.0, 10.0, 20.0)


def score(npz_path, conf: float, ransac_px: float, box_conf: float = 0.25) -> dict:
    d = np.load(npz_path)
    found = d["box_conf"] >= box_conf if "box_conf" in d.files else np.ones(len(d["frames"]), bool)
    return summarise([frame_scores(xy, c * ok, gxy, gv, conf, ransac_px)          # no pitch detection: no keypoints
                      for xy, c, ok, gxy, gv in zip(d["pred_xy"], d["pred_conf"], found, d["gt_xy"], d["gt_vis"])])


def rank_key(s: dict) -> tuple:
    """Lower median minimap error first (frames without a homography count as infinitely wrong), then more frames
    within 2 m."""
    median = s["pitch_error_m_median"]
    return (median if median is not None else float("inf"), -s["frames_within_2m"])


def tune(runs: list[tuple[str, str, str]]) -> dict:
    grid = [{"run": name, "box_conf": b, "conf": c, "ransac_px": r, "val": score(val, c, r, b)}
            for name, val, _ in runs for b in BOX_CONFS for c in CONFS for r in RANSAC_PX]
    best = min(grid, key=lambda g: rank_key(g["val"]))
    test_npz = {name: test for name, _, test in runs}[best["run"]]
    return {"grid": grid, "chosen": {k: best[k] for k in ("run", "box_conf", "conf", "ransac_px")},
            "chosen_val": best["val"], "chosen_test": score(test_npz, best["conf"], best["ransac_px"], best["box_conf"]),
            "default_test": {name: score(test, 0.5, 10.0, 0.25) for name, _, test in runs}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", nargs=3, action="append", required=True, metavar=("NAME", "VAL_NPZ", "TEST_NPZ"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    result = tune([tuple(r) for r in a.run])
    Path(a.out).write_text(json.dumps(result, indent=2))
    print(f"{'run':10s} {'box':>5s} {'conf':>5s} {'ransac':>6s} | validation: median m, homography found, within 2 m")
    for g in sorted(result["grid"], key=lambda g: rank_key(g["val"]))[:10]:
        v = g["val"]
        print(f"{g['run']:10s} {g['box_conf']:5.2f} {g['conf']:5.1f} {g['ransac_px']:6.0f} | {v['pitch_error_m_median']}, "
              f"{v['homography_found']:.2f}, {v['frames_within_2m']:.2f}")
    print("chosen on validation:", result["chosen"])
    print("test at the chosen setting:", json.dumps(result["chosen_test"]))
    for name, s in result["default_test"].items():
        print(f"test at the old default (box 0.25, conf 0.5, 10 px), {name}:", json.dumps(s))
    return 0


if __name__ == "__main__":
    sys.exit(main())

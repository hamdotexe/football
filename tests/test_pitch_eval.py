"""Tests for pitch keypoint scoring and the Roboflow keypoint mapping. Run: python -m pytest tests -q"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from footy.pitch import KEYPOINT_NAMES, KEYPOINT_XY  # noqa: E402
from footy.pitch_metrics import frame_scores, parse_label, summarise  # noqa: E402
from footy.roboflow_pitch import OUR_INDEX, ROBOFLOW_TO_OURS, to_our_keypoints  # noqa: E402

ARC_Y = 7.312489316231511


def roboflow_vertices_in_our_frame():
    """Roboflow's `vertices` formulas (sports/configs/soccer.py) with real pitch sizes, moved to our frame
    (origin at the centre spot). Their #11/#12/#19/#20 use the 6-yard width; we map them to the arc points."""
    L, W, PBW, PBL, GBW, GBL, CCR, PSD = 105, 68, 40.32, 16.5, 18.32, 5.5, 9.15, 11
    v = [(0, 0), (0, (W - PBW) / 2), (0, (W - GBW) / 2), (0, (W + GBW) / 2), (0, (W + PBW) / 2), (0, W),
         (GBL, (W - GBW) / 2), (GBL, (W + GBW) / 2), (PSD, W / 2), (PBL, (W - PBW) / 2), (PBL, (W - GBW) / 2),
         (PBL, (W + GBW) / 2), (PBL, (W + PBW) / 2), (L / 2, 0), (L / 2, W / 2 - CCR), (L / 2, W / 2 + CCR), (L / 2, W),
         (L - PBL, (W - PBW) / 2), (L - PBL, (W - GBW) / 2), (L - PBL, (W + GBW) / 2), (L - PBL, (W + PBW) / 2),
         (L - PSD, W / 2), (L - GBL, (W - GBW) / 2), (L - GBL, (W + GBW) / 2), (L, 0), (L, (W - PBW) / 2),
         (L, (W - GBW) / 2), (L, (W + GBW) / 2), (L, (W + PBW) / 2), (L, W), (L / 2 - CCR, W / 2), (L / 2 + CCR, W / 2)]
    return np.array(v, dtype=float) - (L / 2, W / 2)


def test_roboflow_mapping_matches_their_geometry():
    assert len(ROBOFLOW_TO_OURS) == 32 and len(set(ROBOFLOW_TO_OURS)) == 32
    theirs = roboflow_vertices_in_our_frame()
    ours = KEYPOINT_XY[OUR_INDEX]
    for i, (t, o) in enumerate(zip(theirs, ours)):
        if i in (10, 11, 18, 19):            # arc meets box line: same x, |y| 7.31 instead of their 9.16
            assert abs(t[0] - o[0]) < 0.01 and abs(abs(o[1]) - ARC_Y) < 0.01 and np.sign(t[1]) == np.sign(o[1])
        else:
            assert np.allclose(t, o, atol=0.01), (i, ROBOFLOW_TO_OURS[i], t, o)


def test_to_our_keypoints_leaves_only_the_centre_spot_empty():
    xy, conf = to_our_keypoints(np.arange(64, dtype=float).reshape(32, 2), np.ones(32))
    missing = [KEYPOINT_NAMES[i] for i in np.flatnonzero(conf == 0)]
    assert missing == ["centre_spot"] and xy.shape == (33, 2)


def broadcast_view():
    """A camera-like pitch->image homography and the true keypoints it puts inside a 1280x720 frame."""
    H = cv2.getPerspectiveTransform(np.float32([[-52.5, -34], [52.5, -34], [52.5, 34], [-52.5, 34]]),
                                    np.float32([[300, 250], [980, 250], [1500, 700], [-220, 700]]))
    xy = cv2.perspectiveTransform(KEYPOINT_XY.reshape(-1, 1, 2), H).reshape(-1, 2)
    vis = ((xy[:, 0] >= 0) & (xy[:, 0] < 1280) & (xy[:, 1] >= 0) & (xy[:, 1] < 720)).astype(float) * 2
    return xy, vis


def test_perfect_prediction_scores_perfectly():
    xy, vis = broadcast_view()
    s = frame_scores(xy, (vis > 0).astype(float), xy, vis)
    assert s["hits"] == s["visible"] > 10 and s["pitch_err_m"] < 1e-6


def test_small_pixel_noise_keeps_hits_and_small_pitch_error():
    xy, vis = broadcast_view()
    rng = np.random.default_rng(0)
    s = frame_scores(xy + rng.normal(0, 2, xy.shape), (vis > 0).astype(float), xy, vis)
    assert s["hits"] == s["visible"] and s["pitch_err_m"] < 0.5


def test_too_few_keypoints_gives_no_homography():
    xy, vis = broadcast_view()
    conf = np.zeros(len(xy))
    conf[np.flatnonzero(vis)[:3]] = 1
    assert frame_scores(xy, conf, xy, vis)["pitch_err_m"] is None


def test_parse_label_checks_length_and_scales():
    line = "0 0.5 0.5 1 1 " + " ".join(["0.5 0.25 2"] * 33)
    xy, vis = parse_label(line, 1280, 720)
    assert np.allclose(xy[0], (640, 180)) and vis[0] == 2
    with pytest.raises(ValueError):
        parse_label("0 0.5 0.5 1 1 0.1 0.2 2", 1280, 720)


def test_summary_counts_missing_homographies_as_failures():
    rows = [{"visible": 10, "detected": 10, "hits": 9, "errors_px": [1.0], "pitch_err_m": 0.5},
            {"visible": 10, "detected": 0, "hits": 0, "errors_px": [], "pitch_err_m": None}]
    s = summarise(rows)
    assert s["homography_found"] == 0.5 and s["frames_within_1m"] == 0.5 and s["keypoint_recall_at_10px"] == 0.45


def test_tuning_picks_the_threshold_on_validation_and_scores_test_once(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import tune_pitch_conf as tune
    gt = KEYPOINT_XY * (10, 8) + (640, 360)                   # every landmark inside a 1280x720 frame
    n = len(KEYPOINT_XY)
    for split in ("val", "test"):         # exact landmarks, but the model is unsure of the pitch (0.1) and landmarks (0.35)
        np.savez(tmp_path / f"{split}.npz", frames=np.array(["a", "b"]), pred_xy=np.stack([gt, gt]),
                 pred_conf=np.full((2, n), 0.35), box_conf=np.array([0.1, 0.1]), gt_xy=np.stack([gt, gt]),
                 gt_vis=np.full((2, n), 2.0))
    r = tune.tune([("s", str(tmp_path / "val.npz"), str(tmp_path / "test.npz"))])
    assert r["chosen"]["box_conf"] <= 0.1 and r["chosen"]["conf"] <= 0.3
    assert r["chosen_test"]["pitch_error_m_median"] == 0.0
    assert r["default_test"]["s"]["homography_found"] == 0.0   # at the defaults neither the pitch nor a landmark counts
    assert tune.score(tmp_path / "val.npz", 0.3, 10.0, 0.25)["homography_found"] == 0.0     # pitch below 0.25

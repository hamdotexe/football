"""Tests for merging the two ball models and reading frame triplets. Run: python -m pytest tests -q"""
import sys
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from footy.ball import fuse_balls, sampled_triplets, triplets  # noqa: E402


def ball_at(*xyc):
    return sv.Detections(xyxy=np.array([[x - 5, y - 5, x + 5, y + 5] for x, y, _ in xyc], float),
                         confidence=np.array([c for _, _, c in xyc]), class_id=np.zeros(len(xyc), int),
                         data={"class_name": np.array(["ball"] * len(xyc))})


def test_two_models_on_the_same_ball_become_one_stronger_candidate():
    fused = fuse_balls(ball_at((100, 100, 0.6)), ball_at((108, 104, 0.5)))
    assert len(fused) == 1 and abs(fused.confidence[0] - 0.8) < 1e-9          # 1 - 0.4 * 0.5


def test_different_balls_stay_separate_and_one_model_alone_is_kept():
    assert len(fuse_balls(ball_at((100, 100, 0.6)), ball_at((400, 300, 0.5)))) == 2
    assert len(fuse_balls(ball_at((100, 100, 0.6)), sv.Detections.empty())) == 1
    assert len(fuse_balls(sv.Detections.empty(), ball_at((100, 100, 0.6)))) == 1


def test_triplets_pad_the_ends_with_the_frame_itself():
    assert list(triplets([1, 2, 3])) == [(1, 1, 2), (1, 2, 3), (2, 3, 3)]


def test_sampled_triplets_give_each_sampled_frame_with_its_true_neighbours(tmp_path):
    path = str(tmp_path / "n.avi")
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 25, (64, 48))
    for i in range(23):                                # frame i is a flat image of brightness 10 * i
        out.write(np.full((48, 64, 3), 10 * i, np.uint8))
    out.release()
    got = [(k, int(p.mean().round() / 10), int(c.mean().round() / 10), int(n.mean().round() / 10))
           for k, p, c, n in sampled_triplets(path, every=5)]
    assert got == [(0, 0, 0, 1), (5, 4, 5, 6), (10, 9, 10, 11), (15, 14, 15, 16), (20, 19, 20, 21)]

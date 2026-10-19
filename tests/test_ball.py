"""Unit tests for the ball model's building blocks. Run: python -m pytest tests -q"""
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from footy.ball_data import BallClips, Sample  # noqa: E402
from footy.ball_metrics import best_threshold, score  # noqa: E402
from footy.ballnet import BallNet, decode, focal_loss, gaussian_heatmap  # noqa: E402


def test_heatmap_peaks_at_ball_and_is_empty_without_ball():
    hm = gaussian_heatmap(72, 128, (40.4, 20.6))
    assert hm.max() == 1.0 and np.unravel_index(hm.argmax(), hm.shape) == (21, 40)
    assert gaussian_heatmap(72, 128, None).max() == 0.0
    assert gaussian_heatmap(72, 128, (500.0, 20.0)).max() == 0.0     # outside the image -> empty


def test_decode_finds_peak_with_subpixel_centroid():
    logits = torch.full((1, 1, 72, 128), -10.0)
    logits[0, 0, 30, 50] = 5.0
    logits[0, 0, 30, 51] = 5.0                       # two equal pixels -> centroid halfway between them
    (x, y, conf), = decode(logits, threshold=0.5)
    assert abs(x - 50.5) < 1e-4 and abs(y - 30) < 1e-4 and conf > 0.99
    assert decode(torch.full((1, 1, 8, 8), -10.0), threshold=0.5) == [None]


def test_focal_loss_prefers_correct_heatmap():
    target = torch.from_numpy(gaussian_heatmap(32, 32, (10, 10)))[None, None]
    good = torch.logit(target.clamp(1e-4, 1 - 1e-4))
    bad = torch.zeros_like(good)
    assert focal_loss(good, target) < focal_loss(bad, target)


def test_model_shapes():
    out = BallNet()(torch.zeros(2, 9, 64, 96))
    assert out.shape == (2, 1, 64, 96)


def test_score_counts_hits_misses_and_far_predictions():
    truths = [(10, 10), (10, 10), None, (10, 10)]
    preds = [(12, 10, 0.9),      # hit (2 px)
             (30, 10, 0.9),      # far: false positive AND false negative
             (5, 5, 0.9),        # prediction without a ball: false positive
             None]               # missed: false negative
    s = score(preds, truths)
    assert (s["precision"], s["recall"]) == (round(1 / 3, 4), round(1 / 3, 4))
    assert score(preds, truths, threshold=0.95)["recall"] == 0.0


def test_best_threshold_picks_max_f1():
    truths = [(10, 10)] * 4 + [None] * 4
    preds = [(10, 10, 0.9)] * 4 + [(50, 50, 0.3)] * 4    # good predictions are confident, junk is not
    th, s = best_threshold(preds, truths)
    assert 0.3 < th <= 0.9 and s["f1"] == 1.0


@pytest.mark.parametrize("seed", range(20))
def test_augmentation_keeps_ball_label_on_the_ball(seed):
    """Random crops + mirroring must move the label with the pixels: draw a white dot, check it's at the label."""
    def frames(clip, n):
        img = np.zeros((720, 1280, 3), np.uint8)
        img[300, 900] = 255                              # the "ball" pixel
        return img
    random.seed(seed)
    ds = BallClips(frames, [Sample("c", 5, (900.0, 300.0))], train=True, crop=(384, 640), ball_crop_prob=1.0)
    x, heat, target = ds[0]
    assert target[2] == 1
    bx, by = int(round(target[0].item())), int(round(target[1].item()))
    assert x.dtype == torch.uint8
    assert x[3:6, by, bx].max() == 255                   # middle frame is bright exactly at the label
    assert heat[0, by, bx] == 1.0


def test_learning_check_cold_and_warm_start():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from train_ballnet import learning_check
    cold = [3.2] * 20 + [2.0] * 280                     # from scratch: big drop -> fine
    assert learning_check(cold, warm_start=False)[0]
    warm = [2.0] * 20 + [1.8] * 280                     # run 2's real numbers: warm start, small drop -> must pass
    assert not learning_check(warm, warm_start=False)[0]
    assert learning_check(warm, warm_start=True)[0]
    blown = [2.0] * 20 + [3.5] * 280                    # warm start that diverges -> stop
    assert not learning_check(blown, warm_start=True)[0]

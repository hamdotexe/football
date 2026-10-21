"""Tests for the pitch training script's checks and its hard timeout. Run: python -m pytest tests -q"""
import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import train_pitch as tp  # noqa: E402

META = json.loads((Path(__file__).resolve().parents[1] / "data" / "pitch_pose" / "keypoints.json").read_text()) \
    if (Path(__file__).resolve().parents[1] / "data" / "pitch_pose" / "keypoints.json").exists() else None


@pytest.mark.skipif(META is None, reason="local pitch labels not present")
def test_real_flip_idx_passes_and_a_broken_one_fails():
    tp.check_flip_idx(META["keypoints"], META["flip_idx"])
    broken = list(META["flip_idx"])
    broken[0], broken[1] = broken[1], broken[0]              # L_corner_top no longer maps to R_corner_top
    with pytest.raises(tp.RunError):
        tp.check_flip_idx(META["keypoints"], broken)


def test_label_line_checks():
    good = "0 0.5 0.5 1 1 " + " ".join(["0.5 0.25 2"] * 33)
    tp.check_label_line(good, 33)
    with pytest.raises(tp.RunError, match="values"):
        tp.check_label_line("0 0.5 0.5 1 1 0.1 0.2 2", 33)
    with pytest.raises(tp.RunError, match="outside"):
        tp.check_label_line("0 0.5 0.5 1 1 " + " ".join(["1.5 0.25 2"] * 33), 33)
    with pytest.raises(tp.RunError, match="visibility"):
        tp.check_label_line("0 0.5 0.5 1 1 " + " ".join(["0.5 0.25 3"] * 33), 33)


def test_projected_epochs():
    # 397 frames took 120 s -> 7944 frames take ~2400 s -> 3 h fit 4.5 epochs
    assert abs(tp.projected_epochs(7944, 397, 120, 3.0) - 4.5) < 0.05


def test_read_results_tolerates_padding_and_blank_cells(tmp_path):
    p = tmp_path / "results.csv"
    p.write_text("   epoch,  train/pose_loss,metrics/mAP50(P)\n1, 2.5,\n2, 2.0,0.4\n")
    rows = tp.read_results(p)
    assert rows[1]["metrics/mAP50(P)"] == 0.4 and rows[0]["epoch"] == 1 and rows[0]["metrics/mAP50(P)"] != rows[0]["metrics/mAP50(P)"]


def test_hung_child_is_killed_at_the_timeout(tmp_path):
    t = time.time()
    code, seconds = tp.run_child({"mode": "sleep", "seconds": 60}, tmp_path / "out.txt", timeout_s=3)
    assert code is None and time.time() - t < 20

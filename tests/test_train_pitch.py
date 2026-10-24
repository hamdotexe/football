"""Tests for the pitch training script's checks and its hard timeout. Run: python -m pytest tests -q"""
import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import train_pitch as tp  # noqa: E402

KEYPOINTS_JSON = Path(__file__).resolve().parents[1] / "data" / "pitch_pose" / "keypoints.json"
META = json.loads(KEYPOINTS_JSON.read_text()) if KEYPOINTS_JSON.exists() else None


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
    # 17.2 images/s -> 7944 frames in 462 s, +20% for validation -> 3 h fit ~19.5 epochs
    assert abs(tp.projected_epochs(7944, 17.2, 3.0) - 19.5) < 0.1


# real lines from run 2's preflight output (colour codes, carriage-return progress updates, the out-of-memory retry)
RUN2_LINES = (
    "\x1b[34m\x1b[1moptimizer:\x1b[0m AdamW(lr=0.002, momentum=0.9) with parameter groups\n"
    "WARNING ⚠️ CUDA out of memory with batch=16. Reducing to batch=8 and retrying (1/3).\n"
    "\x1b[K        1/1      7.58G      5.673      11.84     0.7041      51.61       4.94          4       1280: 93% "
    "━━━ 93/100 4.3it/s 25.9s<1.6s\r"
    "\x1b[K        1/1      7.62G      5.671      11.84     0.7042       51.9      4.954          1       1280: 99% "
    "━━━ 99/100 3.1it/s 31.8s<0.3s\n"
    "\x1b[K\x1b[34m\x1b[1mval: \x1b[0mScanning /tmp/pitch/valid/labels... 1007 images: 75% 1007/1326 92.0it/s 11.2s\n"
)


def test_scan_finds_the_memory_trouble_peak_memory_and_speed_of_run_2():
    seen = tp.scan_training_output(RUN2_LINES)
    assert seen["memory_trouble"] == ["out of memory", "reducing to batch"]
    assert seen["peak_gpu_gb"] == 7.62 and abs(seen["batches_per_s"] - 3.7) < 1e-9   # median of 4.3, 3.1; scan line ignored


def test_scan_of_a_clean_run_and_slow_progress():
    clean = "        3/12      5.10G      1.2      2.3      0.5      0.9      1.1          4       1280: 10% 50/497 1.6s/it 80s<12m\n"
    seen = tp.scan_training_output(clean)
    assert seen["memory_trouble"] == [] and seen["peak_gpu_gb"] == 5.1 and abs(seen["batches_per_s"] - 0.625) < 1e-9


RUN2_PREFLIGHT = Path(__file__).resolve().parents[1] / "outputs" / "pitch_run2" / "pitch" / "preflight_output.txt"


@pytest.mark.skipif(not RUN2_PREFLIGHT.exists(), reason="run 2 outputs not downloaded")
def test_scan_flags_the_real_run_2_preflight():
    seen = tp.scan_training_output(RUN2_PREFLIGHT.read_text(errors="replace"))
    assert seen["memory_trouble"] and seen["peak_gpu_gb"] > 7 and 4.0 <= seen["batches_per_s"] <= 4.5   # ran at ~4.3 it/s


def test_read_results_tolerates_padding_and_blank_cells(tmp_path):
    p = tmp_path / "results.csv"
    p.write_text("   epoch,  train/pose_loss,metrics/mAP50(P)\n1, 2.5,\n2, 2.0,0.4\n")
    rows = tp.read_results(p)
    assert rows[1]["metrics/mAP50(P)"] == 0.4 and rows[0]["epoch"] == 1 and rows[0]["metrics/mAP50(P)"] != rows[0]["metrics/mAP50(P)"]


def test_hung_child_is_killed_at_the_timeout(tmp_path):
    t = time.time()
    code, seconds = tp.run_child({"mode": "sleep", "seconds": 60}, tmp_path / "out.txt", timeout_s=3)
    assert code is None and time.time() - t < 20

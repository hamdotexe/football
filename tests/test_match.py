"""Tests for whole-match processing helpers. Run: python -m pytest tests -q"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from footy.match import check_deadline, half_states, score_at, true_score  # noqa: E402


class NoCalibrator:
    def reset(self, fps=None):
        pass


def test_the_time_budget_stops_with_an_error_only_once_it_has_passed():
    check_deadline(None, "anything")
    check_deadline(time.time() + 60, "anything")
    with pytest.raises(TimeoutError, match="half 2"):
        check_deadline(time.time() - 1, "half 2")


def test_a_half_past_the_budget_stops_before_analysing_a_frame(tmp_path):
    path = str(tmp_path / "h.avi")
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 25, (64, 48))
    for _ in range(12):
        out.write(np.zeros((48, 64, 3), np.uint8))
    out.release()
    with pytest.raises(TimeoutError, match="minute 0.0 of h.avi"):     # detector None: analysing would fail otherwise
        half_states(path, None, None, NoCalibrator(), deadline=time.time() - 1)


def test_score_counts_goals_before_the_time_and_the_real_score_comes_from_the_name():
    goals = [(1, 100.0, 0), (1, 900.0, 1), (2, 50.0, 0)]
    assert score_at(goals, 1, 500) == [1, 0] and score_at(goals, 2, 10) == [1, 1] and score_at(goals, 2, 1e9) == [2, 1]
    assert true_score("2017-03-08 - 22-45 Barcelona 6 - 1 Paris SG") == (6, 1) and true_score("no score") is None


def test_precision_accepts_labels_not_shown_and_recall_counts_only_shown_ones():
    from footy import events as ev
    from footy.match import evaluate
    labels = {"annotations": [{"label": "Kick-off", "gameTime": "1 - 00:00", "position": "0", "visibility": "not shown"},
                              {"label": "Kick-off", "gameTime": "1 - 10:00", "position": "600000", "visibility": "visible"},
                              {"label": "Kick-off", "gameTime": "1 - 20:00", "position": "1200000", "visibility": "visible"}]}
    found = {1: [ev.Event(1.0, ev.KICK_OFF, 0, 0), ev.Event(602.0, ev.KICK_OFF, 0, 0), ev.Event(900.0, ev.KICK_OFF, 0, 0)]}
    r = evaluate(found, labels)[ev.KICK_OFF]
    assert (r["matched_any"], r["matched"], r["precision"], r["recall"]) == (2, 1, round(2 / 3, 3), 0.5)


class StubAnnotator:
    """Stands in for the models: returns the frame, optionally slowly."""
    delay = 0.0

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, frame, prev=None, nxt=None):
        time.sleep(self.delay)
        return frame.copy()


def small_video(path, n=30):
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (320, 240))
    for i in range(n):
        out.write(np.full((240, 320, 3), 4 * i % 256, np.uint8))
    out.release()
    return str(path)


def test_a_half_is_rendered_in_parts_and_joined_whole(tmp_path, monkeypatch):
    import footy.match as M
    monkeypatch.setattr(M, "Annotator", StubAnnotator)
    video = small_video(tmp_path / "half.avi")
    out = M.render_half(video, 1, [], [], tmp_path / "1H.mp4", None, None, NoCalibrator(), part_s=0.4,
                        work=tmp_path / "work", log=lambda *_: None)
    assert int(cv2.VideoCapture(str(out)).get(cv2.CAP_PROP_FRAME_COUNT)) == 30      # 3 parts of 10 frames, joined


def test_past_the_budget_the_frames_rendered_so_far_are_still_saved(tmp_path, monkeypatch):
    import footy.match as M
    monkeypatch.setattr(M, "Annotator", type("Slow", (StubAnnotator,), {"delay": 0.05}))
    video = small_video(tmp_path / "half.avi", n=60)
    with pytest.raises(TimeoutError, match="stops at"):
        M.render_half(video, 1, [], [], tmp_path / "1H.mp4", None, None, NoCalibrator(), part_s=0.4,
                      deadline=time.time() + 0.6, work=tmp_path / "work", log=lambda *_: None)
    n = int(cv2.VideoCapture(str(tmp_path / "1H.mp4")).get(cv2.CAP_PROP_FRAME_COUNT))
    assert 3 <= n < 60

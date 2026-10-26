"""Tests for the ball trail: only physically sensible paths, moved with the camera. Run: python -m pytest tests -q"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from footy.ball import BallTrail, CameraMotion, fit_curve  # noqa: E402

rng = np.random.default_rng(0)


def line(t, start=(100.0, 300.0), v=(12.0, -3.0), noise=1.0):
    return np.array([[start[0] + v[0] * f, start[1] + v[1] * f] for f in t]) + rng.normal(0, noise, (len(t), 2))


def test_a_rolling_ball_gets_its_whole_recent_path():
    t = np.arange(20)
    fit = fit_curve(t, line(t), tol=4.0)
    assert fit is not None and fit[0] == 0


def test_a_ball_in_the_air_is_a_parabola_and_fits_too():
    t = np.arange(20)
    xy = np.stack([200 + 10 * t, 400 - 15 * t + 0.6 * t ** 2], 1) + rng.normal(0, 1.0, (20, 2))
    assert fit_curve(t, xy, tol=4.0, max_accel=3.0)[0] == 0


def test_a_zigzag_of_wrong_sightings_gets_no_trail():
    t = np.arange(20)
    xy = line(t) + np.array([[0, 30] if f % 2 else [0, -30] for f in t])
    assert fit_curve(t, xy, tol=4.0) is None


def test_a_touch_restarts_the_trail_where_the_ball_changed_direction():
    before, after = line(np.arange(12)), line(np.arange(9), start=(232.0, 267.0), v=(-4.0, 14.0))
    fit = fit_curve(np.arange(21), np.vstack([before, after]), tol=4.0)
    assert fit is not None and 11 <= fit[0] <= 12


def test_a_dribble_back_and_forth_is_no_hook_the_trail_starts_at_the_turn():
    t = np.arange(21)
    xy = np.stack([300 - 0.8 * (t - 10.0) ** 2, np.full(21, 360.0)], 1) + rng.normal(0, 0.5, (21, 2))
    fit = fit_curve(t, xy, tol=4.0)                           # one quadratic fits it all, but it turns back: no
    assert fit is not None and fit[0] >= 9


def test_one_false_sighting_does_not_break_the_trail():
    t = np.arange(20)
    xy = line(t)
    xy[9] += (60, -40)                                        # a sock, a head: one frame
    assert fit_curve(t, xy, tol=4.0)[0] == 0


def test_the_trail_moves_with_the_camera_and_a_cut_clears_it():
    trail = BallTrail(fps=25, width=1280)
    pan = np.array([[1, 0, -15.0], [0, 1, 0], [0, 0, 1]])   # the camera pans right: the scene moves 15 px left
    for f in range(15):                                       # the ball rolls right at 10 px/frame on the pitch
        trail.update(pan if f else None, (600.0 + 10 * f - 15 * f, 360.0))
    path = trail.path()
    # on screen the ball drifts left (600 -> 530), but moved with the camera its path runs right: 390 -> 530, as on
    # the pitch; without the camera it would be drawn backwards
    assert path is not None and abs(path[0][0] - 390) < 3 and abs(path[-1][0] - 530) < 3
    trail.update(None, (100.0, 100.0))                        # a cut: only the ball until a new path builds up
    assert trail.path() is None


def test_camera_motion_recovers_a_pan():
    tex = cv2.GaussianBlur((rng.random((720, 1280)) * 255).astype(np.uint8), (0, 0), 3)
    frame = cv2.cvtColor(cv2.normalize(tex, None, 0, 255, cv2.NORM_MINMAX), cv2.COLOR_GRAY2BGR)
    shifted = cv2.warpAffine(frame, np.float32([[1, 0, 9], [0, 1, -4]]), (1280, 720), borderMode=cv2.BORDER_REFLECT)
    motion = CameraMotion()
    assert motion(frame) is None                              # nothing to compare the first frame with
    H = motion(shifted)
    assert H is not None and np.allclose(H[:2, 2], (9, -4), atol=1.0) and np.allclose(H[:2, :2], np.eye(2), atol=0.01)

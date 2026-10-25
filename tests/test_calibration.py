"""Tests for keypoints -> homography and the steady homography used on video. Run: python -m pytest tests -q"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from footy.calibration import SteadyHomography  # noqa: E402
from footy.pitch import HL, HW, KEYPOINT_XY, homography_from_keypoints, project  # noqa: E402

W, H = 1280, 720
# a broadcast-like view: the far touchline is short and high in the frame, the near one long and low
PITCH_TO_IMG = cv2.getPerspectiveTransform(np.float32([(-HL, -HW), (HL, -HW), (HL, HW), (-HL, HW)]),
                                           np.float32([(200, 170), (1080, 170), (1270, 700), (10, 700)]))
IMG_TO_PITCH = np.linalg.inv(PITCH_TO_IMG)


def shifted(dx):
    """The same camera moved dx metres along the pitch."""
    return np.array([[1, 0, dx], [0, 1, 0], [0, 0, 1]], float) @ IMG_TO_PITCH


def centre_on_pitch(Hm):
    return project(Hm, [(W / 2, 0.7 * H)])[0]


def test_keypoints_give_the_true_homography_and_a_100_px_outlier_is_ignored():
    px = project(PITCH_TO_IMG, KEYPOINT_XY)
    px[0] += 100                                              # one landmark badly misplaced
    Hm, inliers = homography_from_keypoints(px, np.ones(len(px)))
    assert inliers == len(px) - 1
    truth = project(PITCH_TO_IMG, KEYPOINT_XY[1:])
    assert np.abs(project(Hm, truth) - KEYPOINT_XY[1:]).max() < 0.01


def test_too_few_confident_keypoints_give_no_homography():
    conf = np.zeros(len(KEYPOINT_XY))
    conf[:3] = 0.9
    assert homography_from_keypoints(project(PITCH_TO_IMG, KEYPOINT_XY), conf) == (None, 0)


def test_steady_holds_the_last_view_for_a_while_then_gives_up():
    s = SteadyHomography(W, H, hold=3)
    s.update(IMG_TO_PITCH)
    for _ in range(3):
        assert np.allclose(centre_on_pitch(s.update(None)), centre_on_pitch(IMG_TO_PITCH), atol=0.01)
    assert s.update(None) is None


def test_steady_ignores_a_one_frame_jump_but_follows_a_cut():
    s = SteadyHomography(W, H)
    s.update(IMG_TO_PITCH)
    x0 = centre_on_pitch(IMG_TO_PITCH)[0]
    assert abs(centre_on_pitch(s.update(shifted(30)))[0] - x0) < 0.01      # one wild frame: ignored
    assert abs(centre_on_pitch(s.update(IMG_TO_PITCH))[0] - x0) < 0.01
    s.update(shifted(30))
    assert abs(centre_on_pitch(s.update(shifted(30)))[0] - (x0 + 30)) < 0.01   # two agreeing frames: a cut


def test_steady_smooths_small_moves_and_rejects_garbage():
    s = SteadyHomography(W, H, alpha=0.5)
    s.update(IMG_TO_PITCH)
    x0 = centre_on_pitch(IMG_TO_PITCH)[0]
    assert abs(centre_on_pitch(s.update(shifted(2)))[0] - (x0 + 1)) < 0.01          # halfway to the new view
    sky = np.array([[1, 0, 500], [0, 1, 0], [0, 0, 1]], float) @ IMG_TO_PITCH        # puts the grass 500 m away
    assert abs(centre_on_pitch(s.update(sky))[0] - (x0 + 1)) < 0.01

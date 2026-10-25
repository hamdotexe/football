"""Video calibration: the pitch keypoint model gives an image->pitch homography per frame; SteadyHomography keeps it
steady over time (single frames are missed or wrong, the broadcast camera itself moves smoothly)."""
import cv2
import numpy as np

from footy.pitch import HL, HW, homography_from_keypoints, project

# chosen on the held-out validation clips (scripts/tune_pitch_conf.py): every broadcast frame shows the pitch, so the
# pitch detection is kept however unsure; landmarks count from 0.2; RANSAC tolerance 20 px at 1280 px wide
BOX_CONF, KEYPOINT_CONF, RANSAC_PX_AT_1280 = 0.01, 0.2, 20.0


class SteadyHomography:
    """Per-frame image->pitch homographies (or None) -> a steady one.

    A view is summarised by where a grid of image points (lower part of the frame, where the grass is) lands on the
    pitch. Then:
    - smoothing: the shown view is an exponential moving average of those points (`alpha` = weight of the new frame);
    - jumps: an estimate whose view is more than `max_jump_m` (median over the grid) from the shown one is held back;
      if the next estimate agrees with it, it was a camera cut (e.g. to a replay) and becomes the view at once;
    - gaps: a frame without an estimate keeps the last view for up to `hold` frames, then there is no view;
    - an estimate that puts the grid off the pitch is treated as no estimate.
    """

    def __init__(self, width, height, hold=25, max_jump_m=5.0, alpha=0.5):
        xs, ys = np.meshgrid(np.linspace(0.1, 0.9, 5) * width, np.linspace(0.45, 0.95, 4) * height)
        self.grid = np.stack([xs.ravel(), ys.ravel()], 1)
        self.hold, self.max_jump_m, self.alpha = hold, max_jump_m, alpha
        self.view, self.pending, self.missed = None, None, 0

    def _view_of(self, H):
        if H is None:
            return None
        with np.errstate(all="ignore"):
            v = project(H, self.grid)
        if not np.all(np.isfinite(v)):
            return None
        centre = np.median(v, axis=0)
        if abs(centre[0]) > HL + 10 or abs(centre[1]) > HW + 10:
            return None
        return v

    @staticmethod
    def _shift(a, b):
        return float(np.median(np.linalg.norm(a - b, axis=1)))

    def update(self, H):
        """Feed this frame's estimate (image->pitch, or None); returns the homography to use, or None."""
        new = self._view_of(H)
        if new is not None and (self.view is None or self._shift(new, self.view) <= self.max_jump_m):
            self.view = new if self.view is None else self.alpha * new + (1 - self.alpha) * self.view
            self.pending, self.missed = None, 0
        elif new is not None and self.pending is not None and self._shift(new, self.pending) <= self.max_jump_m:
            self.view, self.pending, self.missed = new, None, 0          # two frames agree: a cut, not an outlier
        else:                                                           # no estimate, or a jump nothing confirms yet
            self.pending = new if new is not None else self.pending
            self.missed += 1
            if self.missed > self.hold:
                self.view = self.pending = None
        return None if self.view is None else cv2.findHomography(self.grid, self.view)[0]


class PitchCalibrator:
    """Frame -> steady image->pitch homography (or None), using the pitch keypoint model."""

    def __init__(self, weights, fps=25, imgsz=1280, device=None, half=False):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.fps, self.imgsz, self.device, self.half = fps, imgsz, device, half
        self.reset()

    def reset(self, fps=None):
        """Start a new video (at `fps` analysed frames per second if given): forget the view, zero the counts."""
        self.fps = fps or self.fps
        self.steady = None
        self.frames = self.estimated = self.shown = 0     # frames seen / with their own homography / with a minimap

    def estimate(self, frame):
        """This frame's own homography (no smoothing), or None."""
        fp16 = {"quantize": 16} if self.half else {}       # Ultralytics >= 8.4: `half` is deprecated (warns each call)
        r = self.model(frame, imgsz=self.imgsz, conf=BOX_CONF, device=self.device, verbose=False, **fp16)[0]
        if r.keypoints is None or len(r.boxes) == 0:
            return None
        best = int(r.boxes.conf.argmax())
        H, _ = homography_from_keypoints(r.keypoints.xy[best].cpu().numpy(), r.keypoints.conf[best].cpu().numpy(),
                                         KEYPOINT_CONF, RANSAC_PX_AT_1280 * frame.shape[1] / 1280)
        return H

    def __call__(self, frame):
        if self.steady is None:
            self.steady = SteadyHomography(frame.shape[1], frame.shape[0], hold=int(self.fps))
        H = self.estimate(frame)
        shown = self.steady.update(H)
        self.frames += 1
        self.estimated += H is not None
        self.shown += shown is not None
        return shown

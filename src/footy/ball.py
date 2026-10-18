"""Physically plausible ball selection: reject detections the ball couldn't have reached ("the ball can't teleport")."""
import numpy as np
import supervision as sv


class BallTracker:
    def __init__(self, max_px_per_frame=60.0, slack_px=40.0, reacquire_after=25):
        """
        max_px_per_frame: fastest plausible ball movement on screen (a hard shot in a wide 1080p shot is ~30-60 px/frame)
        slack_px:         extra tolerance for detection jitter
        reacquire_after:  frames without an accepted detection before any detection is trusted again (1 s at 25 fps)
        """
        self.max_px, self.slack, self.reacquire_after = max_px_per_frame, slack_px, reacquire_after
        self.last_xy, self.frames_since = None, 10 ** 9
        self.segment = 0   # increments when the ball is re-acquired, so the drawn trail restarts instead of jumping

    def update(self, balls: sv.Detections) -> sv.Detections:
        """Return at most one ball detection: the plausible one, or nothing."""
        self.frames_since += 1
        if len(balls) == 0:
            return balls
        xy = balls.get_anchors_coordinates(sv.Position.CENTER)
        if self.last_xy is None or self.frames_since > self.reacquire_after:
            best = int(np.argmax(balls.confidence))           # lost: trust the most confident detection
            self.segment += 1
        else:
            dist = np.linalg.norm(xy - self.last_xy, axis=1)
            ok = dist <= self.max_px * self.frames_since + self.slack
            if not ok.any():
                return balls[[]]
            best = int(np.flatnonzero(ok)[np.argmax(balls.confidence[ok])])
        self.last_xy, self.frames_since = xy[best], 0
        ball = balls[[best]]
        ball.tracker_id = np.array([self.segment])
        return ball

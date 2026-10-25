"""The ball: BallNet-L (3 frames) and the detector (1 frame) merged, then a physically plausible pick ("the ball
can't teleport")."""
import cv2
import numpy as np
import supervision as sv

BALL = 0   # class id of the detector


class BallNetDetector:
    """BallNet-L on frames t-1, t, t+1 -> the ball in frame t as one detection (or none), in the frame's pixels.

    It sees motion, so it finds blurred balls a single frame hides (test recall 0.67 vs 0.64 for the detector)."""

    def __init__(self, weights, device="cpu", half=False):
        import torch
        from footy.ballnet import load_ballnet
        self.torch = torch
        self.model, ckpt = load_ballnet(weights, device)
        self.threshold = float(ckpt.get("threshold", 0.3))     # chosen on validation clips when it was trained
        self.device, self.half = device, half and str(device) != "cpu"

    def __call__(self, prev, cur, nxt) -> sv.Detections:
        from footy.ballnet import decode
        torch = self.torch
        h, w = cur.shape[:2]
        frames = [f if f.shape[:2] == (720, 1280) else cv2.resize(f, (1280, 720), interpolation=cv2.INTER_AREA)
                  for f in (prev, cur, nxt)]                    # trained on 1280x720 BGR frames
        x = torch.from_numpy(np.ascontiguousarray(np.concatenate(frames, axis=2).transpose(2, 0, 1)))[None]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=self.half):
            p = decode(self.model(x.to(self.device).float().div_(255)), self.threshold)[0]
        if p is None:
            return sv.Detections.empty()
        cx, cy, r = p[0] * w / 1280, p[1] * h / 720, 6 * w / 1280
        return sv.Detections(xyxy=np.array([[cx - r, cy - r, cx + r, cy + r]]), confidence=np.array([p[2]]),
                             class_id=np.array([BALL]), data={"class_name": np.array(["ball"])})  # same fields as YOLO's


def fuse_balls(a: sv.Detections, b: sv.Detections, agree_px=20.0) -> sv.Detections:
    """Ball candidates from two models. Where both see the same ball (centres within agree_px), one candidate with
    the combined confidence 1 - (1 - p)(1 - q): two independent models agreeing is stronger evidence than either."""
    if len(a) == 0 or len(b) == 0:
        return sv.Detections.merge([a, b])
    ca, cb = a.get_anchors_coordinates(sv.Position.CENTER), b.get_anchors_coordinates(sv.Position.CENTER)
    d = np.linalg.norm(ca[:, None] - cb[None], axis=2)
    conf_a, keep_b = a.confidence.copy(), np.ones(len(b), bool)
    for i in range(len(a)):
        j = int(np.argmin(d[i]))
        if d[i, j] <= agree_px and keep_b[j]:
            conf_a[i] = 1 - (1 - conf_a[i]) * (1 - b.confidence[j])
            keep_b[j] = False
    a = a[:]
    a.confidence = conf_a
    return sv.Detections.merge([a, b[keep_b]])


def triplets(frames):
    """(previous, current, next) for each frame of a sequence; the first / last frame stands in for its missing
    neighbour."""
    it = iter(frames)
    prev, cur = None, next(it, None)
    while cur is not None:
        nxt = next(it, None)
        yield (prev if prev is not None else cur), cur, (nxt if nxt is not None else cur)
        prev, cur = cur, nxt


def sampled_triplets(video, every=5):
    """(index, previous, current, next) for every `every`-th frame of a video, decoding only those 3 frames
    (the others are skipped without decoding)."""
    assert every >= 3, "frames before and after a sampled frame must not be sampled frames"
    cap = cv2.VideoCapture(str(video))
    prev = cur = None
    k = i = 0
    try:
        while True:
            r = i % every
            if r in (0, 1, every - 1):
                ok, f = cap.read()
            else:
                ok, f = cap.grab(), None
            if not ok:
                break
            if r == 0:
                cur, k = f, i
            elif r == 1 and cur is not None:
                yield k, (prev if prev is not None else cur), cur, f
                prev = cur = None
            if r == every - 1:
                prev = f
            i += 1
        if cur is not None:
            yield k, (prev if prev is not None else cur), cur, cur
    finally:
        cap.release()


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

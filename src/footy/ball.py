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


class CameraMotion:
    """Frame-to-frame motion of a broadcast camera as a homography (previous frame's pixels -> this frame's), from
    sparse optical flow on background features (crowd, boards, lines) with RANSAC rejecting the moving players.

    The main camera pans, tilts and zooms about a fixed point, and for such a camera one homography moves every point
    correctly, whatever its depth, so it also carries a ball in the air. Returns None for the first frame and after a
    cut (too few features follow, or an implausibly large change)."""

    def __init__(self, width=640, max_corners=400, min_inliers=40, min_share=0.5):
        self.width, self.max_corners, self.min_inliers, self.min_share = width, max_corners, min_inliers, min_share
        self.prev = None

    def __call__(self, frame):
        s = self.width / frame.shape[1]
        gray = cv2.cvtColor(cv2.resize(frame, (self.width, round(frame.shape[0] * s)), interpolation=cv2.INTER_AREA),
                            cv2.COLOR_BGR2GRAY)
        prev, self.prev = self.prev, gray
        if prev is None:
            return None
        p0 = cv2.goodFeaturesToTrack(prev, self.max_corners, qualityLevel=0.01, minDistance=8)
        if p0 is None or len(p0) < self.min_inliers:
            return None
        p1, ok, _ = cv2.calcOpticalFlowPyrLK(prev, gray, p0, None, winSize=(21, 21), maxLevel=3)
        ok = ok.ravel() == 1
        if ok.sum() < self.min_inliers:
            return None
        H, inl = cv2.findHomography(p0[ok], p1[ok], cv2.RANSAC, 1.5)
        if H is None or inl.sum() < max(self.min_inliers, self.min_share * ok.sum()):
            return None
        scale = np.diag([s, s, 1.0])
        H = np.linalg.inv(scale) @ H @ scale                            # back to the frame's own pixels
        corners = np.float64([[0, 0], [frame.shape[1], 0], [frame.shape[1], frame.shape[0]], [0, frame.shape[0]]])
        moved = cv2.perspectiveTransform(corners[None], H)[0]
        if np.abs(moved - corners).max() > 0.25 * frame.shape[1]:     # a quarter of the frame in 1/25 s: a cut
            return None
        return H


def fit_curve(t, xy, tol, min_points=5, min_share=0.8, max_accel=None, max_back_deg=85.0):
    """The longest recent stretch of ball sightings that one smooth curve explains, ending at the newest sighting.

    t: frame numbers (ascending), xy: (n, 2) pixels. The curve is a quadratic in time per image axis (over a second, a
    rolling ball moves along a straight line, a ball in the air along a parabola). Each candidate stretch is fitted by
    RANSAC over curves through the newest sighting and two others, so a few false sightings don't break it; it counts
    if >= min_share of its sightings (and >= min_points) lie within tol px, if max_accel is given the curve doesn't
    bend harder than max_accel px/frame^2, and the ball keeps going forward: its velocity stays within max_back_deg
    of the direction from the stretch's start to its end. Between touches a ball never stops and comes back (a
    quadratic can, to fit a dribble's back-and-forth), while a lob (rising, then falling, on screen) or a slowing pass
    keeps going. The velocity is linear in time, so checking both ends checks it all. -> (t of the first sighting on
    it, x coefficients, y coefficients) for np.polyval, or None."""
    t, xy = np.asarray(t, np.float64), np.asarray(xy, np.float64)
    n = len(t)
    for start in range(0, n - min_points + 1):                     # oldest start first: the longest stretch wins
        tt, pp = t[start:] - t[-1], xy[start:]                       # time relative to now: 0 at the newest
        m = len(tt)
        i, j = np.triu_indices(m - 1, 1)                             # curves through the newest + two others
        k = np.full_like(i, m - 1)
        ta, tb, tc = tt[i][:, None], tt[j][:, None], tt[k][:, None]
        T = tt[None]
        la = (T - tb) * (T - tc) / ((ta - tb) * (ta - tc))           # Lagrange basis of each triple, at every t
        lb = (T - ta) * (T - tc) / ((tb - ta) * (tb - tc))
        lc = (T - ta) * (T - tb) / ((tc - ta) * (tc - tb))
        pred = (la[..., None] * pp[i][:, None] + lb[..., None] * pp[j][:, None] + lc[..., None] * pp[k][:, None])
        inl = np.linalg.norm(pred - pp[None], axis=2) <= tol
        best = int(np.argmax(inl.sum(1)))
        keep = inl[best]
        if keep.sum() < max(min_points, min_share * m):
            continue
        cx, cy = np.polyfit(tt[keep], pp[keep, 0], 2), np.polyfit(tt[keep], pp[keep, 1], 2)   # refine on inliers
        fitted = np.stack([np.polyval(cx, tt), np.polyval(cy, tt)], 1)
        keep = np.linalg.norm(fitted - pp, axis=1) <= tol
        if not keep[-1] or keep.sum() < max(min_points, min_share * m):
            continue
        if max_accel is not None and 2 * np.hypot(cx[0], cy[0]) > max_accel:
            continue
        t0 = tt[keep][0]
        chord = np.array([np.polyval(cx, 0) - np.polyval(cx, t0), np.polyval(cy, 0) - np.polyval(cy, t0)])
        ends = np.array([[2 * cx[0] * t0 + cx[1], 2 * cy[0] * t0 + cy[1]], [cx[1], cy[1]]])   # velocity then, now
        cos = ends @ chord / np.maximum(np.linalg.norm(ends, axis=1) * np.linalg.norm(chord), 1e-9)
        if cos.min() < np.cos(np.radians(max_back_deg)):
            continue
        return float(t0 + t[-1]), cx, cy
    return None


class BallTrail:
    """The ball's recent path, drawn only where it is physically sensible.

    Past sightings are kept in the current frame's pixels: each frame moves them with the camera (CameraMotion) and a
    cut forgets them. The trail is the curve from fit_curve, from the first sighting on it to the ball; while no curve
    fits (a touch just happened, the ball was just found, false sightings), there is no trail, only the ball."""

    def __init__(self, fps=25.0, seconds=1.0, width=1280, tol_px=4.0, min_points=5, min_span_s=0.2,
                 max_accel_px=3.0, min_length_px=8.0):
        k = width / 1280                                             # tolerances are for 1280 px wide frames
        self.max_age, self.min_span = int(round(seconds * fps)), min_span_s * fps
        self.tol, self.min_points, self.max_accel = tol_px * k, min_points, max_accel_px * k
        self.min_length = min_length_px * k                          # shorter: a still ball's jitter, not a path
        self.t, self.pts = 0, []                                     # [(frame number, x, y)]

    def update(self, H, ball_xy):
        """H: previous frame -> this frame (CameraMotion; None after a cut). ball_xy: the ball here, or None."""
        self.t += 1
        if H is None:
            self.pts = []
        elif self.pts:
            moved = cv2.perspectiveTransform(np.float64([[(x, y) for _, x, y in self.pts]]), H)[0]
            self.pts = [(f, x, y) for (f, _, _), (x, y) in zip(self.pts, moved)]
        self.pts = [p for p in self.pts if self.t - p[0] < self.max_age]
        if ball_xy is not None:
            self.pts.append((self.t, float(ball_xy[0]), float(ball_xy[1])))

    def path(self, step=0.5):
        """The trail as an (n, 2) polyline from its oldest point to the ball, or None (draw no trail)."""
        if not self.pts or self.pts[-1][0] != self.t:
            return None
        t = [p[0] for p in self.pts]
        fit = fit_curve(t, [p[1:] for p in self.pts], self.tol, self.min_points, max_accel=self.max_accel)
        if fit is None or self.t - fit[0] < self.min_span:
            return None
        ts = np.arange(fit[0] - self.t, step / 2, step)
        path = np.stack([np.polyval(fit[1], ts), np.polyval(fit[2], ts)], 1)
        return path if np.linalg.norm(np.diff(path, axis=0), axis=1).sum() >= self.min_length else None


def draw_trail(frame, path, color=(255, 255, 255), max_thickness=3):
    """A tapered trail: thin and faint at its old end, full at the ball."""
    if path is None or len(path) < 2:
        return frame
    n = len(path)
    for i in range(n - 1):
        w = (i + 1) / n
        c = tuple(int(v * (0.35 + 0.65 * w)) for v in color)
        cv2.line(frame, tuple(np.round(path[i]).astype(int)), tuple(np.round(path[i + 1]).astype(int)), c,
                 max(1, round(max_thickness * w)), cv2.LINE_AA)
    return frame


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

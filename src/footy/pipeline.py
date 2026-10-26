"""Detection -> tracking -> team assignment -> pitch positions -> annotated video (or the game state for events)."""
from collections import Counter, defaultdict
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from trackers import ByteTrackTracker
from ultralytics import YOLO

from footy.ball import BallTrail, BallTracker, CameraMotion, draw_trail, fuse_balls, triplets
from footy.minimap import Minimap
from footy.pitch import HL, HW, project
from footy.soccernet import CLASS_NAMES
from footy.teams import TeamClassifier, TrackTeamVotes, goalkeeper_teams, torso_crop

BALL, GOALKEEPER, PLAYER, REFEREE = range(4)
COCO_TO_OURS = {32: BALL, 0: PLAYER}   # a COCO model has no goalkeeper/referee classes

TEAM_COLORS = ["#00BFFF", "#FF1493"]   # team A, team B
REFEREE_COLOR = "#FFD700"
BALL_COLOR = "#FFFFFF"


class Detector:
    def __init__(self, weights, imgsz=1280, conf=0.1, device=None, half=False):
        self.model = YOLO(weights)
        self.imgsz, self.conf, self.device, self.half = imgsz, conf, device, half   # half: fp16 on a GPU
        names = self.model.names
        self.class_map = COCO_TO_OURS if len(names) == 80 else {i: i for i in range(len(CLASS_NAMES))}

    def __call__(self, frame):
        fp16 = {"quantize": 16} if self.half else {}       # Ultralytics >= 8.4: `half` is deprecated (warns each call)
        r = self.model(frame, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False, **fp16)[0]
        det = sv.Detections.from_ultralytics(r)
        det = det[np.isin(det.class_id, list(self.class_map))]
        det.class_id = np.array([self.class_map[c] for c in det.class_id], dtype=int)
        return det


def fit_team_classifier(detector, video, stride=25, device="cpu", max_frames=None):
    """Collect player torso crops from every `stride`-th frame of one video (or a list: both halves of a match) and
    cluster them into 2 teams."""
    crops = []
    for v in ([video] if isinstance(video, (str, Path)) else video):
        for frame in sv.get_video_frames_generator(str(v), stride=stride, end=max_frames):
            players = detector(frame)
            players = players[(players.class_id == PLAYER) & (players.confidence >= 0.5)]
            crops += [torso_crop(frame, xyxy) for xyxy in players.xyxy]
    print(f"fitting team classifier on {len(crops)} player crops")
    return TeamClassifier(device=device).fit(crops)


def draw_minimap(minimap, H, people, people_colors, ball):
    """Players' feet (bottom centre of their boxes) and the ball, mapped onto the pitch with the homography H."""
    img = minimap.base.copy()
    if H is None:
        return img

    def on_pitch(pts):   # drops points the homography sends off the pitch (e.g. a box high in the stands)
        with np.errstate(all="ignore"):
            xy = project(H, pts)
        return xy, np.isfinite(xy).all(1) & (np.abs(xy[:, 0]) <= HL + 5) & (np.abs(xy[:, 1]) <= HW + 5)

    if len(people):
        xy, keep = on_pitch(people.get_anchors_coordinates(sv.Position.BOTTOM_CENTER))
        minimap.draw(xy[keep], [c for c, k in zip(people_colors, keep) if k], radius=11, canvas=img)
    if len(ball):   # the ball's image position is mapped as if on the grass: off when it is in the air
        xy, keep = on_pitch(ball.get_anchors_coordinates(sv.Position.CENTER))
        minimap.draw(xy[keep], sv.Color.from_hex(BALL_COLOR).as_bgr(), radius=8, canvas=img)
    return img


def draw_caption(frame, text, scale=0.9, margin=16):
    """White text on a dark box, bottom-left (in place)."""
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    y = frame.shape[0] - margin
    cv2.rectangle(frame, (margin - 8, y - h - 10), (margin + w + 8, y + base + 4), (0, 0, 0), -1)
    cv2.putText(frame, text, (margin, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)


class Analysis(NamedTuple):
    people: sv.Detections      # tracked players, goalkeepers and referees; class_id = role (PLAYER / GOALKEEPER / REFEREE)
    color: np.ndarray          # per person: 0 / 1 = team, 2 = referee
    texts: list                # per person: "#12", "GK #3", "REF"
    ball: sv.Detections        # at most one, physically plausible
    H: np.ndarray | None       # image -> pitch homography, or None


class FrameAnalyzer:
    """One frame at a time: detection -> tracking -> roles -> teams -> plausible ball -> pitch homography."""

    def __init__(self, detector, teams, fps, calibrator=None, ballnet=None):
        # ByteTrack: high-confidence detections (>= 0.5) start/extend tracks; low ones (0.1-0.5) can only extend
        self.tracker = ByteTrackTracker(frame_rate=fps, lost_track_buffer=int(fps),
                                        track_activation_threshold=0.5, high_conf_det_threshold=0.5)
        self.detector, self.teams, self.calibrator, self.ballnet = detector, teams, calibrator, ballnet
        self.votes = TrackTeamVotes()
        self.ball_tracker = BallTracker(max_px_per_frame=60.0 * 25 / fps, reacquire_after=int(fps))
        self.role_votes = defaultdict(Counter)   # tracker_id -> votes for player / goalkeeper / referee

    def __call__(self, frame, prev=None, nxt=None) -> Analysis:
        """prev / nxt: the frames just before and after (BallNet-L uses all three); missing ones = this frame."""
        det = self.detector(frame)

        # ball: the detector's and BallNet-L's candidates merged, then only a physically plausible one is kept (see
        # footy/ball.py); padded so the marker is visible
        balls = det[(det.class_id == BALL) & (det.confidence >= 0.3)]
        if self.ballnet is not None:
            balls = fuse_balls(balls, self.ballnet(frame if prev is None else prev, frame,
                                                   frame if nxt is None else nxt))
        ball = self.ball_tracker.update(balls)
        if len(ball):
            ball.xyxy = sv.pad_boxes(ball.xyxy, px=8)

        people = self.tracker.update(det[det.class_id != BALL])
        people = people[people.tracker_id >= 0]   # drop detections not (yet) part of a confirmed track
        # role = majority vote over the track's history (the detector is right about a goalkeeper ~half the time
        # per frame, but over a whole track the majority is reliable, and roles stop flickering)
        for tid, cls in zip(people.tracker_id, people.class_id):
            self.role_votes[tid][int(cls)] += 1
        people.class_id = np.array([self.role_votes[t].most_common(1)[0][0] for t in people.tracker_id], dtype=int)

        players = people[people.class_id == PLAYER]
        # classify only tracks that still need votes, all in one batched call
        pending = [(tid, xyxy) for tid, xyxy in zip(players.tracker_id, players.xyxy) if self.votes.needs_vote(tid)]
        if pending:
            predictions = self.teams.predict([torso_crop(frame, xyxy) for _, xyxy in pending])
            for (tid, _), team in zip(pending, predictions):
                self.votes.add(tid, team)
        player_teams = np.array([self.votes.team(t) for t in players.tracker_id], dtype=int)

        keepers = people[people.class_id == GOALKEEPER]
        keeper_teams = goalkeeper_teams(keepers, players, player_teams) if len(keepers) and len(players) else \
            np.zeros(len(keepers), dtype=int)

        refs = people[people.class_id == REFEREE]
        texts = [f"#{t}" for t in players.tracker_id] + [f"GK #{t}" for t in keepers.tracker_id] + ["REF"] * len(refs)
        H = self.calibrator(frame) if self.calibrator is not None else None
        return Analysis(sv.Detections.merge([players, keepers, refs]),
                        np.concatenate([player_teams, keeper_teams, np.full(len(refs), 2)]).astype(int), texts, ball, H)


def ball_holder(ball_xy, boxes):
    """Index of the person the ball is on (horizontally within their box, between half a body above the head and
    the feet) and where on them: rel_y = 0 at the top of the box, 1 at the feet, < 0 above the head. Or None."""
    if len(boxes) == 0:
        return None
    bx, by = ball_xy
    x1, y1, x2, y2 = boxes.T
    h = np.maximum(y2 - y1, 1.0)
    dx = np.maximum.reduce([x1 - bx, np.zeros_like(x1), bx - x2]) / h
    rel_y = (by - y1) / h
    ok = (dx <= 0.15) & (rel_y >= -0.5) & (rel_y <= 1.1)
    if not ok.any():
        return None
    i = int(np.flatnonzero(ok)[np.argmin(dx[ok])])
    return i, float(rel_y[i])


def frame_state(a: Analysis, t: float) -> dict:
    """What the event rules need from one frame, in pitch metres: the ball (mapped as if on the grass, so off when
    it is in the air), everyone on screen (track id, role, team or -1 for referees, x, y) and who the ball is on
    (track id, role, team, x, y, rel_y: < 0.2 means at or above the head). Without a homography only t is known."""
    s = {"t": round(t, 2), "calibrated": a.H is not None, "ball": None, "people": [], "holder": None}
    if a.H is None:
        return s
    with np.errstate(all="ignore"):
        feet = project(a.H, a.people.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)) if len(a.people) \
            else np.zeros((0, 2))
    on_pitch = np.isfinite(feet).all(1) & (np.abs(feet[:, 0]) <= HL + 5) & (np.abs(feet[:, 1]) <= HW + 5)

    def person(i):
        return [int(a.people.tracker_id[i]), int(a.people.class_id[i]), int(a.color[i]) if a.color[i] < 2 else -1,
                round(float(feet[i, 0]), 2), round(float(feet[i, 1]), 2)]

    s["people"] = [person(i) for i in np.flatnonzero(on_pitch)]
    if len(a.ball):
        centre = a.ball.get_anchors_coordinates(sv.Position.CENTER)
        with np.errstate(all="ignore"):
            xy = project(a.H, centre)[0]
        if np.isfinite(xy).all() and np.abs(xy).max() < 150:
            s["ball"] = [round(float(xy[0]), 2), round(float(xy[1]), 2)]
        held = ball_holder(centre[0], a.people.xyxy)
        if held is not None and on_pitch[held[0]]:
            s["holder"] = person(held[0]) + [round(held[1], 2)]
    return s


def annotate_video(detector, teams, video, target, seconds=None, ball_trail=True, calibrator=None, caption=None,
                   extra=None, ballnet=None):
    """extra: optional callable(frame, t_seconds) that draws more on each output frame (event banners, the score).
    ballnet: optional BallNetDetector, merged with the detector's ball."""
    info = sv.VideoInfo.from_video_path(video)
    end = int(info.fps * seconds) if seconds else None
    analyze = FrameAnalyzer(detector, teams, info.fps, calibrator, ballnet)

    # colour index: 0/1 = teams, 2 = referee
    palette = sv.ColorPalette.from_hex(TEAM_COLORS + [REFEREE_COLOR])
    ellipse = sv.EllipseAnnotator(color=palette, thickness=2)
    label = sv.LabelAnnotator(color=palette, text_color=sv.Color.BLACK, text_scale=0.45, text_padding=3,
                              text_position=sv.Position.BOTTOM_CENTER)
    ball_marker = sv.TriangleAnnotator(color=sv.Color.from_hex(BALL_COLOR), base=22, height=18,
                                       outline_thickness=1)
    # the trail: the last second of the ball's path, moved with the camera, drawn only where one smooth curve explains
    # it (see footy/ball.py: BallTrail); otherwise just the ball
    camera, trail = CameraMotion(), BallTrail(info.fps, width=info.width)
    minimap = Minimap() if calibrator is not None else None
    bgr = [c.as_bgr() for c in palette.colors]

    with sv.VideoSink(target, info) as sink:
        frames = tqdm(sv.get_video_frames_generator(video, end=end), total=end or info.total_frames)
        for i, (prev, frame, nxt) in enumerate(triplets(frames)):
            a = analyze(frame, prev, nxt)
            trail.update(camera(frame) if ball_trail else None,
                         a.ball.get_anchors_coordinates(sv.Position.CENTER)[0] if len(a.ball) else None)
            shown = a.people[:]                   # a copy coloured by team instead of role
            shown.class_id = a.color
            out = ellipse.annotate(frame.copy(), shown)
            out = label.annotate(out, shown, a.texts)
            if len(a.ball):
                if ball_trail:
                    out = draw_trail(out, trail.path(), sv.Color.from_hex(BALL_COLOR).as_bgr())
                out = ball_marker.annotate(out, a.ball)
            if minimap is not None:
                Minimap.overlay(out, draw_minimap(minimap, a.H, shown, [bgr[c] for c in a.color], a.ball))
            if caption:
                draw_caption(out, caption)
            if extra is not None:
                extra(out, i / info.fps)
            sink.write_frame(out)

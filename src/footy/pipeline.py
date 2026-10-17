"""Detection -> tracking -> team assignment -> annotated video."""
import numpy as np
import supervision as sv
from tqdm import tqdm
from trackers import ByteTrackTracker
from ultralytics import YOLO

from footy.soccernet import CLASS_NAMES
from footy.teams import TeamClassifier, TrackTeamVotes, goalkeeper_teams, torso_crop

BALL, GOALKEEPER, PLAYER, REFEREE = range(4)
COCO_TO_OURS = {32: BALL, 0: PLAYER}   # a COCO model has no goalkeeper/referee classes

TEAM_COLORS = ["#00BFFF", "#FF1493"]   # team A, team B
REFEREE_COLOR = "#FFD700"
BALL_COLOR = "#FFFFFF"


class Detector:
    def __init__(self, weights, imgsz=1280, conf=0.1, device=None):
        self.model = YOLO(weights)
        self.imgsz, self.conf, self.device = imgsz, conf, device
        names = self.model.names
        self.class_map = COCO_TO_OURS if len(names) == 80 else {i: i for i in range(len(CLASS_NAMES))}

    def __call__(self, frame):
        r = self.model(frame, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)[0]
        det = sv.Detections.from_ultralytics(r)
        det = det[np.isin(det.class_id, list(self.class_map))]
        det.class_id = np.array([self.class_map[c] for c in det.class_id], dtype=int)
        return det


def fit_team_classifier(detector, video, stride=25, device="cpu", max_frames=None):
    """Collect player torso crops from every `stride`-th frame and cluster them into 2 teams."""
    crops = []
    for i, frame in enumerate(sv.get_video_frames_generator(video, stride=stride, end=max_frames)):
        players = detector(frame)
        players = players[(players.class_id == PLAYER) & (players.confidence >= 0.5)]
        crops += [torso_crop(frame, xyxy) for xyxy in players.xyxy]
    print(f"fitting team classifier on {len(crops)} player crops")
    return TeamClassifier(device=device).fit(crops)


def annotate_video(detector, teams, video, target, seconds=None, ball_trail=True):
    info = sv.VideoInfo.from_video_path(video)
    end = int(info.fps * seconds) if seconds else None
    # ByteTrack: high-confidence detections (>= 0.5) start/extend tracks; low ones (0.1-0.5) can only extend
    tracker = ByteTrackTracker(frame_rate=info.fps, lost_track_buffer=int(info.fps),
                               track_activation_threshold=0.5, high_conf_det_threshold=0.5)
    votes = TrackTeamVotes()

    # colour index: 0/1 = teams, 2 = referee
    palette = sv.ColorPalette.from_hex(TEAM_COLORS + [REFEREE_COLOR])
    ellipse = sv.EllipseAnnotator(color=palette, thickness=2)
    label = sv.LabelAnnotator(color=palette, text_color=sv.Color.BLACK, text_scale=0.45, text_padding=3,
                              text_position=sv.Position.BOTTOM_CENTER)
    ball_marker = sv.TriangleAnnotator(color=sv.Color.from_hex(BALL_COLOR), base=22, height=18,
                                       outline_thickness=1)
    trail = sv.TraceAnnotator(color=sv.Color.from_hex(BALL_COLOR), thickness=2, trace_length=int(info.fps))

    with sv.VideoSink(target, info) as sink:
        for frame in tqdm(sv.get_video_frames_generator(video, end=end), total=end or info.total_frames):
            det = detector(frame)

            # ball: keep only the most confident detection; padded so the marker is visible
            ball = det[(det.class_id == BALL) & (det.confidence >= 0.3)]
            if len(ball):
                ball = ball[[int(np.argmax(ball.confidence))]]
                ball.xyxy = sv.pad_boxes(ball.xyxy, px=8)
                ball.tracker_id = np.array([0])

            people = tracker.update(det[det.class_id != BALL])
            people = people[people.tracker_id >= 0]   # drop detections not (yet) part of a confirmed track

            players = people[people.class_id == PLAYER]
            # classify only tracks that still need votes, all in one batched call
            pending = [(tid, xyxy) for tid, xyxy in zip(players.tracker_id, players.xyxy) if votes.needs_vote(tid)]
            if pending:
                predictions = teams.predict([torso_crop(frame, xyxy) for _, xyxy in pending])
                for (tid, _), team in zip(pending, predictions):
                    votes.add(tid, team)
            player_teams = np.array([votes.team(t) for t in players.tracker_id], dtype=int)

            keepers = people[people.class_id == GOALKEEPER]
            keeper_teams = goalkeeper_teams(keepers, players, player_teams) if len(keepers) and len(players) else \
                np.zeros(len(keepers), dtype=int)

            refs = people[people.class_id == REFEREE]

            shown = sv.Detections.merge([players, keepers, refs])
            shown.class_id = np.concatenate([player_teams, keeper_teams, np.full(len(refs), 2)]).astype(int)
            texts = ([f"#{t}" for t in players.tracker_id] + [f"GK #{t}" for t in keepers.tracker_id]
                     + ["REF"] * len(refs))

            out = ellipse.annotate(frame.copy(), shown)
            out = label.annotate(out, shown, texts)
            if len(ball):
                if ball_trail:
                    out = trail.annotate(out, ball)
                out = ball_marker.annotate(out, ball)
            sink.write_frame(out)

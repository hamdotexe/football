"""A whole match: the game state of each half at a few frames per second, events and the score from it, and a
short rendered clip around each event (banner + running score + minimap) to pick highlights from."""
from __future__ import annotations

import gzip
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm

from footy import events as ev
from footy.ball import sampled_triplets
from footy.pipeline import TEAM_COLORS, FrameAnalyzer, annotate_video, fit_team_classifier, frame_state


def match_teams(detector, videos, device, every_s=30.0):
    """One team classifier for the whole match (both halves), so team ids mean the same thing throughout."""
    fps = sv.VideoInfo.from_video_path(str(videos[0])).fps
    return fit_team_classifier(detector, [str(v) for v in videos], stride=int(every_s * fps), device=device)


def check_deadline(deadline, what):
    """deadline: unix time (or None). Past it, stop with an error instead of starting `what`."""
    if deadline is not None and time.time() > deadline:
        raise TimeoutError(f"time budget used up: stopped before {what}")


def half_states(video, detector, teams, calibrator, ballnet=None, every=5, deadline=None):
    """The game state of every `every`-th frame of one half (5 per second from 25 fps video). BallNet-L also gets
    the frames just before and after each one."""
    info = sv.VideoInfo.from_video_path(str(video))
    fps = info.fps / every
    calibrator.reset(fps)
    analyze = FrameAnalyzer(detector, teams, fps, calibrator, ballnet)
    states = []
    for i, prev, cur, nxt in tqdm(sampled_triplets(video, every), total=info.total_frames // every, mininterval=60):
        check_deadline(deadline, f"minute {i / info.fps / 60:.1f} of {Path(video).name}")
        states.append(frame_state(analyze(cur, prev, nxt), i / info.fps))
    return states


def save_states(path, states):
    with gzip.open(path, "wt") as f:
        for s in states:
            f.write(json.dumps(s, separators=(",", ":")) + "\n")


def load_states(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f]


def score_at(goals, half, t):
    """Goals per team (0, 1) before time t of a half; goals = [(half, t, team)]."""
    s = [0, 0]
    for h, gt, team in goals:
        if team in (0, 1) and (h < half or (h == half and gt <= t)):
            s[team] += 1
    return s


def evaluate(events_by_half, labels, tolerance_s=None):
    """Precision / recall of each kind against SoccerNet's Labels-v2. Recall counts only the events the broadcast
    shows ("visible"); precision counts a detection as right if it matches any label of its kind, shown or not (most
    kick-offs after a goal are labelled "not shown": the replay is still on, but the camera may catch its end)."""
    tolerance_s = tolerance_s or {ev.GOAL: 60.0}
    out = {}
    for kind in ev.SOCCERNET_LABEL:
        m = m_any = n_ours = n_truth = 0
        for half, events in events_by_half.items():
            ours, labelled = [e.t for e in events if e.kind == kind], ev.soccernet_events(labels, half, kind)
            tol = tolerance_s.get(kind, 5.0)
            mm, no, nt = ev.match_events(ours, [t for t, vis in labelled if vis == "visible"], tol)
            m, m_any, n_ours, n_truth = m + mm, m_any + ev.match_events(ours, [t for t, _ in labelled], tol)[0], \
                n_ours + no, n_truth + nt
        out[kind] = {"found": n_ours, "labelled_visible": n_truth, "matched": m, "matched_any": m_any,
                     "precision": round(m_any / n_ours, 3) if n_ours else None,
                     "recall": round(m / n_truth, 3) if n_truth else None}
    return out


# --- clips -------------------------------------------------------------------------------------------------------
BANNER = {ev.GOAL: "GOAL!", ev.CORNER: "CORNER", ev.THROW_IN: "THROW-IN", ev.GOAL_KICK: "GOAL KICK",
          ev.KICK_OFF: "KICK-OFF", ev.PENALTY: "PENALTY"}


def hex_bgr(h):
    return tuple(int(h[i:i + 2], 16) for i in (5, 3, 1))


def draw_banner(frame, text, team=None):
    """Top centre: the event, with a stripe in the colour of the team taking it / scoring."""
    k = frame.shape[1] / 1920                                  # sizes are for 1080p, scaled to the frame
    scale, th = 2.2 * k, max(2, round(4 * k))
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, th)
    x, y, p = (frame.shape[1] - w) // 2, int(90 * k) + h, int(26 * k)
    colour = hex_bgr(TEAM_COLORS[team]) if team in (0, 1) else (255, 255, 255)
    cv2.rectangle(frame, (x - p, y - h - p), (x + w + p, y + base + p), (0, 0, 0), -1)
    cv2.rectangle(frame, (x - p, y - h - p), (x - p + max(6, int(12 * k)), y + base + p), colour, -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, scale, (255, 255, 255), th, cv2.LINE_AA)


def draw_score(frame, score):
    """Top right: [team-0 colour] a - b [team-1 colour]."""
    k = frame.shape[1] / 1920
    scale, th = 1.3 * k, max(2, round(3 * k))
    text = f"{score[0]} - {score[1]}"
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, th)
    x, y, chip, p = frame.shape[1] - w - int(110 * k), int(30 * k), int(25 * k), int(12 * k)
    cv2.rectangle(frame, (x - 2 * chip - p, y - p), (x + w + 2 * chip + p, y + h + base + p), (0, 0, 0), -1)
    cv2.rectangle(frame, (x - chip - p, y), (x - p, y + h + base), hex_bgr(TEAM_COLORS[0]), -1)
    cv2.rectangle(frame, (x + w + p, y), (x + w + p + chip, y + h + base), hex_bgr(TEAM_COLORS[1]), -1)
    cv2.putText(frame, text, (x, y + h), cv2.FONT_HERSHEY_DUPLEX, scale, (255, 255, 255), th, cv2.LINE_AA)


def ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe is None:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    return exe


def ffmpeg(*args):
    r = subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", *map(str, args)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr[-800:]}")


def render_clip(video, half, event, goals, out_path, detector, teams, calibrator, ballnet=None, before=8.0,
                after=6.0, work=None):
    """The full pipeline on [event - before, event + after] of a half, with the event's banner and the score.
    Rendered locally, then written to out_path as H.264 (plays in a browser / Google Drive)."""
    work = Path(work or Path(tempfile.gettempdir()) / "clips")
    work.mkdir(parents=True, exist_ok=True)
    raw, drawn = work / "raw.mp4", work / "drawn.mp4"
    start = max(event.t - before, 0.0)
    ffmpeg("-ss", f"{start:.2f}", "-i", video, "-t", f"{before + after:.2f}", "-an", "-c:v", "libx264", "-crf", "18",
           "-pix_fmt", "yuv420p", raw)

    def extra(frame, t):
        now = start + t
        draw_score(frame, score_at(goals, half, now))
        if event.t - 1.0 <= now <= event.t + 4.0:
            draw_banner(frame, BANNER[event.kind], event.team)

    calibrator.reset(sv.VideoInfo.from_video_path(str(raw)).fps)
    annotate_video(detector, teams, str(raw), str(drawn), calibrator=calibrator, extra=extra, ballnet=ballnet,
                   caption=f"{half}H {int(event.t // 60):02d}:{int(event.t % 60):02d}  {BANNER[event.kind]}")
    ffmpeg("-i", drawn, "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path)
    return Path(out_path)


def true_score(game_name):
    """The real final score from a SoccerNet game folder name ('... Barcelona 6 - 1 Paris SG'), or None."""
    import re
    m = re.search(r" (\d+) - (\d+) ", game_name)
    return (int(m[1]), int(m[2])) if m else None


def process_match(name, halves, out_dir, detector, calibrator, team_device, ballnet=None, labels=None, render=True,
                  log=print, deadline=None):
    """Teams -> game state of each half -> events and score -> (evaluation) -> clips. Every step's result is saved
    in out_dir and reused if it is already there, so an interrupted run continues where it stopped. Past `deadline`
    (unix time) it stops with a TimeoutError; finished halves and clips stay saved."""
    from footy.teams import TeamClassifier
    out = Path(out_dir)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    teams_pkl = out / "teams.pkl"
    if teams_pkl.exists():
        teams = TeamClassifier.load(teams_pkl, device=team_device)
    else:
        teams = match_teams(detector, halves, device=team_device)
        teams.save(teams_pkl)
    events_by_half = {}
    for h, video in enumerate(halves, 1):
        path = out / f"state_{h}.jsonl.gz"
        if not path.exists():
            log(f"{name}: game state of half {h}")
            tmp = Path(tempfile.gettempdir()) / f"{out.name}_{path.name}"     # per match: workers run side by side
            t0 = time.time()
            states = half_states(video, detector, teams, calibrator, ballnet, deadline=deadline)
            dt = time.time() - t0
            log(f"{name}: half {h}: {len(states)} states in {dt / 60:.1f} min ({len(states) / dt:.1f} per s), "
                f"minimap in {calibrator.shown / max(calibrator.frames, 1):.0%}")
            save_states(tmp, states)
            shutil.copy(tmp, path)
        events_by_half[h] = ev.detect(load_states(path))
    goals = [(h, e.t, e.team) for h, es in events_by_half.items() for e in es if e.kind == ev.GOAL]
    final = score_at(goals, len(halves), 1e9)
    summary = {"match": name, "score_ours": final, "score_true": true_score(name),
               "events": {h: [e.to_dict() for e in es] for h, es in events_by_half.items()}}
    if labels is not None:
        summary["vs_soccernet_labels"] = evaluate(events_by_half, labels)
    (out / "events.json").write_text(json.dumps(summary, indent=1))
    log(f"{name}: score {final[0]}-{final[1]} (true {summary['score_true']}), "
        + ", ".join(f"{k}: {sum(e.kind == k for es in events_by_half.values() for e in es)}" for k in BANNER))
    if render:
        for h, e in pick_clips(events_by_half):
            clip = out / "clips" / f"{h}H_{int(e.t // 60):02d}-{int(e.t % 60):02d}_{e.kind.replace(' ', '-')}.mp4"
            if not clip.exists():
                check_deadline(deadline, f"clip {clip.name}")
                render_clip(halves[h - 1], h, e, goals, clip, detector, teams, calibrator, ballnet,
                            work=Path(tempfile.gettempdir()) / f"{out.name}_clip")   # per match, as above
                log(f"  clip {clip.name}")
    return summary


def pick_clips(events_by_half, limits=None):
    """Which events get a clip: every goal, penalty, corner and kick-off; a few throw-ins and goal kicks."""
    limits = limits or {ev.THROW_IN: 4, ev.GOAL_KICK: 4}
    chosen = []
    for half, events in events_by_half.items():
        for kind in BANNER:
            these = [e for e in events if e.kind == kind]
            if kind in limits and len(these) > limits[kind]:
                these = [these[i] for i in np.linspace(0, len(these) - 1, limits[kind]).round().astype(int)]
            chosen += [(half, e) for e in these]
    return sorted(chosen, key=lambda he: (he[0], he[1].t))

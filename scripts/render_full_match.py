"""One half of a match rendered whole, every frame, through the full pipeline (footy.match.render_half): players and
referees ringed by team, the ball with its trail, the minimap, event banners, the running score, the clock and the
broadcast's sound. Events and the score come from the match's saved game state (scripts/process_matches.py), so the
banners and score are the same as in the clips.

Before the half, the same rendering runs on 20 s of it (from minute 10) as a quick check. Any error is logged and
stops the run (exit code 1); so does passing --stop-at, after joining what was rendered so far.

    python scripts/render_full_match.py --video half1.mkv --half 1 --match-dir outputs/matches/<game> --out 1H.mp4 \
        --detector models/football_yolo11m_1280.pt --pitch models/pitch_keypoints.pt --ballnet models/ballnet_l.pt
"""
import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from footy import events as ev  # noqa: E402
from footy import match as M  # noqa: E402


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, help="the half's video")
    p.add_argument("--half", type=int, required=True, choices=(1, 2))
    p.add_argument("--match-dir", required=True, help="teams.pkl, state_1.jsonl.gz and state_2.jsonl.gz of the match")
    p.add_argument("--out", required=True)
    p.add_argument("--detector", required=True)
    p.add_argument("--pitch", required=True)
    p.add_argument("--ballnet", required=True)
    p.add_argument("--device", default="0", help="GPU index, or cpu")
    p.add_argument("--part-minutes", type=float, default=5.0)
    p.add_argument("--check-seconds", type=float, default=20.0)
    p.add_argument("--check-start", type=float, default=600.0)
    p.add_argument("--stop-at", type=float, default=None, help="unix time: stop (with an error) once it passes")
    a = p.parse_args()
    try:
        from footy.ball import BallNetDetector
        from footy.calibration import PitchCalibrator
        from footy.pipeline import Detector
        from footy.teams import TeamClassifier
        d = Path(a.match_dir)
        gpu = a.device != "cpu"
        device, torch_device = (int(a.device), f"cuda:{a.device}") if gpu else ("cpu", "cpu")
        detector = Detector(a.detector, device=device, half=gpu)
        calibrator = PitchCalibrator(a.pitch, device=device, half=gpu)
        ballnet = BallNetDetector(a.ballnet, device=torch_device, half=gpu)
        teams = TeamClassifier.load(d / "teams.pkl", device=torch_device)
        events = {h: ev.detect(M.load_states(d / f"state_{h}.jsonl.gz")) for h in (1, 2)}
        goals = [(h, e.t, e.team) for h, es in events.items() for e in es if e.kind == ev.GOAL]
        log(f"half {a.half}: {len(events[a.half])} events, score after it "
            f"{M.score_at(goals, a.half, 1e9)[0]}-{M.score_at(goals, a.half, 1e9)[1]}")

        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        piece = out.parent / f"check_{a.half}H_piece.mkv"
        M.ffmpeg("-ss", a.check_start, "-i", a.video, "-t", a.check_seconds, "-c", "copy", piece)
        t0 = time.time()
        M.render_half(piece, a.half, events[a.half], goals, out.parent / f"check_{a.half}H.mp4", detector, teams,
                      calibrator, ballnet, part_s=a.check_seconds / 2, log=log)
        piece.unlink()
        log(f"check passed in {time.time() - t0:.0f} s")
        M.render_half(a.video, a.half, events[a.half], goals, out, detector, teams, calibrator, ballnet,
                      part_s=a.part_minutes * 60, deadline=a.stop_at, log=log)
    except Exception:
        log("ERROR\n" + traceback.format_exc())
        sys.exit(1)
    log("ALL DONE")


if __name__ == "__main__":
    main()

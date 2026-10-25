"""Whole matches through the pipeline on one device: footy.match.process_match for each (teams -> game state at
5 fps -> events and score -> check against SoccerNet's labels -> clips).

Before the first match, the whole path (clip rendering included) runs on short pieces of its halves, so a broken
step fails within minutes, not hours into a match.

Jobs come from a JSON file: [{"name": "<SoccerNet game folder name>", "halves": [half1, half2], "labels": path|null}].
Every result is saved in --out/<name>/ and reused, so running again continues where it stopped. Any error is
logged and stops the run (exit code 1); so does passing --stop-at (finished halves and clips stay saved).

    python scripts/process_matches.py --jobs jobs.json --out outputs/matches --detector models/football_yolo11m_1280.pt \
        --pitch models/pitch_keypoints.pt --ballnet models/ballnet_l.pt --device 0
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from footy import events as ev  # noqa: E402
from footy import match as M  # noqa: E402


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def load_models(a):
    """detector, calibrator, team device, BallNet-L: on GPU `a.device` in fp16, or on CPU."""
    from footy.ball import BallNetDetector
    from footy.calibration import PitchCalibrator
    from footy.pipeline import Detector
    gpu = a.device != "cpu"
    device, torch_device = (int(a.device), f"cuda:{a.device}") if gpu else ("cpu", "cpu")
    return (Detector(a.detector, device=device, half=gpu), PitchCalibrator(a.pitch, device=device, half=gpu),
            torch_device, BallNetDetector(a.ballnet, device=torch_device, half=gpu))


def check(job, out, detector, calibrator, team_device, ballnet, seconds, start):
    """Teams, game state, events and one rendered clip on `seconds` of each half (from `start`)."""
    from footy.pipeline import fit_team_classifier
    from footy.teams import TeamClassifier
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pieces = []
    for h, video in enumerate(job["halves"], 1):
        piece = out / f"piece_{h}.mkv"
        M.ffmpeg("-ss", start, "-i", video, "-t", seconds, "-an", "-c", "copy", piece)
        pieces.append(piece)
    teams = fit_team_classifier(detector, [str(p) for p in pieces], stride=50, device=team_device)
    teams.save(out / "teams.pkl")                       # process_match reads it back instead of fitting at 30 s
    M.process_match("check", pieces, out, detector, calibrator, team_device, ballnet, render=False, log=log)
    teams = TeamClassifier.load(out / "teams.pkl", device=team_device)
    t1 = time.time()
    M.render_clip(pieces[0], 1, ev.Event(seconds / 2, ev.THROW_IN, 0.0, 0.0), [], out / "clip.mp4", detector, teams,
                  calibrator, ballnet, before=1.0, after=1.0, work=out / "work")
    for p in pieces:
        p.unlink()
    log(f"check passed in {time.time() - t0:.0f} s (2 s clip rendered in {time.time() - t1:.0f} s)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jobs", required=True, help="JSON file: [{name, halves: [h1, h2], labels}]")
    p.add_argument("--out", required=True)
    p.add_argument("--detector", required=True)
    p.add_argument("--pitch", required=True)
    p.add_argument("--ballnet", required=True)
    p.add_argument("--device", default="0", help="GPU index, or cpu")
    p.add_argument("--render", action="store_true", help="also render a clip around each picked event")
    p.add_argument("--check-seconds", type=float, default=20.0)
    p.add_argument("--check-start", type=float, default=600.0, help="where in each half the check's piece starts")
    p.add_argument("--stop-at", type=float, default=None, help="unix time: stop with an error once it passes")
    a = p.parse_args()
    try:
        jobs = json.loads(Path(a.jobs).read_text())
        missing = [f for j in jobs for f in [*j["halves"], j.get("labels")] if f and not Path(f).exists()]
        assert not missing, f"input files not found: {missing}"
        out = Path(a.out)
        models = load_models(a)
        check(jobs[0], out / "_check" / f"device{a.device}", *models, a.check_seconds, a.check_start)
        for job in jobs:
            labels = json.loads(Path(job["labels"]).read_text()) if job.get("labels") else None
            t0 = time.time()
            M.process_match(job["name"], job["halves"], out / job["name"], *models, labels=labels, render=a.render,
                            log=log, deadline=a.stop_at)
            log(f"{job['name']}: done in {(time.time() - t0) / 60:.0f} min")
    except Exception:
        log("ERROR\n" + traceback.format_exc())
        sys.exit(1)
    log("ALL DONE")


if __name__ == "__main__":
    main()

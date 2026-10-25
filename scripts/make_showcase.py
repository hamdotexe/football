"""Showcase video: the whole pipeline on short pieces of many clips no model was trained on, joined into one video.

    python scripts/make_showcase.py --weights football_yolo11m_1280.pt --pitch-weights pitch_keypoints.pt \
        --soccernet-zip test.zip --every 4 --video 0bfacc_0.mp4 --out outputs/showcase --device 0

Pieces: every n-th clip of a SoccerNet GSR split zip (frames <clip>/img1/000001.jpg...) and any extra videos. Each
piece gets its own team classifier (the teams differ per match) and a fresh tracker and calibration; a caption names
its source. Fail-fast: any failed step logs the reason and stops; finished pieces stay in <out>/pieces.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

log = logging.getLogger("showcase")
W, H, FPS = 1920, 1080, 25


class RunError(RuntimeError):
    pass


def check(ok: bool, message: str) -> None:
    if not ok:
        raise RunError(message)


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass
    check(exe is not None, "ffmpeg not found (install it or imageio-ffmpeg)")
    return exe


def ffmpeg(*args: str) -> None:
    r = subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", *args], capture_output=True, text=True)
    check(r.returncode == 0, f"ffmpeg failed: {r.stderr.strip()[-1500:]}")


def frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


ENCODE = ["-vf", f"scale={W}:{H}", "-r", str(FPS), "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p"]


def soccernet_piece(zf: zipfile.ZipFile, clip: str, start_s: float, seconds: float, work: Path) -> Path:
    """Frames start_s..start_s+seconds of a SoccerNet clip -> an mp4 (frames are numbered from 000001)."""
    first, n = int(start_s * FPS) + 1, int(seconds * FPS)
    names = [f"{clip}/img1/{i:06d}.jpg" for i in range(first, first + n)]
    have = set(zf.namelist())
    missing = [x for x in names if x not in have]
    check(not missing, f"{clip}: {len(missing)} of {n} frames missing from the zip (first: {missing[:1]})")
    frames = work / "frames" / clip
    for name in names:
        zf.extract(name, work / "frames")
    out = work / f"{clip}.mp4"
    ffmpeg("-framerate", str(FPS), "-start_number", str(first), "-i", str(frames / "img1" / "%06d.jpg"),
           "-frames:v", str(n), *ENCODE, str(out))
    shutil.rmtree(frames)
    return out


def video_piece(video: Path, start_s: float, seconds: float, work: Path) -> Path:
    out = work / f"{video.stem}.mp4"
    ffmpeg("-ss", str(start_s), "-i", str(video), "-t", str(seconds), *ENCODE, str(out))
    return out


def run(a) -> None:
    from footy.calibration import PitchCalibrator
    from footy.pipeline import Detector, annotate_video, fit_team_classifier

    out, work = Path(a.out), Path(a.out) / "work"
    pieces_dir = out / "pieces"
    for d in (work, pieces_dir):
        d.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + a.max_minutes * 60
    ffmpeg_exe()
    if a.device != "cpu":
        import torch
        check(torch.cuda.is_available(), "CUDA is not available: is the GPU accelerator enabled?")
        log.info("GPU: %s", torch.cuda.get_device_name(0))

    # (caption, raw piece maker)
    plan = []
    if a.soccernet_zip:
        zf = zipfile.ZipFile(a.soccernet_zip)
        clips = sorted({n.split("/")[0] for n in zf.namelist() if n.endswith("Labels-GameState.json")})
        check(len(clips) > 0, f"no SoccerNet clips in {a.soccernet_zip}")
        chosen = clips[:: a.every]
        log.info("%d SoccerNet clips in the zip, using every %d-th: %s", len(clips), a.every, chosen)
        plan += [(f"SoccerNet {a.split_name} clip {c} (never used in training)",
                  lambda c=c: soccernet_piece(zf, c, a.start, a.seconds, work)) for c in chosen]
    for v in a.video or []:
        check(Path(v).is_file(), f"video not found: {v}")
        plan.append((f"{Path(v).stem}: Bundesliga clip (never used in training)",
                     lambda v=v: video_piece(Path(v), a.start, a.seconds, work)))
    check(len(plan) > 0, "nothing to render: give --soccernet-zip and/or --video")

    detector = Detector(a.weights, imgsz=1280, device=None if a.device == "auto" else a.device)
    calibrator = PitchCalibrator(a.pitch_weights, fps=FPS, device=None if a.device == "auto" else a.device)
    team_device = "cpu" if a.device == "cpu" else "cuda"
    done, stats = [], []
    for k, (caption, make) in enumerate(plan):
        if time.time() > deadline:
            log.error("time budget of %.0f min reached after %d of %d pieces: joining those", a.max_minutes, k, len(plan))
            break
        t = time.time()
        raw = make()
        n = frame_count(raw)
        check(n >= 0.95 * a.seconds * FPS, f"piece {k} ({caption}) has {n} frames, expected {a.seconds * FPS:.0f}")
        teams = fit_team_classifier(detector, str(raw), stride=10, device=team_device)   # 25 frames of a 10 s piece
        calibrator.reset()
        piece = pieces_dir / f"{k:02d}.mp4"
        annotate_video(detector, teams, str(raw), str(piece), calibrator=calibrator, caption=caption)
        check(frame_count(piece) == n, f"piece {k}: annotated {frame_count(piece)} of {n} frames")
        raw.unlink()
        s = {"piece": k, "caption": caption, "frames": n, "seconds": round(time.time() - t, 1),
             "minimap_shown": round(calibrator.shown / max(calibrator.frames, 1), 3),
             "own_homography": round(calibrator.estimated / max(calibrator.frames, 1), 3)}
        stats.append(s)
        done.append(piece)
        log.info("piece %d/%d done in %.0f s (%.1f frames/s): minimap in %.0f%% of frames (own estimate in %.0f%%) | %s",
                 k + 1, len(plan), s["seconds"], n / s["seconds"], 100 * s["minimap_shown"], 100 * s["own_homography"],
                 caption)

    check(len(done) > 0, "no piece finished")
    listing = work / "pieces.txt"
    listing.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for p in done))
    final = out / "showcase.mp4"
    ffmpeg("-f", "concat", "-safe", "0", "-i", str(listing), "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(final))
    total = sum(s["frames"] for s in stats)
    check(abs(frame_count(final) - total) <= len(done), f"showcase has {frame_count(final)} frames, pieces had {total}")
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    log.info("showcase: %s (%d pieces, %.0f s, %.0f MB)", final, len(done), total / FPS, final.stat().st_size / 1e6)
    shutil.rmtree(pieces_dir)
    shutil.rmtree(work)
    check(len(done) == len(plan), f"only {len(done)} of {len(plan)} pieces fit in the time budget")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="player / goalkeeper / referee / ball detector")
    ap.add_argument("--pitch-weights", required=True, help="pitch keypoint model")
    ap.add_argument("--soccernet-zip", default=None)
    ap.add_argument("--split-name", default="test")
    ap.add_argument("--every", type=int, default=4, help="use every n-th SoccerNet clip")
    ap.add_argument("--video", action="append", help="extra video(s) to include")
    ap.add_argument("--start", type=float, default=5.0, help="piece start within each clip (s)")
    ap.add_argument("--seconds", type=float, default=10.0, help="piece length (s)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="0", help="0, cpu or auto")
    ap.add_argument("--max-minutes", type=float, default=60)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(Path(a.out) / "showcase.log")):
        handler.setFormatter(fmt)
        log.addHandler(handler)
    log.setLevel(logging.INFO)
    try:
        run(a)
    except RunError as e:
        log.error("STOPPED: %s", e)
        return 1
    except Exception:
        log.exception("STOPPED: unexpected error")
        return 1
    log.info("finished OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

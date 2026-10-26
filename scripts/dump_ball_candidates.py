"""Every frame's ball candidates (the detector's and BallNet-L's, before any tracking) for a piece of a video, so ball
tracking and trail drawing can be developed offline against real detections.

    python scripts/dump_ball_candidates.py data/match_videos/dortmund_leverkusen_1.mkv --start 1536 --seconds 12 \
        --out outputs/ball_dump/dortmund_25-36.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from footy.ball import BALL, BallNetDetector, triplets  # noqa: E402
from footy.pipeline import Detector  # noqa: E402


def frames(video, start, n):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    for _ in range(n):
        ok, f = cap.read()
        if not ok:
            break
        yield f
    cap.release()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video")
    p.add_argument("--start", type=float, required=True, help="seconds into the video")
    p.add_argument("--seconds", type=float, default=12.0)
    p.add_argument("--out", required=True)
    p.add_argument("--detector", default="models/football_yolo11m_1280.pt")
    p.add_argument("--ballnet", default="models/ballnet_l.pt")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    fps = cv2.VideoCapture(a.video).get(cv2.CAP_PROP_FPS)
    det = Detector(a.detector, device=a.device)
    ballnet = BallNetDetector(a.ballnet, device=a.device)
    out, t0 = [], time.time()
    for i, (prev, cur, nxt) in enumerate(triplets(frames(a.video, a.start, int(a.seconds * fps)))):
        d = det(cur)
        d = d[(d.class_id == BALL) & (d.confidence >= 0.3)]
        b = ballnet(prev, cur, nxt)
        out.append({"i": i, "detector": [[*map(float, xy), float(c)] for xy, c in zip(d.xyxy, d.confidence)],
                    "ballnet": [[*map(float, xy), float(c)] for xy, c in zip(b.xyxy, b.confidence)]})
        if i % 25 == 0:
            print(f"frame {i}: {time.time() - t0:.0f} s", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"video": a.video, "start": a.start, "fps": fps, "frames": out}))
    print("wrote", a.out, len(out), "frames")


if __name__ == "__main__":
    main()

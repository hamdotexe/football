"""Run detection + tracking + team assignment on a video and write an annotated copy.

Usage: python scripts/run_pipeline.py --source data/clips/0bfacc_0.mp4 --target outputs/step4.mp4
                                      --weights models/yolo11m.pt --seconds 10
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from footy.pipeline import Detector, annotate_video, fit_team_classifier  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--source", required=True)
p.add_argument("--target", required=True)
p.add_argument("--weights", default="models/yolo11m.pt")
p.add_argument("--seconds", type=float, default=None, help="only process the first N seconds")
p.add_argument("--imgsz", type=int, default=1280)
p.add_argument("--device", default=None, help="cpu, 0, ... (default: auto)")
args = p.parse_args()

detector = Detector(args.weights, imgsz=args.imgsz, device=args.device)
teams = fit_team_classifier(detector, args.source, device="cuda" if args.device not in (None, "cpu") else "cpu",
                            max_frames=int(25 * args.seconds) if args.seconds else None)
annotate_video(detector, teams, args.source, args.target, seconds=args.seconds)
print("wrote", args.target)

"""Download the Roboflow football-players-detection dataset in YOLO format.

The API key is read from the gitignored .env file (ROBOFLOW_API_KEY=...), never from code.

Usage: python scripts/download_roboflow.py [version]
"""
import sys
from pathlib import Path

from roboflow import Roboflow

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE, PROJECT = "roboflow-jvuqo", "football-players-detection-3zvbc"


def read_key():
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("ROBOFLOW_API_KEY="):
            return line.split("=", 1)[1].strip()
    sys.exit("ROBOFLOW_API_KEY missing from .env")


project = Roboflow(api_key=read_key()).workspace(WORKSPACE).project(PROJECT)
versions = sorted(int(v.version.split("/")[-1]) for v in project.versions())
print("available versions:", versions)

version = int(sys.argv[1]) if len(sys.argv) > 1 else versions[-1]
out = ROOT / "data" / "roboflow" / f"v{version}"
project.version(version).download("yolov11", location=str(out))
print("downloaded to", out)

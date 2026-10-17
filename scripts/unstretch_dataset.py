"""Undo Roboflow's square "Stretch to" preprocessing: resize every image back to 16:9.

YOLO labels are normalised (fractions of width/height), so they stay valid after a resize
and are copied unchanged.

Usage: python scripts/unstretch_dataset.py data/roboflow/v13 data/roboflow/v13_1280x720
"""
import shutil
import sys
from pathlib import Path

import cv2

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
W, H = 1280, 720

for split in ["train", "valid", "test"]:
    for img_path in sorted((src / split / "images").glob("*")):
        out_img = dst / split / "images" / img_path.name
        out_img.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_img), cv2.resize(cv2.imread(str(img_path)), (W, H), interpolation=cv2.INTER_AREA))

        label = src / split / "labels" / (img_path.stem + ".txt")
        out_label = dst / split / "labels" / label.name
        out_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(label, out_label)
    print(split, len(list((dst / split / "images").glob("*"))), "images")

shutil.copy(src / "data.yaml", dst / "data.yaml")

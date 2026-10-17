"""Convert SoccerNet Game State Reconstruction (GSR) clips into a YOLO detection dataset.

GSR layout inside each split zip:
    SNGS-033/Labels-GameState.json   # COCO-like: info, images, annotations, categories
    SNGS-033/img1/000001.jpg ...     # 750 frames (30 s at 25 fps), 1920x1080

Each annotation has `bbox_image` = {x, y, w, h} in pixels (x, y = top-left corner).
"""
import json
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

# Our class order, shared with the Roboflow dataset so the two can be merged.
CLASS_NAMES = ["ball", "goalkeeper", "player", "referee"]

# SoccerNet category_id -> our class id. Dropped: 5 pitch, 6 camera, 7 "other" (staff, ball boys, ...),
# which matches Roboflow, where coaches and staff are also unlabelled.
SOCCERNET_TO_YOLO = {4: 0, 2: 1, 1: 2, 3: 3}


def frame_number(image):
    return int(Path(image["file_name"]).stem)


def select_frames(images, stride):
    """Keep every `stride`-th labelled frame (frames are numbered from 1)."""
    return [im for im in images if im["is_labeled"] and (frame_number(im) - 1) % stride == 0]


def boxes_by_image(annotations):
    boxes = {}
    for a in annotations:
        cls = SOCCERNET_TO_YOLO.get(a.get("category_id"))
        if cls is not None and a.get("bbox_image"):
            boxes.setdefault(a["image_id"], []).append((cls, a["bbox_image"]))
    return boxes


def to_yolo_lines(boxes, width, height):
    """Pixel top-left boxes -> YOLO lines 'cls cx cy w h', normalised to 0-1 and clipped to the image."""
    lines = []
    for cls, b in boxes:
        x1, y1 = max(0.0, b["x"]), max(0.0, b["y"])
        x2, y2 = min(width, b["x"] + b["w"]), min(height, b["y"] + b["h"])
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        lines.append(f"{cls} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
                     f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}")
    return lines


def _write_frame(jpg_bytes, img_out, size, quality=90):
    img = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(img_out), img, [cv2.IMWRITE_JPEG_QUALITY, quality])


def convert_split(zf: zipfile.ZipFile, out_root, split, stride=5, size=(1280, 720),
                  max_clips=None, max_frames=None, labels_out=None, workers=8):
    """Convert every clip in an opened GSR split zip. Returns per-split statistics."""
    out_img = Path(out_root) / split / "images"
    out_lbl = Path(out_root) / split / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    label_files = sorted(n for n in zf.namelist() if n.endswith("Labels-GameState.json"))[:max_clips]
    stats = {"clips": 0, "images": 0, "skipped_ignore_regions": 0, "boxes": Counter(), "game_ids": set()}

    with ThreadPoolExecutor(workers) as pool:
        for label_file in label_files:
            clip = label_file.split("/")[0]
            raw = zf.read(label_file)
            data = json.loads(raw)
            if labels_out:  # keep the raw labels: pitch lines etc. are needed for the minimap step
                Path(labels_out, split).mkdir(parents=True, exist_ok=True)
                Path(labels_out, split, f"{clip}.json").write_bytes(raw)

            stats["clips"] += 1
            stats["game_ids"].add(data["info"].get("game_id"))
            boxes = boxes_by_image(data["annotations"])
            jobs = []
            for im in select_frames(data["images"], stride)[:max_frames]:
                if im.get("ignore_regions_x"):
                    stats["skipped_ignore_regions"] += 1
                    continue
                lines = to_yolo_lines(boxes.get(im["image_id"], []), im["width"], im["height"])
                name = f"{clip}_{Path(im['file_name']).stem}"
                (out_lbl / f"{name}.txt").write_text("\n".join(lines))
                stats["boxes"].update(CLASS_NAMES[int(l.split()[0])] for l in lines)
                jpg = zf.read(f"{clip}/{data['info'].get('im_dir', 'img1')}/{im['file_name']}")
                jobs.append(pool.submit(_write_frame, jpg, out_img / f"{name}.jpg", size))
            for j in jobs:
                j.result()
            stats["images"] += len(jobs)
            print(f"  {split}/{clip}: {len(jobs)} frames", flush=True)

    stats["game_ids"] = sorted(stats["game_ids"], key=str)
    stats["boxes"] = dict(stats["boxes"])
    return stats

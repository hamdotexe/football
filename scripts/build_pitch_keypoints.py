"""Build YOLO-pose pitch keypoint labels from SoccerNet GSR annotations.

For every frame used in the detection dataset (every 5th), fit a ground-truth homography from the players' feet
(image <-> pitch metres), refine it against the annotated pitch lines, drop badly-fitting frames, and project our
33 pitch keypoints into the image.

Labels only: the images are the same 1280x720 frames as the detection dataset (matched by file name).

Usage: python scripts/build_pitch_keypoints.py --out data/pitch_pose [--max-clips 2]
"""
import argparse
import collections
import json
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from footy.pitch import (HL, HW, KEYPOINT_NAMES, KEYPOINT_XY, homography_from_players,  # noqa: E402
                         line_observations, project, refine_homography)
from footy.soccernet import select_frames  # noqa: E402

STRIDE = 5
MAX_MEDIAN_PX, MAX_P90_PX = 4.0, 10.0      # quality bar for a frame's fitted homography (1920x1080 pixels)
FLIP_IDX = [KEYPOINT_NAMES.index(n.replace("L_", "#").replace("R_", "L_").replace("#", "R_")
                                 .replace("circle_left", "#").replace("circle_right", "circle_left").replace("#", "circle_right"))
            for n in KEYPOINT_NAMES]


def in_front(H, pts):
    """True where a pitch point projects in front of the camera (homogeneous w > 0, same side as the pitch centre)."""
    w = (H[2, 0] * pts[:, 0] + H[2, 1] * pts[:, 1] + H[2, 2])
    w0 = H[2, 0] * 0 + H[2, 1] * 0 + H[2, 2]
    return np.sign(w) == np.sign(w0)


def pose_label(H, width, height):
    """YOLO-pose line: class, pitch bbox (normalised), then x y v for each keypoint."""
    kp = project(H, KEYPOINT_XY)
    visible = in_front(H, KEYPOINT_XY) & (kp[:, 0] >= 0) & (kp[:, 0] < width) & (kp[:, 1] >= 0) & (kp[:, 1] < height)

    outline = np.concatenate([np.linspace(a, b, 100) for a, b in
                              [((-HL, -HW), (HL, -HW)), ((HL, -HW), (HL, HW)), ((HL, HW), (-HL, HW)), ((-HL, HW), (-HL, -HW))]])
    o = project(H, outline)[in_front(H, outline)]
    o = o[(o[:, 0] > -width) & (o[:, 0] < 2 * width) & (o[:, 1] > -height) & (o[:, 1] < 2 * height)]
    x1, y1 = np.clip(o.min(0), 0, [width, height]) if len(o) else (0, 0)
    x2, y2 = np.clip(o.max(0), 0, [width, height]) if len(o) else (width, height)

    parts = [f"0 {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} {(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"]
    for (x, y), v in zip(kp, visible):
        parts.append(f"{x / width:.6f} {y / height:.6f} 2" if v else "0 0 0")
    return " ".join(parts), int(visible.sum())


def process_clip(args):
    split, clip, raw, out = args
    data = json.loads(raw)
    by_image = collections.defaultdict(list)
    for a in data["annotations"]:
        by_image[a["image_id"]].append(a)

    out_dir = Path(out, split, "labels")
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = collections.Counter()
    for im in select_frames(data["images"], STRIDE):
        anns = by_image[im["image_id"]]
        obs = [o for a in anns if a.get("category_id") == 5 for o in line_observations(a, im["width"], im["height"])]
        H_inv, _, _ = homography_from_players(anns)
        if H_inv is None or not obs:
            stats["no_fit"] += 1
            continue
        H, err = refine_homography(np.linalg.inv(H_inv), obs)
        if len(err) < 4 or np.median(err) > MAX_MEDIAN_PX or np.percentile(err, 90) > MAX_P90_PX:
            stats["rejected"] += 1
            continue
        line, n_visible = pose_label(H, im["width"], im["height"])
        (out_dir / f"{clip}_{Path(im['file_name']).stem}.txt").write_text(line)
        stats["kept"] += 1
        stats["visible_keypoints"] += n_visible
    return split, clip, stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "data" / "pitch_pose"))
    p.add_argument("--max-clips", type=int, default=None)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    jobs = []
    for split in ["train", "valid", "test"]:
        zf = zipfile.ZipFile(fs.open(f"datasets/SoccerNet/SN-GSR-2025/{split}.zip", "rb", block_size=2 ** 20))
        for name in sorted(n for n in zf.namelist() if n.endswith("Labels-GameState.json"))[:args.max_clips]:
            jobs.append((split, name.split("/")[0], zf.read(name), args.out))   # reads only this file (range request)
        print(f"{split}: fetched {sum(j[0] == split for j in jobs)} label files", flush=True)

    totals = collections.defaultdict(collections.Counter)
    with ProcessPoolExecutor(args.workers) as pool:
        for split, clip, stats in pool.map(process_clip, jobs):
            totals[split].update(stats)
            print(f"  {split}/{clip}: {dict(stats)}", flush=True)

    meta = {"keypoints": KEYPOINT_NAMES, "flip_idx": FLIP_IDX, "pitch_xy_m": KEYPOINT_XY.tolist(),
            "stride": STRIDE, "quality": {"max_median_px": MAX_MEDIAN_PX, "max_p90_px": MAX_P90_PX},
            "splits": {s: dict(c) for s, c in totals.items()}}
    Path(args.out, "keypoints.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta["splits"], indent=2))


if __name__ == "__main__":
    main()

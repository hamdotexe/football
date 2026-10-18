"""Training samples for BallNet: 3 consecutive SoccerNet frames + a heatmap of the ball in the middle frame.

Frames are read straight out of the SoccerNet split zip (no extraction), resized from 1920x1080 to 1280x720.
"""
import json
import random
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from footy.ballnet import gaussian_heatmap

W, H = 1280, 720
SCALE = W / 1920


def ball_index(zf, clips=None):
    """[(clip, im_dir, frame_number, ball_xy_or_None)] for every frame that has both neighbours (t-1, t+1)."""
    items = []
    for label_file in sorted(n for n in zf.namelist() if n.endswith("Labels-GameState.json")):
        clip = label_file.split("/")[0]
        if clips is not None and clip not in clips:
            continue
        data = json.loads(zf.read(label_file))
        frame_of = {im["image_id"]: int(Path(im["file_name"]).stem) for im in data["images"] if im["is_labeled"]}
        ball = {}
        for a in data["annotations"]:
            if a.get("category_id") == 4 and a.get("bbox_image") and a["image_id"] in frame_of:
                b = a["bbox_image"]
                ball[frame_of[a["image_id"]]] = (b["x_center"] * SCALE, b["y_center"] * SCALE)
        frames = set(frame_of.values())
        im_dir = data["info"].get("im_dir", "img1")
        items += [(clip, im_dir, f, ball.get(f)) for f in sorted(frames) if f - 1 in frames and f + 1 in frames]
    return items


class BallClips(Dataset):
    def __init__(self, zip_path, items, crop=(384, 640), train=True, ball_crop_prob=0.7, opener=None):
        """opener: optional callable returning an open ZipFile (e.g. a remote one); default opens zip_path."""
        self.zip_path, self.items, self.train = zip_path, items, train
        self.opener = opener or (lambda: zipfile.ZipFile(self.zip_path))
        self.crop_h, self.crop_w = crop
        self.ball_crop_prob = ball_crop_prob
        self._zf = None   # opened lazily, once per DataLoader worker (zip handles can't be shared across processes)

    def __len__(self):
        return len(self.items)

    def _frame(self, clip, im_dir, n):
        if self._zf is None:
            self._zf = self.opener()
        jpg = self._zf.read(f"{clip}/{im_dir}/{n:06d}.jpg")
        return cv2.resize(cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR), (W, H), interpolation=cv2.INTER_AREA)

    def __getitem__(self, i):
        clip, im_dir, f, xy = self.items[i]
        frames = np.concatenate([self._frame(clip, im_dir, n) for n in (f - 1, f, f + 1)], axis=2)   # H x W x 9

        if self.train:
            # random crop: usually around the ball, sometimes anywhere (so the model also learns "no ball here")
            if xy is not None and random.random() < self.ball_crop_prob:
                x0 = int(np.clip(xy[0] - random.uniform(16, self.crop_w - 16), 0, W - self.crop_w))
                y0 = int(np.clip(xy[1] - random.uniform(16, self.crop_h - 16), 0, H - self.crop_h))
            else:
                x0, y0 = random.randint(0, W - self.crop_w), random.randint(0, H - self.crop_h)
            frames = frames[y0:y0 + self.crop_h, x0:x0 + self.crop_w]
            xy = None if xy is None else (xy[0] - x0, xy[1] - y0)
            if random.random() < 0.5:                          # mirror (same flip for all 3 frames)
                frames = frames[:, ::-1]
                xy = None if xy is None else (self.crop_w - 1 - xy[0], xy[1])
            frames = np.clip(frames.astype(np.float32) * random.uniform(0.8, 1.2) + random.uniform(-20, 20), 0, 255)

        h, w = frames.shape[:2]
        if xy is not None and not (0 <= xy[0] < w and 0 <= xy[1] < h):
            xy = None
        x = torch.from_numpy(np.ascontiguousarray(frames, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        y = torch.from_numpy(gaussian_heatmap(h, w, xy))[None]
        target = torch.tensor([xy[0], xy[1], 1.0] if xy is not None else [0.0, 0.0, 0.0])
        return x, y, target

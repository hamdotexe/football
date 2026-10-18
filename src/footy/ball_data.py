"""Training samples for BallNet: 3 consecutive SoccerNet frames + a heatmap of the ball in the middle frame.

SoccerNet GSR split zips contain `<clip>/img1/<frame:06d>.jpg` (1920x1080, 25 fps) and `<clip>/Labels-GameState.json`.
Everything here works at 1280x720 (ball coordinates are scaled accordingly).

Two frame sources:
- ZipFrames: decode straight from the zip (no extra disk, but full-HD decoding is CPU-heavy)
- DirFrames: read frames pre-extracted at 1280x720 by `extract_frames` (~3x cheaper per frame)
"""
from __future__ import annotations

import json
import random
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, NamedTuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from footy.ballnet import gaussian_heatmap

W, H = 1280, 720
SCALE = W / 1920
IM_DIR = "img1"


class Sample(NamedTuple):
    clip: str
    frame: int                            # frame number of the middle frame t (frames t-1 and t+1 also exist)
    ball: tuple[float, float] | None      # ball centre at 1280x720, or None when the ball isn't visible


@dataclass(frozen=True)
class ZipRef:
    """Picklable reference to a split zip: a local path, or 'hf://<repo path>' for remote range reads (smoke tests)."""
    path: str

    def open(self) -> zipfile.ZipFile:
        if self.path.startswith("hf://"):
            from huggingface_hub import HfFileSystem
            return zipfile.ZipFile(HfFileSystem().open(self.path[5:], "rb", block_size=2 ** 20))
        return zipfile.ZipFile(self.path)


def load_samples(zf: zipfile.ZipFile, clips: set[str] | None = None) -> list[Sample]:
    """One Sample per labelled frame whose two neighbours are also labelled."""
    samples = []
    for label_file in sorted(n for n in zf.namelist() if n.endswith("Labels-GameState.json")):
        clip = label_file.split("/")[0]
        if clips is not None and clip not in clips:
            continue
        data = json.loads(zf.read(label_file))
        frame_of = {im["image_id"]: int(Path(im["file_name"]).stem) for im in data["images"] if im["is_labeled"]}
        ball = {}
        for a in data["annotations"]:
            box = a.get("bbox_image")
            if a.get("category_id") == 4 and box and a["image_id"] in frame_of:
                ball[frame_of[a["image_id"]]] = (box["x_center"] * SCALE, box["y_center"] * SCALE)
        frames = set(frame_of.values())
        samples += [Sample(clip, f, ball.get(f)) for f in sorted(frames) if f - 1 in frames and f + 1 in frames]
    return samples


def list_clips(zf: zipfile.ZipFile) -> list[str]:
    return sorted(n.split("/")[0] for n in zf.namelist() if n.endswith("Labels-GameState.json"))


def _resize(jpg: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("corrupt JPEG")
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA) if img.shape[:2] != (H, W) else img


class ZipFrames:
    """Frame source decoding directly from a split zip. The zip is opened lazily in each DataLoader worker."""

    def __init__(self, ref: ZipRef):
        self.ref, self._zf = ref, None

    def __call__(self, clip: str, frame: int) -> np.ndarray:
        if self._zf is None:
            self._zf = self.ref.open()
        return _resize(self._zf.read(f"{clip}/{IM_DIR}/{frame:06d}.jpg"))

    def __getstate__(self):                # open zip handles must not be copied into worker processes
        return {"ref": self.ref, "_zf": None}


class DirFrames:
    """Frame source reading pre-extracted 1280x720 JPEGs: <root>/<clip>/<frame:06d>.jpg."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def __call__(self, clip: str, frame: int) -> np.ndarray:
        img = cv2.imread(str(self.root / clip / f"{frame:06d}.jpg"), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(self.root / clip / f"{frame:06d}.jpg")
        return img


def _extract_clip(args: tuple[ZipRef, str, list[int], str]) -> int:
    ref, clip, frames, out_root = args
    out = Path(out_root, clip)
    out.mkdir(parents=True, exist_ok=True)
    written = 0
    with ref.open() as zf:
        for n in frames:
            dst = out / f"{n:06d}.jpg"
            if not dst.exists():
                cv2.imwrite(str(dst), _resize(zf.read(f"{clip}/{IM_DIR}/{n:06d}.jpg")), [cv2.IMWRITE_JPEG_QUALITY, 92])
                written += 1
    return written


def extract_frames(ref: ZipRef, samples: list[Sample], out_root: str | Path, workers: int = 4) -> int:
    """Write every frame the samples need (t-1, t, t+1) as 1280x720 JPEGs. Idempotent: existing files are skipped."""
    needed = defaultdict(set)
    for s in samples:
        needed[s.clip].update((s.frame - 1, s.frame, s.frame + 1))
    jobs = [(ref, clip, sorted(frames), str(out_root)) for clip, frames in needed.items()]
    with ProcessPoolExecutor(workers) as pool:
        return sum(pool.map(_extract_clip, jobs))


class BallClips(Dataset):
    """(x, heatmap, target) with x = 9 x H x W (frames t-1, t, t+1), heatmap = 1 x H x W, target = (x, y, has_ball).

    train=True: random crops (usually around the ball), mirroring and brightness/contrast jitter.
    train=False: full 1280x720 frames, no augmentation.
    """

    def __init__(self, frames: Callable[[str, int], np.ndarray], samples: list[Sample], train: bool,
                 crop: tuple[int, int] = (384, 640), ball_crop_prob: float = 0.7):
        self.frames, self.samples, self.train = frames, samples, train
        self.crop_h, self.crop_w = crop
        self.ball_crop_prob = ball_crop_prob

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        img = np.concatenate([self.frames(s.clip, n) for n in (s.frame - 1, s.frame, s.frame + 1)], axis=2)
        xy = s.ball
        if self.train:
            img, xy = self._augment(img, xy)
        h, w = img.shape[:2]
        if xy is not None and not (0 <= xy[0] < w and 0 <= xy[1] < h):
            xy = None
        x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32) / 255.0)
        heat = torch.from_numpy(gaussian_heatmap(h, w, xy))[None]
        target = torch.tensor([xy[0], xy[1], 1.0] if xy is not None else [0.0, 0.0, 0.0])
        return x, heat, target

    def _augment(self, img: np.ndarray, xy):
        ch, cw = self.crop_h, self.crop_w
        if xy is not None and random.random() < self.ball_crop_prob:   # crop containing the ball, at a random offset
            x0 = int(np.clip(xy[0] - random.uniform(16, cw - 16), 0, W - cw))
            y0 = int(np.clip(xy[1] - random.uniform(16, ch - 16), 0, H - ch))
        else:                                                          # anywhere: teaches "no ball here" too
            x0, y0 = random.randint(0, W - cw), random.randint(0, H - ch)
        img = img[y0:y0 + ch, x0:x0 + cw]
        xy = None if xy is None else (xy[0] - x0, xy[1] - y0)
        if random.random() < 0.5:                                      # mirror all 3 frames identically
            img = img[:, ::-1]
            xy = None if xy is None else (cw - 1 - xy[0], xy[1])
        gain, bias = random.uniform(0.8, 1.2), random.uniform(-20, 20)
        img = np.clip(img.astype(np.float32) * gain + bias, 0, 255)
        return img, xy

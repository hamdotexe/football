"""Train BallNet (3-frame ball heatmap model) on SoccerNet GSR, end to end, on one GPU machine.

    python scripts/train_ballnet.py --out /kaggle/working/ballnet --work /tmp/ballnet --hours 3 --deadline-hours 4

Phases: environment checks -> download + frame extraction -> training (time-budgeted) with per-epoch validation
-> final score on the unseen test split (BallNet, and optionally the YOLO detector on the same frames).

Fail-fast policy: every phase validates its result. On any failed check the run logs the reason and exits with an
error immediately, so a broken run costs minutes, not hours. The log is written to <out>/train.log.

Smoke test (CPU, remote data, a few frames; exercises every code path):
    python scripts/train_ballnet.py --smoke --out /tmp/ballnet_smoke --work /tmp/ballnet_smoke_work
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import logging
import math
import os
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from footy.ball_data import (BallClips, DirFrames, Sample, ZipRef, extract_frames, list_clips,  # noqa: E402
                             load_samples)
from footy.ball_metrics import best_threshold, score  # noqa: E402
from footy.ballnet import ARCHS, build_ballnet, decode, focal_loss  # noqa: E402

log = logging.getLogger("ballnet")
REPO = "SoccerNet/SN-GSR-2025"
EXPECTED_CLIPS = {"train": 57, "valid": 58, "test": 49}


class RunError(RuntimeError):
    """A check failed. The run stops and says why, instead of wasting GPU time."""


def check(ok: bool, message: str) -> None:
    if not ok:
        raise RunError(message)


@dataclass
class Config:
    out: str
    work: str                               # scratch space for zips and extracted frames (large, not kept)
    hours: float = 3.0                      # training budget
    deadline_hours: float = 4.0             # the whole script must finish within this
    eval_reserve_min: float = 25.0          # time kept back for the final test evaluation
    batch: int = 16
    crop: tuple[int, int] = (384, 640)
    epoch_samples: int = 40_000
    lr: float = 5e-4
    init: str | None = None                 # start from these weights (e.g. a previous run's best)
    arch: str = "resnet34"                  # "small" (1.95M, from scratch) or "resnet34" (~24.5M, ImageNet encoder)
    encoder_lr_scale: float = 0.2           # pretrained encoder learns 5x slower than the new decoder
    widths: tuple[int, ...] = (32, 64, 128, 256)   # "small" only
    val_clips: int = 6                      # first 6 valid clips: validation (same clips in every run)
    train_on_valid: bool = True             # the other 52 labelled valid clips are added to training
    eval_stride: int = 5                    # validate/test on every 5th frame
    workers: int = os.cpu_count() or 4
    detector: str | None = None             # YOLO weights to compare on the same test frames
    min_free_disk_gb: float = 60.0
    min_samples_per_sec: float = 20.0
    max_bad_steps_in_row: int = 5            # non-finite losses tolerated in a row before stopping
    smoke: bool = False


class Deadline:
    def __init__(self, hours: float):
        self.start = time.time()
        self.end = self.start + hours * 3600

    def remaining(self) -> float:
        return self.end - time.time()

    def elapsed_min(self) -> float:
        return (time.time() - self.start) / 60


@contextlib.contextmanager
def phase(name: str, max_minutes: float):
    """Log a phase and fail if it took longer than its budget (a symptom of something being wrong)."""
    log.info("=== %s (budget %.0f min)", name, max_minutes)
    t = time.time()
    yield
    minutes = (time.time() - t) / 60
    log.info("=== %s done in %.1f min", name, minutes)
    check(minutes <= max_minutes, f"{name} took {minutes:.1f} min (budget {max_minutes:.0f}): too slow, stopping")


# ---------------------------------------------------------------------------------------------------------- setup
def setup_logging(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(out / "train.log")):
        handler.setFormatter(fmt)
        log.addHandler(handler)
    log.setLevel(logging.INFO)


def check_environment(cfg: Config) -> tuple[str, torch.dtype, int]:
    """Returns (device, autocast dtype, number of GPUs)."""
    Path(cfg.work).mkdir(parents=True, exist_ok=True)
    if cfg.detector:
        check(Path(cfg.detector).is_file(), f"detector weights not found: {cfg.detector}")
        import ultralytics  # noqa: F401  (fail now, not after 3 hours of training)
    if cfg.smoke:
        return "cpu", torch.float32, 0

    check(torch.cuda.is_available(), "CUDA is not available: is the GPU accelerator enabled?")
    n_gpu = torch.cuda.device_count()
    for i in range(n_gpu):
        name = torch.cuda.get_device_name(i)
        try:
            a = torch.randn(512, 512, device=f"cuda:{i}")
            check(torch.isfinite((a @ a).sum()).item(), f"GPU {i} ({name}) returned non-finite values")
        except RuntimeError as e:
            raise RunError(f"GPU {i} ({name}) is not usable with torch {torch.__version__}: {e}") from e
        log.info("GPU %d: %s, %.0f GB", i, name, torch.cuda.get_device_properties(i).total_memory / 1e9)
    free = shutil.disk_usage(cfg.work).free / 1e9
    check(free >= cfg.min_free_disk_gb, f"only {free:.0f} GB free in {cfg.work}, need {cfg.min_free_disk_gb:.0f}")
    # bf16 needs Ampere (compute capability 8+): on a T4 (7.5) torch reports it "supported" but emulates it, slowly
    major, minor = torch.cuda.get_device_capability(0)
    dtype = torch.bfloat16 if major >= 8 else torch.float16
    log.info("torch %s | %d CPU workers | %.0f GB free disk | autocast %s", torch.__version__, cfg.workers, free, dtype)
    return "cuda", dtype, n_gpu


# ----------------------------------------------------------------------------------------------------------- data
def fetch_zip(split: str, cfg: Config) -> ZipRef:
    if cfg.smoke:
        return ZipRef(f"hf://datasets/{REPO}/{split}.zip")
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(REPO, f"{split}.zip", repo_type="dataset", local_dir=str(Path(cfg.work) / "zips"))
    size = os.path.getsize(path) / 1e9
    check(size > 5, f"{split}.zip is only {size:.1f} GB: incomplete download?")
    log.info("%s.zip: %.1f GB", split, size)
    return ZipRef(path)


def split_clips(ref: ZipRef, split: str, cfg: Config) -> list[str]:
    with ref.open() as zf:
        clips = list_clips(zf)
    if not cfg.smoke:
        check(len(clips) == EXPECTED_CLIPS[split], f"{split}: {len(clips)} clips, expected {EXPECTED_CLIPS[split]}")
    return clips


def clip_samples(ref: ZipRef, name: str, clips: list[str], stride: int) -> list[Sample]:
    check(len(clips) > 0, f"{name}: no clips selected")
    with ref.open() as zf:
        samples = load_samples(zf, set(clips))[::stride]
    check(len(samples) > 0, f"{name}: no samples")
    visible = float(np.mean([s.ball is not None for s in samples]))
    check(0.7 <= visible <= 1.0, f"{name}: ball visible in {visible:.0%} of frames: label parsing looks wrong")
    log.info("%s: %d samples from %d clips, ball visible in %.0f%%", name, len(samples), len(clips), 100 * visible)
    return samples


def extract_and_verify(sources: list[tuple[ZipRef, list[Sample]]], root: Path, workers: int) -> DirFrames:
    """Extract the frames of several (zip, samples) sources into one folder, then verify every needed frame exists."""
    written = sum(extract_frames(ref, samples, root, workers) for ref, samples in sources if samples)
    needed = {(s.clip, n) for _, samples in sources for s in samples for n in (s.frame - 1, s.frame, s.frame + 1)}
    missing = [k for k in needed if not (root / k[0] / f"{k[1]:06d}.jpg").is_file()]
    check(not missing, f"{len(missing)} extracted frames missing, e.g. {missing[:3]}")
    frames = DirFrames(root)
    for clip, n in random.sample(sorted(needed), min(5, len(needed))):
        shape = frames(clip, n).shape
        check(shape == (720, 1280, 3), f"extracted frame {clip}/{n} has shape {shape}, expected (720, 1280, 3)")
    log.info("%s: %d frames on disk (%d newly written)", root.name, len(needed), written)
    return frames


def check_first_batch(x: torch.Tensor, heat: torch.Tensor, target: torch.Tensor) -> None:
    check(x.dtype == torch.uint8 and x.shape[1] == 9, f"input should be uint8 with 9 channels, got {x.dtype} {tuple(x.shape)}")
    has_ball = target[:, 2] == 1
    peaks = heat.amax(dim=(1, 2, 3))
    check(bool(torch.all(peaks[has_ball] == 1)), "heatmap peak is not 1.0 for a sample that has a ball")
    check(bool(torch.all(peaks[~has_ball] == 0)), "heatmap is non-zero for a sample without a ball")
    log.info("first batch OK: x %s, %d/%d samples with a ball", tuple(x.shape), int(has_ball.sum()), len(has_ball))


# ---------------------------------------------------------------------------------------------- train / evaluate
def autocast(device: str, dtype: torch.dtype):
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=device == "cuda")


def to_input(x: torch.Tensor, device: str, augment: bool) -> torch.Tensor:
    """uint8 frames -> float [0, 1] on the GPU; training adds per-sample brightness/contrast jitter (same for all 3 frames)."""
    x = x.to(device, non_blocking=True).float().div_(255)
    if augment:
        n = x.shape[0]
        gain = torch.empty(n, 1, 1, 1, device=device).uniform_(0.8, 1.2)
        bias = torch.empty(n, 1, 1, 1, device=device).uniform_(-0.08, 0.08)
        x = (x * gain + bias).clamp_(0, 1)
    return x.contiguous(memory_format=torch.channels_last)


def benchmark(loader: DataLoader, model: nn.Module, device: str, dtype: torch.dtype, batches: int = 20) -> None:
    """Log data-loading speed vs GPU speed, so the bottleneck is measured rather than guessed."""
    it, t = iter(loader), time.time()
    batch = [next(it) for _ in range(batches)]
    load_rate = batches * batch[0][0].shape[0] / (time.time() - t)
    x, heat = to_input(batch[0][0], device, True), batch[0][1].to(device)
    if device == "cuda":
        torch.cuda.synchronize()
    t = time.time()
    for _ in range(batches):
        with autocast(device, dtype):
            focal_loss(model(x), heat).backward()
    if device == "cuda":
        torch.cuda.synchronize()
    gpu_rate = batches * x.shape[0] / (time.time() - t)
    model.zero_grad(set_to_none=True)
    log.info("benchmark: data loader %.0f samples/s | GPU forward+backward %.0f samples/s -> bottleneck: %s",
             load_rate, gpu_rate, "data loading (CPU)" if load_rate < gpu_rate else "GPU")


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: str, dtype: torch.dtype, stop_at: float):
    """Peak (x, y, confidence) per frame. Stops early (returns complete=False) if the wall clock passes `stop_at`."""
    model.eval()
    preds, truths = [], []
    for x, _, target in loader:
        with autocast(device, dtype):
            logits = model(to_input(x, device, augment=False))
        preds += decode(logits, threshold=0.0)
        truths += [(a, b) if has else None for a, b, has in target.tolist()]
        if time.time() > stop_at:
            model.train()
            return preds, truths, False
    model.train()
    return preds, truths, True


def learning_check(losses: list[float], warm_start: bool) -> tuple[bool, str]:
    """After ~300 steps. From scratch the loss must have dropped by 20%; a warm-started model is already trained,
    so its loss can't drop that much and must only not have blown up (> +50%)."""
    first, recent = float(np.mean(losses[:20])), float(np.mean(losses[-50:]))
    if warm_start:
        return recent < 1.5 * first, f"warm-started loss rose {first:.2f} -> {recent:.2f} after {len(losses)} steps: diverging"
    return recent < 0.8 * first, f"loss {first:.2f} -> {recent:.2f} after {len(losses)} steps: not learning"


def lr_at(progress: float, step: int, cfg: Config, warmup: int = 200) -> float:
    """Linear warm-up, then cosine decay over the *time* budget (we don't know steps/hour on a new machine)."""
    if step < warmup:
        return cfg.lr * (step + 1) / warmup
    return 1e-5 + 0.5 * (cfg.lr - 1e-5) * (1 + math.cos(math.pi * min(progress, 1.0)))


def param_groups(model: nn.Module, cfg: Config) -> list[dict]:
    """Pretrained encoder weights get a smaller learning rate than the freshly initialised decoder."""
    net = model.module if isinstance(model, nn.DataParallel) else model
    pretrained = list(net.pretrained_parameters()) if hasattr(net, "pretrained_parameters") else []
    ids = {id(p) for p in pretrained}
    rest = [p for p in net.parameters() if id(p) not in ids]
    groups = [{"params": rest, "lr_scale": 1.0}]
    if pretrained:
        groups.append({"params": pretrained, "lr_scale": cfg.encoder_lr_scale})
    check(sum(len(g["params"]) for g in groups) == len(list(net.parameters())), "optimizer does not cover every parameter")
    return groups


def train(cfg: Config, model: nn.Module, train_dl: DataLoader, val_dl: DataLoader, device: str, dtype: torch.dtype,
          deadline: Deadline, out: Path) -> dict:
    budget = min(cfg.hours * 3600, deadline.remaining() - cfg.eval_reserve_min * 60)
    check(budget > (30 if cfg.smoke else 600), f"only {budget / 60:.0f} min left for training")
    log.info("training budget %.0f min", budget / 60)

    opt = torch.optim.AdamW(param_groups(model, cfg), lr=cfg.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda" and dtype == torch.float16)   # fp16 needs loss scaling
    if not cfg.smoke:
        benchmark(train_dl, model, device, dtype)
    start, step, epoch, losses = time.time(), 0, 0, []
    bad_total = bad_in_row = 0
    best = {"f1": -1.0}
    history = []

    while time.time() - start < budget:
        epoch += 1
        epoch_losses = []
        for x, heat, target in train_dl:
            if step == 0:
                check_first_batch(x, heat, target)
            lr = lr_at((time.time() - start) / budget, step, cfg)
            for g in opt.param_groups:
                g["lr"] = lr * g["lr_scale"]
            x = to_input(x, device, augment=True)
            heat = heat.to(device, non_blocking=True)
            with autocast(device, dtype):
                loss = focal_loss(model(x), heat)
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                # an isolated overflow (fp16) is skipped, like GradScaler skips inf gradients; repeated ones stop the run
                bad_total, bad_in_row = bad_total + 1, bad_in_row + 1
                log.warning("non-finite loss at step %d (lr %.1e): batch skipped (%d in a row, %d total)",
                            step, lr, bad_in_row, bad_total)
                check(bad_in_row < cfg.max_bad_steps_in_row, f"loss non-finite {bad_in_row} steps in a row: training diverged")
                check(bad_total <= max(20, 0.01 * step), f"loss non-finite on {bad_total} of {step} steps: training unstable")
                continue
            bad_in_row = 0
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            step += 1
            losses.append(loss.item())
            epoch_losses.append(loss.item())

            if not cfg.smoke:
                if step == 20:
                    t20 = time.time()
                if step == 170:   # throughput over 150 steps, after warm-up of the data loader
                    rate = 150 * cfg.batch / (time.time() - t20)
                    log.info("throughput %.0f samples/s -> ~%.0fk samples in the budget", rate, rate * budget / 1000)
                    check(rate >= cfg.min_samples_per_sec,
                          f"throughput {rate:.0f} samples/s < {cfg.min_samples_per_sec:.0f}: data loading is starving the GPU")
                if step == 300:
                    ok, message = learning_check(losses, warm_start=cfg.init is not None)
                    check(ok, message)
                if step % 200 == 0:
                    log.info("step %d | loss %.3f | lr %.1e | %.0f min", step, np.mean(losses[-200:]), lr, (time.time() - start) / 60)
            if cfg.smoke or time.time() - start > budget:
                break

        preds, truths, _ = predict(model, val_dl, device, dtype, stop_at=deadline.end)
        th, val = best_threshold(preds, truths)
        row = {"epoch": epoch, "step": step, "minutes": round((time.time() - start) / 60, 1),
               "train_loss": round(float(np.mean(epoch_losses)), 4), "lr": lr, "val_threshold": th,
               **{f"val_{k}": v for k, v in val.items()}}
        history.append(row)
        log.info("epoch %d | %s", epoch, json.dumps(row))
        check(not math.isnan(val["f1"]), "validation F1 is NaN")
        if not cfg.smoke and epoch >= 3:
            check(max(r["val_f1"] for r in history) > 0.05, "validation F1 still ~0 after 3 epochs: not learning")

        state = (model.module if isinstance(model, nn.DataParallel) else model).state_dict()
        ckpt = {"model": state, "arch": cfg.arch, "widths": cfg.widths, "threshold": th, "epoch": epoch, "val": val}
        torch.save(ckpt, out / "ballnet_last.pt")
        if val["f1"] > best["f1"]:
            best = {"f1": val["f1"], "epoch": epoch, "threshold": th, "val": val}
            torch.save(ckpt, out / "ballnet_best.pt")
        with open(out / "history.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        if cfg.smoke:
            break
    return best


def detector_predictions(weights: str, frames: DirFrames, samples: list[Sample], stop_at: float, batch: int = 16):
    """Most confident 'ball' (class 0) box centre per frame from our fine-tuned YOLO detector."""
    from ultralytics import YOLO
    model, preds = YOLO(weights), []
    for i in range(0, len(samples), batch):
        imgs = [frames(s.clip, s.frame) for s in samples[i:i + batch]]
        for r in model(imgs, imgsz=1280, conf=0.01, classes=[0], verbose=False):
            if len(r.boxes):
                j = int(r.boxes.conf.argmax())
                x1, y1, x2, y2 = r.boxes.xyxy[j].tolist()
                preds.append(((x1 + x2) / 2, (y1 + y2) / 2, float(r.boxes.conf[j])))
            else:
                preds.append(None)
        if time.time() > stop_at:
            break
    return preds


def save_examples(frames: DirFrames, samples: list[Sample], preds: list, threshold: float, path: Path, n: int = 12):
    """Zoomed crops around the ball: green = truth, red = BallNet prediction (shown only above the threshold)."""
    with_ball = [i for i, s in enumerate(samples[:len(preds)]) if s.ball is not None]
    tiles = []
    for i in random.Random(0).sample(with_ball, min(n, len(with_ball))):
        s, p = samples[i], preds[i]
        img = frames(s.clip, s.frame).copy()
        cx, cy = int(s.ball[0]), int(s.ball[1])
        cv2.circle(img, (cx, cy), 7, (0, 255, 0), 1)
        if p is not None and p[2] >= threshold:
            cv2.drawMarker(img, (int(p[0]), int(p[1])), (0, 0, 255), cv2.MARKER_CROSS, 10, 1)
        x0, y0 = int(np.clip(cx - 80, 0, 1280 - 160)), int(np.clip(cy - 48, 0, 720 - 96))
        tiles.append(cv2.resize(img[y0:y0 + 96, x0:x0 + 160], (480, 288), interpolation=cv2.INTER_NEAREST))
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [cv2.hconcat(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    cv2.imwrite(str(path), cv2.vconcat(rows))


# ----------------------------------------------------------------------------------------------------------- main
def make_model(cfg: Config) -> nn.Module:
    check(cfg.arch in ARCHS, f"unknown --arch {cfg.arch!r}, expected one of {ARCHS}")
    try:
        model = build_ballnet(cfg.arch, cfg.widths, pretrained=cfg.init is None)
    except Exception as e:   # torchvision weight download (network, disk) or a missing package
        raise RunError(f"could not build BallNet {cfg.arch!r}: {type(e).__name__}: {e}") from e
    if cfg.init:
        check(Path(cfg.init).is_file(), f"--init weights not found: {cfg.init}")
        init = torch.load(cfg.init, map_location="cpu")
        init_arch = init.get("arch", "small")
        check(init_arch == cfg.arch, f"--init weights are arch {init_arch!r}, this run is {cfg.arch!r}")
        if cfg.arch == "small":
            check(tuple(init.get("widths", cfg.widths)) == tuple(cfg.widths), "--init weights have different widths")
        model.load_state_dict(init["model"])
        log.info("initialised from %s (epoch %s, val %s)", cfg.init, init.get("epoch"), init.get("val"))
    log.info("BallNet %s: %.2fM parameters%s", cfg.arch, sum(p.numel() for p in model.parameters()) / 1e6,
             "" if cfg.init or cfg.arch == "small" else " (ImageNet-pretrained encoder)")
    return model


def run(cfg: Config) -> None:
    out, work = Path(cfg.out), Path(cfg.work)
    deadline = Deadline(cfg.deadline_hours)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    with phase("environment", 3):
        device, dtype, n_gpu = check_environment(cfg)
        model = make_model(cfg)        # downloads the ImageNet weights now: a network problem stops the run in seconds

    with phase("data", 6 if cfg.smoke else 55):
        refs = {split: fetch_zip(split, cfg) for split in ("train", "valid", "test")}
        clips = {split: split_clips(refs[split], split, cfg) for split in refs}
        for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
            check(not set(clips[a]) & set(clips[b]), f"clip names shared by {a} and {b}: frames would mix")
        n = 1 if cfg.smoke else None
        val_clips = clips["valid"][:1 if cfg.smoke else cfg.val_clips]
        extra_clips = clips["valid"][len(val_clips):][:n] if cfg.train_on_valid else []
        train_s = clip_samples(refs["train"], "train", clips["train"][:n], stride=1)
        extra_s = clip_samples(refs["valid"], "valid (training part)", extra_clips, stride=1) if extra_clips else []
        val_s = clip_samples(refs["valid"], "valid", val_clips, stride=cfg.eval_stride)
        test_s = clip_samples(refs["test"], "test", clips["test"][:n], stride=cfg.eval_stride)
        check(not {s.clip for s in extra_s} & {s.clip for s in val_s}, "a validation clip is also in training")
        if cfg.smoke:
            train_s, extra_s, val_s, test_s = train_s[300:306], extra_s[300:303], val_s[:4], test_s[:4]
        frames = {"train": extract_and_verify([(refs["train"], train_s), (refs["valid"], extra_s)], work / "frames" / "train", cfg.workers),
                  "valid": extract_and_verify([(refs["valid"], val_s)], work / "frames" / "valid", cfg.workers),
                  "test": extract_and_verify([(refs["test"], test_s)], work / "frames" / "test", cfg.workers)}
        train_s = train_s + extra_s
        log.info("training on %d samples from %d clips", len(train_s), len({s.clip for s in train_s}))
        if not cfg.smoke:
            shutil.rmtree(work / "zips", ignore_errors=True)   # frames are extracted; free ~30 GB

    loader_kw = dict(num_workers=0 if cfg.smoke else cfg.workers, pin_memory=device == "cuda",
                     persistent_workers=not cfg.smoke)
    train_ds = BallClips(frames["train"], train_s, train=True, crop=cfg.crop)
    sampler = RandomSampler(train_ds, replacement=True, num_samples=2 * cfg.batch if cfg.smoke else cfg.epoch_samples)
    train_dl = DataLoader(train_ds, batch_size=2 if cfg.smoke else cfg.batch, sampler=sampler, drop_last=True,
                          **loader_kw, **({} if cfg.smoke else {"prefetch_factor": 4}))
    val_dl = DataLoader(BallClips(frames["valid"], val_s, train=False), batch_size=4, **loader_kw)

    model = model.to(device).to(memory_format=torch.channels_last)
    if n_gpu > 1:
        model = nn.DataParallel(model)   # in-process multi-GPU: no subprocesses that can hang

    stopped = None
    try:
        with phase("training", cfg.hours * 60 + 20):
            train(cfg, model, train_dl, val_dl, device, dtype, deadline, out)
    except RunError as e:
        # a failure mid-training must not throw away the epochs already saved: still score the best checkpoint
        check((out / "ballnet_best.pt").is_file(), f"{e} (and no checkpoint was saved)")
        log.error("training stopped early: %s -> evaluating the best saved checkpoint anyway", e)
        stopped = e
    check((out / "ballnet_best.pt").is_file(), "no checkpoint was saved")

    with phase("test evaluation", cfg.eval_reserve_min + 5):
        ckpt = torch.load(out / "ballnet_best.pt", map_location=device)
        (model.module if isinstance(model, nn.DataParallel) else model).load_state_dict(ckpt["model"])
        test_dl = DataLoader(BallClips(frames["test"], test_s, train=False), batch_size=4, **loader_kw)
        stop_at = deadline.end - 300
        preds, truths, complete = predict(model, test_dl, device, dtype, stop_at=stop_at)
        result = {"ballnet": {"best_epoch": ckpt["epoch"], "threshold": ckpt["threshold"], "val": ckpt["val"],
                              "test": score(preds, truths, ckpt["threshold"])},
                  "test_complete": complete, "test_frames_scored": len(preds), "hit_radius_px_at_1280": 5.0,
                  "minutes_total": round(deadline.elapsed_min(), 1)}
        if not complete:
            log.warning("deadline reached: test scored on %d of %d frames only", len(preds), len(test_s))
        save_examples(frames["test"], test_s, preds, ckpt["threshold"], out / "examples.jpg")

        if cfg.detector and time.time() < stop_at:
            val_det = detector_predictions(cfg.detector, frames["valid"], val_s, stop_at)
            det_th, det_val = best_threshold(val_det, [s.ball for s in val_s[:len(val_det)]])
            test_det = detector_predictions(cfg.detector, frames["test"], test_s[:len(preds)], stop_at)
            result["yolo_detector"] = {"threshold": det_th, "val": det_val,
                                       "test": score(test_det, [s.ball for s in test_s[:len(test_det)]], det_th)}
        result["training_stopped_early"] = str(stopped) if stopped else None
        (out / "metrics.json").write_text(json.dumps(result, indent=2))
        log.info("RESULT %s", json.dumps(result, indent=2))
    check(stopped is None, f"training stopped early ({stopped}); test results above are from the best checkpoint")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--hours", type=float, default=Config.hours)
    ap.add_argument("--deadline-hours", type=float, default=Config.deadline_hours)
    ap.add_argument("--batch", type=int, default=Config.batch)
    ap.add_argument("--widths", type=int, nargs=4, default=Config.widths)
    ap.add_argument("--workers", type=int, default=Config.workers)
    ap.add_argument("--detector", default=None)
    ap.add_argument("--init", default=None, help="start from these BallNet weights")
    ap.add_argument("--lr", type=float, default=Config.lr)
    ap.add_argument("--arch", default=Config.arch, choices=ARCHS)
    ap.add_argument("--no-train-on-valid", action="store_true", help="train on the train split only")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = Config(out=a.out, work=a.work, hours=a.hours, deadline_hours=a.deadline_hours, batch=a.batch,
                 widths=tuple(a.widths), workers=a.workers, detector=a.detector, init=a.init, lr=a.lr, smoke=a.smoke,
                 arch=a.arch, train_on_valid=not a.no_train_on_valid)
    if cfg.smoke:
        cfg.deadline_hours, cfg.hours, cfg.eval_reserve_min, cfg.min_free_disk_gb = 0.5, 0.05, 2.0, 1.0
    setup_logging(Path(cfg.out))
    try:
        run(cfg)
    except RunError as e:
        log.error("STOPPED: %s", e)
        return 1
    except torch.cuda.OutOfMemoryError as e:
        log.error("STOPPED: CUDA out of memory (try a smaller --batch): %s", e)
        return 1
    except Exception:
        log.exception("STOPPED: unexpected error")
        return 1
    log.info("finished OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

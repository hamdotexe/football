"""Train BallNet (3-frame ball heatmap model) on SoccerNet GSR, end to end, on one GPU machine.

    python scripts/train_ballnet.py --hours 2.0 --out outputs/ballnet

Steps: GPU sanity check -> download SoccerNet zips -> train (time-budgeted cosine LR) with per-epoch validation
-> pick the decision threshold on validation -> final score on the unseen test split -> save weights + metrics.

Smoke test without a GPU or downloads:
    python scripts/train_ballnet.py --smoke
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from footy.ball_data import BallClips, ball_index  # noqa: E402
from footy.ballnet import BallNet, decode, focal_loss  # noqa: E402

HIT_PX = 5.0   # a prediction is correct if within 5 px (at 1280x720) of the true ball centre: less than one ball width


def gpu_check(device):
    if device == "cpu":
        return
    if not torch.cuda.is_available():
        sys.exit("CUDA not available: wrong machine/template?")
    name = torch.cuda.get_device_name(0)
    try:   # fails immediately if this PyTorch build doesn't support the GPU (e.g. old builds on RTX 50-series)
        (torch.randn(256, 256, device="cuda") @ torch.randn(256, 256, device="cuda")).sum().item()
    except Exception as e:
        sys.exit(f"GPU {name} is not usable with this PyTorch build: {e}")
    print(f"GPU OK: {name}, {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB, torch {torch.__version__}")


def get_zip(split, data_dir, remote):
    if remote:   # smoke test: read the zip over HTTP (range requests), no download
        from huggingface_hub import HfFileSystem
        return lambda: zipfile.ZipFile(HfFileSystem().open(f"datasets/SoccerNet/SN-GSR-2025/{split}.zip", "rb", block_size=2 ** 20))
    from huggingface_hub import hf_hub_download
    t = time.time()
    path = hf_hub_download("SoccerNet/SN-GSR-2025", f"{split}.zip", repo_type="dataset", local_dir=data_dir)
    print(f"{split}.zip ready ({os.path.getsize(path) / 1e9:.1f} GB, {time.time() - t:.0f}s)", flush=True)
    return lambda: zipfile.ZipFile(path)


def score(preds, targets):
    """preds: list of (x, y, s) or None; targets: list of (x, y) or None. Returns precision/recall/F1/error."""
    tp = fp = fn = 0
    errors = []
    for p, t in zip(preds, targets):
        d = None if p is None or t is None else math.dist(p[:2], t)
        if d is not None:
            errors.append(d)
        hit = d is not None and d <= HIT_PX
        tp += hit
        fp += p is not None and not hit
        fn += t is not None and not hit
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4),
            "median_err_px": round(float(np.median(errors)), 2) if errors else None, "frames": len(preds)}


@torch.no_grad()
def predict(model, loader, device, amp_dtype):
    """Full-frame inference. Returns per-frame peak (x, y, score) at threshold 0 (so any threshold can be applied later)."""
    model.eval()
    peaks, targets = [], []
    for x, _, t in loader:
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device == "cuda"):
            logits = model(x.to(device, non_blocking=True))
        peaks += decode(logits, threshold=0.0)
        targets += [(float(a), float(b)) if has else None for a, b, has in t.tolist()]
    model.train()
    return peaks, targets


def best_threshold(peaks, targets):
    results = {th: score([p if p[2] >= th else None for p in peaks], targets) for th in np.arange(0.1, 0.95, 0.05).round(2)}
    th = max(results, key=lambda k: results[k]["f1"])
    return float(th), results[th]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=2.0, help="training time budget")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "ballnet"))
    ap.add_argument("--data", default=str(ROOT / "data" / "soccernet_zips"))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=max(2, min(24, (os.cpu_count() or 4) - 2)))
    ap.add_argument("--epoch-samples", type=int, default=20000)
    ap.add_argument("--val-clips", type=int, default=12)
    ap.add_argument("--smoke", action="store_true", help="tiny CPU run on remote data, to test the code path")
    args = ap.parse_args()

    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    device = "cpu" if args.smoke else "cuda"
    gpu_check(device)
    amp_dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # ---- data ------------------------------------------------------------------------------------------------
    max_clips = 1 if args.smoke else None
    opener = {s: get_zip(s, args.data, args.smoke) for s in ("train", "valid", "test")}
    def items_for(split, n_clips, frame_stride):
        zf = opener[split]()
        clips = sorted({n.split("/")[0] for n in zf.namelist() if n.endswith("Labels-GameState.json")})[:n_clips]
        items = ball_index(zf, clips=set(clips))
        return items[::frame_stride]
    train_items = items_for("train", max_clips, 1)                       # every frame: fast motion matters here
    val_items = items_for("valid", 1 if args.smoke else args.val_clips, 5)
    test_items = items_for("test", max_clips, 5)                          # every 5th frame of all 49 test clips
    if args.smoke:
        val_items, test_items = val_items[:4], test_items[:4]
    print(f"train {len(train_items)} | val {len(val_items)} | test {len(test_items)} frames "
          f"(ball visible in {np.mean([i[3] is not None for i in train_items]):.0%} of train)", flush=True)

    loader_kw = dict(num_workers=0 if args.smoke else args.workers, pin_memory=device == "cuda",
                     persistent_workers=not args.smoke)
    train_ds = BallClips(None, train_items, train=True, opener=opener["train"])
    train_dl = DataLoader(train_ds, batch_size=2 if args.smoke else args.batch, drop_last=True,
                          sampler=RandomSampler(train_ds, replacement=True, num_samples=4 if args.smoke else args.epoch_samples),
                          **loader_kw)
    val_dl = DataLoader(BallClips(None, val_items, train=False, opener=opener["valid"]), batch_size=4, **loader_kw)

    # ---- train -----------------------------------------------------------------------------------------------
    model = BallNet().to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    budget = 60 if args.smoke else args.hours * 3600
    start, epoch, best_f1, history = time.time(), 0, -1.0, []
    while time.time() - start < budget:
        epoch += 1
        losses = []
        for x, y, _ in train_dl:
            frac = min(1.0, (time.time() - start) / budget)
            for g in opt.param_groups:                       # cosine learning rate over the *time* budget
                g["lr"] = 1e-5 + 0.5 * (1e-3 - 1e-5) * (1 + math.cos(math.pi * frac))
            x, y = x.to(device, non_blocking=True).to(memory_format=torch.channels_last), y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device == "cuda"):
                loss = focal_loss(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())
            if args.smoke or time.time() - start > budget:
                break

        th, val = best_threshold(*predict(model, val_dl, device, amp_dtype))
        row = {"epoch": epoch, "minutes": round((time.time() - start) / 60, 1), "train_loss": round(float(np.mean(losses)), 4),
               "lr": opt.param_groups[0]["lr"], "val_threshold": th, **{f"val_{k}": v for k, v in val.items()}}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val["f1"] > best_f1:
            best_f1 = val["f1"]
            torch.save({"model": model.state_dict(), "threshold": th, "epoch": epoch, "val": val}, out / "ballnet_best.pt")
        if args.smoke:
            break

    with open(out / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)

    # ---- final, honest score on the unseen test split -------------------------------------------------------
    ckpt = torch.load(out / "ballnet_best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    test_dl = DataLoader(BallClips(None, test_items, train=False, opener=opener["test"]), batch_size=4, **loader_kw)
    peaks, targets = predict(model, test_dl, device, amp_dtype)
    test = score([p if p[2] >= ckpt["threshold"] else None for p in peaks], targets)
    result = {"best_epoch": ckpt["epoch"], "threshold": ckpt["threshold"], "val": ckpt["val"], "test": test,
              "hit_radius_px_at_1280": HIT_PX, "train_hours": round((time.time() - start) / 3600, 2)}
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    print("TEST:", json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

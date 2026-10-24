"""Train the pitch keypoint model (YOLO-pose, 33 landmarks) on ONE GPU, fail-fast, then score it on the test split.

    python scripts/train_pitch.py --images-root .../soccernet_yolo --labels-root .../pitch_pose --out /kaggle/working/pitch

Phases (each checked; any failed check logs the reason and stops with an error):
  environment -> data assembly + label checks -> preflight (the exact training command on 5% of the data, 1 epoch:
  catches data/config/memory errors in minutes and measures speed) -> training (time-budgeted, in a child process
  with a hard timeout) -> test scoring (pose mAP + our homography error in metres, same frames as Roboflow's model).

Lessons from the first attempt (12 h, no output): no multi-GPU DDP, no "retry on failure" (a failure never starts a
second training run; if weights were saved, they are scored), and every step has a wall-clock limit.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

log = logging.getLogger("pitch")
SPLITS = ("train", "valid", "test")


class RunError(RuntimeError):
    """A check failed: stop and say why instead of wasting GPU time."""


def check(ok: bool, message: str) -> None:
    if not ok:
        raise RunError(message)


@dataclass
class Config:
    images_root: str                 # contains {train,valid,test}/images/*.jpg (SoccerNet frames, 1280x720)
    labels_root: str                 # contains {train,valid,test}/labels/*.txt + keypoints.json
    out: str
    work: str = "/tmp/pitch"
    model: str = "yolo11s-pose.pt"
    imgsz: int = 1280
    batch: int = 4                   # 1280 px pose training: batch 4 peaked at 7.6 GB on a 16 GB T4, batch 8 ran out
    workers: int = 4
    hours: float = 3.0               # training budget (Ultralytics `time`: it fits the LR schedule to it)
    deadline_hours: float = 3.7      # the whole script must finish within this
    val_stride: int = 6              # validate each epoch on every 6th valid frame (~1,300)
    test_stride: int = 8             # test scoring on every 8th test frame (the 828 Roboflow was scored on)
    train_on_valid: bool = False     # also train on the valid split, except every `holdout_every`-th clip (validation)
    holdout_every: int = 6
    warmup_epochs: float = 3.0       # Ultralytics' default; a session continuing from trained weights needs less
    warm_threads: int = 64           # read the data once with many threads before Ultralytics scans it one file at a
    warm_minutes: float = 10.0       # time (13 files/s on Kaggle's input disk: 10 min per session in session 1)
    preflight_fraction: float = 0.05
    min_epochs: int = 4              # stop after the preflight if fewer epochs fit in the budget
    max_gpu_mem_frac: float = 0.75   # stop after the preflight if training peaks above this share of GPU memory
    min_free_ram_gb: float = 2.0     # watchdog: stop training on purpose below this (before the machine is killed)
    max_output_gb: float = 15.0      # watchdog: Kaggle keeps at most 20 GB of outputs
    watch_every_s: float = 120.0
    device: str = "0"
    smoke: bool = False


class Deadline:
    def __init__(self, hours: float):
        self.end = time.time() + hours * 3600

    def remaining(self) -> float:
        return self.end - time.time()


# ------------------------------------------------------------------------------------------------- pure checks
def check_flip_idx(names: list[str], flip_idx: list[int]) -> None:
    """flip_idx must be a permutation that swaps every L_/R_ landmark with its partner and is its own inverse."""
    n = len(names)
    check(sorted(flip_idx) == list(range(n)), "flip_idx is not a permutation of the keypoints")
    for i, j in enumerate(flip_idx):
        check(flip_idx[j] == i, f"flip_idx is not symmetric at {names[i]}")
        a, b = names[i], names[j]
        if a.startswith(("L_", "R_")):
            check(b[2:] == a[2:] and a[0] != b[0], f"flip_idx maps {a} to {b}")


def check_label_line(line: str, n_kp: int) -> None:
    v = line.split()
    check(len(v) == 5 + 3 * n_kp, f"label has {len(v)} values, expected {5 + 3 * n_kp}")
    vals = [float(x) for x in v]
    check(vals[0] == 0, "class id must be 0 (pitch)")
    for k in range(n_kp):
        x, y, vis = vals[5 + 3 * k: 8 + 3 * k]
        check(vis in (0, 1, 2), f"keypoint visibility {vis} not in 0/1/2")
        if vis > 0:
            check(0 <= x <= 1 and 0 <= y <= 1, f"visible keypoint {k} at ({x}, {y}) is outside the image")


def read_results(path: Path) -> list[dict]:
    """Ultralytics results.csv -> list of {column: float} (column names stripped of padding)."""
    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")
    with open(path) as f:
        return [{k.strip(): num(v) for k, v in row.items()} for row in csv.DictReader(f)]


def clip_of(path: Path) -> str:
    return path.name.split("_")[0]           # SNGS-021_000001.jpg -> SNGS-021


def split_holdout(images: list[Path], every: int) -> tuple[list[Path], list[Path]]:
    """Whole clips, not frames: every n-th clip is held out (frames of one clip are near-duplicates, so a held-out
    frame whose neighbours were trained on would make validation look better than it is). -> (train, held out)"""
    held = set(sorted({clip_of(p) for p in images})[::every])
    return [p for p in images if clip_of(p) not in held], [p for p in images if clip_of(p) in held]


def projected_epochs(train_images: int, images_per_s: float, hours: float, overhead: float = 1.2) -> float:
    """Epochs that fit in the budget at the measured training speed (+20% for validation and data loading)."""
    return hours * 3600 / (train_images / images_per_s * overhead)


MEMORY_TROUBLE = re.compile(r"out of memory|memory error|reducing to batch", re.I)
PROGRESS = re.compile(r"^\s*\d+/\d+\s+([\d.]+)G\s.*?([\d.]+)(it/s|s/it)")
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def scan_training_output(text: str) -> dict:
    """What Ultralytics printed: memory trouble (it silently retries with a smaller batch after running out of GPU
    memory), peak GPU memory (GB) and the training speed: the median batches/s over its progress lines (the last
    line alone is slow, it includes the partial final batch)."""
    peak, rates = None, []
    for line in text.replace("\r", "\n").splitlines():
        m = PROGRESS.match(ANSI.sub("", line))
        if m:
            peak = max(peak or 0.0, float(m.group(1)))
            value = float(m.group(2))
            rates.append(value if m.group(3) == "it/s" else 1 / value)
    trouble = sorted({m.group(0).lower() for m in MEMORY_TROUBLE.finditer(text)})
    return {"memory_trouble": trouble, "peak_gpu_gb": peak,
            "batches_per_s": float(statistics.median(rates)) if rates else None}


# ------------------------------------------------------------------------------------------------- watchdog
def system_ram_gb() -> tuple[float, float] | None:
    """(available, total) system RAM in GB, or None where /proc doesn't exist (Windows)."""
    try:
        info = {line.split(":")[0]: int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()}
    except OSError:
        return None
    return info["MemAvailable"] / 1e6, info["MemTotal"] / 1e6


def tree_rss_gb(pid: int) -> float | None:
    """Resident memory of a process plus all its descendants (Ultralytics' data-loader workers), from /proc."""
    try:
        children, rss = {}, {}
        for d in Path("/proc").iterdir():
            if d.name.isdigit():
                try:
                    stat = (d / "stat").read_text()
                    ppid = int(stat.rsplit(")", 1)[1].split()[1])
                    rss[int(d.name)] = int(stat.rsplit(")", 1)[1].split()[21]) * os.sysconf("SC_PAGE_SIZE")
                    children.setdefault(ppid, []).append(int(d.name))
                except (OSError, ValueError, IndexError):
                    continue
    except (OSError, AttributeError):
        return None
    todo, total = [pid], 0
    while todo:
        p = todo.pop()
        total += rss.get(p, 0)
        todo += children.get(p, [])
    return total / 1e9


def warm_cache(paths: list[Path], threads: int, limit_s: float) -> dict:
    """Read every file once, many at a time, so the OS keeps them in memory: Ultralytics' label scan and the first
    epoch then read from RAM instead of waiting ~0.1 s per file on Kaggle's network disk. Gives up after limit_s
    (whatever was read is still cached; the rest is read slowly later, as before)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    t, done, size = time.time(), 0, 0
    pool = ThreadPoolExecutor(threads)
    try:
        for f in as_completed([pool.submit(lambda p: len(p.read_bytes()), p) for p in paths]):
            size += f.result()
            done += 1
            if time.time() - t > limit_s:
                break
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return {"files": done, "of": len(paths), "gb": round(size / 1e9, 2), "seconds": round(time.time() - t, 1)}


def dir_size_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file() and not f.is_symlink()) / 1e9


def kill_tree(proc: subprocess.Popen) -> None:
    if os.name != "nt":
        os.killpg(proc.pid, signal.SIGKILL)
    else:
        proc.kill()


class Watchdog(threading.Thread):
    """Every few minutes: log RAM (system + training processes), output size and epochs done. If free RAM or the
    output quota runs low, kill training on purpose: a controlled stop keeps the logs and the epochs saved so far,
    while Kaggle killing the whole machine (pitch runs 1 and 3) loses every output."""

    def __init__(self, proc: subprocess.Popen, cfg: "Config", results_csv: Path):
        super().__init__(daemon=True)
        self.proc, self.cfg, self.results_csv = proc, cfg, results_csv
        self.done, self.reason = threading.Event(), None

    def run(self) -> None:
        while not self.done.wait(self.cfg.watch_every_s):
            self.check_once()

    def check_once(self) -> None:
        ram, rss = system_ram_gb(), tree_rss_gb(self.proc.pid)
        out_gb = dir_size_gb(Path(self.cfg.out))
        epochs = len(read_results(self.results_csv)) if self.results_csv.exists() else 0
        log.info("watchdog | RAM free %s | training processes %s | outputs %.2f GB | epochs done %d",
                 f"{ram[0]:.1f} of {ram[1]:.1f} GB" if ram else "n/a", f"{rss:.1f} GB" if rss is not None else "n/a",
                 out_gb, epochs)
        if ram and ram[0] < self.cfg.min_free_ram_gb:
            self.reason = f"free RAM down to {ram[0]:.1f} GB (training processes {rss or 0:.1f} GB)"
        elif out_gb > self.cfg.max_output_gb:
            self.reason = f"outputs grew to {out_gb:.1f} GB (Kaggle keeps at most 20 GB)"
        if self.reason and self.proc.poll() is None:
            log.error("watchdog: %s -> stopping training now, keeping the logs and saved epochs", self.reason)
            kill_tree(self.proc)


# ------------------------------------------------------------------------------------------------- child process
def run_child(args: dict, log_path: Path, timeout_s: float, cfg: "Config | None" = None,
              results_csv: Path | None = None) -> tuple[int | None, float, str | None]:
    """Run Ultralytics train/val in a separate process with a hard timeout (a hang is killed with its data-loader
    workers, never waited on). With cfg + results_csv a Watchdog guards RAM and output size meanwhile.
    Returns (exit code or None on timeout, seconds, watchdog stop reason or None)."""
    cmd = [sys.executable, str(Path(__file__).resolve()), "--child", json.dumps(args)]
    t = time.time()
    with open(log_path, "w") as out:
        proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, start_new_session=os.name != "nt")
        dog = Watchdog(proc, cfg, results_csv) if cfg is not None else None
        if dog:
            dog.start()
        try:
            code = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            kill_tree(proc)
            proc.wait()
            code = None
        if dog:
            dog.done.set()
            dog.join()
        return code, time.time() - t, dog.reason if dog else None


def child_main(args: dict) -> int:
    mode = args.pop("mode")
    if mode == "sleep":                       # test hook: a child that hangs, to test the hard timeout
        time.sleep(args["seconds"])
        return 0
    from ultralytics import YOLO
    out_json = args.pop("out_json", None)
    model = YOLO(args.pop("model"))
    if mode == "train":
        model.train(**args)
        return 0
    m = model.val(**args)
    Path(out_json).write_text(json.dumps({"pose_mAP50": round(float(m.pose.map50), 4),
                                          "pose_mAP50-95": round(float(m.pose.map), 4),
                                          "box_mAP50": round(float(m.box.map50), 4)}))
    return 0


def tail(path: Path, n: int = 25) -> str:
    try:
        lines = path.read_text(errors="replace").replace("\r", "\n").splitlines()
    except OSError:
        return "(no output)"
    return "\n".join(line for line in lines[-n:] if line.strip())


# ------------------------------------------------------------------------------------------------- phases
def phase(name: str):
    ram = system_ram_gb()
    log.info("=== %s | RAM free %s", name, f"{ram[0]:.1f} of {ram[1]:.1f} GB" if ram else "n/a")
    return time.time()


def check_environment(cfg: Config) -> None:
    import torch
    import ultralytics
    log.info("torch %s | ultralytics %s | CUDA %s", torch.__version__, ultralytics.__version__, torch.cuda.is_available())
    if cfg.smoke:
        return
    check(torch.cuda.is_available(), "CUDA is not available: is the GPU accelerator enabled?")
    log.info("using GPU %s: %s (one GPU only: no DDP)", cfg.device, torch.cuda.get_device_name(int(cfg.device)))
    free = shutil.disk_usage(Path(cfg.work).parent).free / 1e9
    check(free > 20, f"only {free:.0f} GB free disk")


def link(src: Path, dst: Path) -> None:
    """Symlink an image next to its label (no 5 GB copy). Windows without symlink rights (local smoke test): copy."""
    if dst.exists():
        return
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy(src, dst)


def assemble_data(cfg: Config) -> tuple[Path, Path, dict]:
    """YOLO finds a label by swapping /images/ for /labels/ in the image path, so images are linked next to labels."""
    meta = json.loads((Path(cfg.labels_root) / "keypoints.json").read_text())
    names, flip_idx = meta["keypoints"], meta["flip_idx"]
    check(len(names) == 33, f"expected 33 keypoints, keypoints.json has {len(names)}")
    check_flip_idx(names, flip_idx)
    work = Path(cfg.work)
    shutil.rmtree(work, ignore_errors=True)
    counts = {}
    for split in SPLITS:
        lab_dir, img_dir = Path(cfg.labels_root) / split / "labels", Path(cfg.images_root) / split / "images"
        labels = sorted(lab_dir.glob("*.txt"))
        if not cfg.smoke:
            expected = meta["splits"][split]["kept"]
            check(len(labels) == expected, f"{split}: {len(labels)} label files, keypoints.json says {expected}")
        check(len(labels) > 0, f"{split}: no labels in {lab_dir}")
        for lab in labels[:: max(1, len(labels) // 200)]:      # format check on ~200 labels per split
            check_label_line(lab.read_text().splitlines()[0], len(names))
        (work / split / "images").mkdir(parents=True)
        (work / split / "labels").mkdir(parents=True)
        # one directory listing instead of 8,000 existence checks, and links instead of copies: Kaggle's input disk
        # is slow per file (checking and copying file by file took 39 min in run 2)
        have = set(os.listdir(img_dir)) if img_dir.is_dir() else set()
        paired = [lab for lab in labels if f"{lab.stem}.jpg" in have]
        for lab in paired:
            link(img_dir / f"{lab.stem}.jpg", work / split / "images" / f"{lab.stem}.jpg")
            link(lab, work / split / "labels" / lab.name)
        found = len(paired)
        check(found >= 0.99 * len(labels), f"{split}: only {found} of {len(labels)} labelled frames have an image in {img_dir}")
        counts[split] = found
        log.info("%s: %d frames linked", split, found)

    train_list = sorted((work / "train" / "images").glob("*.jpg"))
    valid = sorted((work / "valid" / "images").glob("*.jpg"))
    if cfg.train_on_valid:
        extra, held = split_holdout(valid, cfg.holdout_every)
        check({clip_of(p) for p in extra}.isdisjoint({clip_of(p) for p in held}), "a validation clip is also trained on")
        train_list += extra
        val_list = held[:: cfg.val_stride]
        log.info("training on train + valid: %d frames; validation: %d frames of %d held-out clips %s (every %d-th)",
                 len(train_list), len(val_list), len({clip_of(p) for p in held}),
                 sorted({clip_of(p) for p in held}), cfg.val_stride)
    else:
        val_list = valid[:: cfg.val_stride]
        log.info("validation subset: %d frames (every %d-th)", len(val_list), cfg.val_stride)
    test_list = sorted((work / "test" / "images").glob("*.jpg"))[:: cfg.test_stride]
    counts["train"] = len(train_list)
    for name, files in (("train_list", train_list), ("val_subset", val_list), ("val_tiny", val_list[:32]),
                        ("test_subset", test_list)):
        (work / f"{name}.txt").write_text("\n".join(str(p) for p in files))
    head = f"kpt_shape: [{len(names)}, 3]\nflip_idx: {flip_idx}\nnames: {{0: pitch}}\n"
    yaml = work / "pitch.yaml"
    yaml.write_text(f"train: {work}/train_list.txt\nval: {work}/val_subset.txt\ntest: {work}/test_subset.txt\n" + head)
    # preflight validates on 32 frames only: it checks that validation works, not how good the model is
    pre_yaml = work / "pitch_preflight.yaml"
    pre_yaml.write_text(f"train: {work}/train_list.txt\nval: {work}/val_tiny.txt\ntest: {work}/test_subset.txt\n" + head)

    files = [q for p in train_list + val_list + test_list for q in (p, work / p.parent.parent.name / "labels" / f"{p.stem}.txt")]
    warmed = warm_cache(files, cfg.warm_threads, cfg.warm_minutes * 60)
    log.info("read %d of %d files (%.2f GB) into memory in %.1f min (%.0f files/s)", warmed["files"], warmed["of"],
             warmed["gb"], warmed["seconds"] / 60, warmed["files"] / max(warmed["seconds"], 1e-3))
    return yaml, pre_yaml, counts


def train_args(cfg: Config, data: Path, project: Path, name: str) -> dict:
    # optimizer set explicitly: "auto" picks AdamW for the short preflight but MuSGD for the long run, so the
    # preflight would not test the real setup (run 2: the preflight passed, training ran out of GPU memory)
    return dict(mode="train", model=cfg.model, data=str(data), imgsz=cfg.imgsz, batch=cfg.batch, workers=cfg.workers,
                optimizer="SGD", lr0=0.01, momentum=0.937, warmup_epochs=cfg.warmup_epochs,
                device=cfg.device, cache=False, mosaic=0.0, fliplr=0.5, scale=0.3, translate=0.1, degrees=0.0,
                project=str(project), name=name, exist_ok=True, plots=not cfg.smoke, verbose=True)


def preflight(cfg: Config, pre_yaml: Path, train_images: int, deadline: Deadline) -> None:
    project = Path(cfg.work) / "preflight"
    args = train_args(cfg, pre_yaml, project, "run") | dict(epochs=1, fraction=cfg.preflight_fraction)
    out_txt = Path(cfg.out) / "preflight_output.txt"
    code, seconds, _ = run_child(args, out_txt, timeout_s=min(20 * 60, deadline.remaining()))
    check(code is not None, f"preflight hung for {seconds / 60:.0f} min: stopping\n{tail(out_txt)}")
    check(code == 0, f"preflight failed (exit {code}):\n{tail(out_txt)}")
    seen = scan_training_output(out_txt.read_text(errors="replace"))
    check(not seen["memory_trouble"], f"preflight ran out of GPU memory at batch {cfg.batch} ({seen['memory_trouble']}): "
          f"Ultralytics would silently retry with a smaller batch; lower --batch instead")
    rows = read_results(project / "run" / "results.csv")
    check(len(rows) == 1, f"preflight results.csv has {len(rows)} rows, expected 1")
    losses = {k: round(v, 3) for k, v in rows[0].items() if k.startswith("train/")}
    check(all(math.isfinite(v) for v in losses.values()), f"preflight training losses not finite: {losses}")
    log.info("preflight OK in %.1f min: losses %s | peak GPU memory %s GB | %s batches/s", seconds / 60, losses,
             seen["peak_gpu_gb"], seen["batches_per_s"])
    if cfg.smoke:
        return
    import torch
    total_gb = torch.cuda.get_device_properties(int(cfg.device)).total_memory / 1e9
    check(seen["peak_gpu_gb"] is not None and seen["batches_per_s"] is not None,
          "preflight printed no training progress lines: cannot check memory and speed")
    check(seen["peak_gpu_gb"] <= cfg.max_gpu_mem_frac * total_gb,
          f"preflight peaked at {seen['peak_gpu_gb']:.1f} of {total_gb:.1f} GB GPU memory (limit "
          f"{cfg.max_gpu_mem_frac:.0%}): too close to running out, lower --batch")
    images_per_s = seen["batches_per_s"] * cfg.batch
    epochs = projected_epochs(train_images, images_per_s, cfg.hours)
    log.info("speed %.1f images/s -> about %.0f epochs fit in %.1f h", images_per_s, epochs, cfg.hours)
    check(epochs >= cfg.min_epochs, f"only ~{epochs:.1f} epochs fit in {cfg.hours} h: too slow, stopping before training")


def train(cfg: Config, yaml: Path, deadline: Deadline) -> tuple[Path, dict]:
    project = Path(cfg.out) / "runs"
    args = train_args(cfg, yaml, project, "pitch") | dict(epochs=300, time=cfg.hours, patience=300)
    budget = min(cfg.hours * 3600 + 25 * 60, deadline.remaining() - 20 * 60)   # Ultralytics' own stop + margin
    check(budget > cfg.hours * 3600 * 0.8, f"only {budget / 3600:.1f} h left for a {cfg.hours} h training budget")
    log.info("training: %s, %d px, batch %d, %.1f h budget (hard kill after %.0f min)", cfg.model, cfg.imgsz, cfg.batch,
             cfg.hours, budget / 60)
    out_txt = Path(cfg.out) / "training_output.txt"
    code, seconds, stopped_by = run_child(args, out_txt, timeout_s=budget, cfg=cfg,
                                          results_csv=project / "pitch" / "results.csv")
    weights = project / "pitch" / "weights"
    best = weights / "best.pt" if (weights / "best.pt").exists() else weights / "last.pt"
    if code != 0:
        reason = (f"was stopped by the watchdog ({stopped_by})" if stopped_by else
                  "hung and was killed" if code is None else f"exited with code {code}")
        check(best.exists(), f"training {reason} and saved no weights:\n{tail(out_txt)}")
        log.error("training %s after %.0f min; scoring the saved weights %s (no retraining)\n%s", reason, seconds / 60,
                  best.name, tail(out_txt))
    trouble = scan_training_output(out_txt.read_text(errors="replace"))["memory_trouble"]
    if trouble:
        log.warning("GPU memory trouble during training (%s): Ultralytics may have lowered the batch size mid-run", trouble)
    rows = read_results(project / "pitch" / "results.csv")
    check(len(rows) >= 1, "training finished no epoch")
    key = "metrics/mAP50(P)"
    best_row = max(rows, key=lambda r: r.get(key, 0))
    summary = {"epochs": len(rows), "minutes": round(seconds / 60, 1), "best_epoch": int(best_row["epoch"]),
               "val_pose_mAP50": round(best_row.get(key, 0), 4),
               "val_pose_mAP50-95": round(best_row.get("metrics/mAP50-95(P)", 0), 4),
               "completed_normally": code == 0, "stopped_by_watchdog": stopped_by}
    for r in rows:
        log.info("epoch %d | pose loss %.3f | val pose mAP50 %.3f mAP50-95 %.3f", r["epoch"], r.get("train/pose_loss", float("nan")),
                 r.get(key, float("nan")), r.get("metrics/mAP50-95(P)", float("nan")))
    if not cfg.smoke:
        check(summary["val_pose_mAP50"] > 0.05, f"validation pose mAP50 {summary['val_pose_mAP50']} after {len(rows)} epochs: not learning")
    return best, summary


def score_test(cfg: Config, best: Path, yaml: Path, deadline: Deadline) -> dict:
    out = Path(cfg.out)
    result = {}
    # 1) our metric: homography error in metres on the same test frames Roboflow's model was scored on
    cmd = [sys.executable, str(ROOT / "scripts" / "eval_pitch_keypoints.py"), "--weights", str(best), "--mapping", "ours",
           "--images", str(Path(cfg.work) / "test" / "images"), "--labels", str(Path(cfg.work) / "test" / "labels"),
           "--out", str(out / "test_eval"), "--stride", str(cfg.test_stride), "--imgsz", str(cfg.imgsz),
           "--max-minutes", "20", "--min-frames", "2" if cfg.smoke else "50"]
    try:
        r = subprocess.run(cmd, timeout=min(30 * 60, deadline.remaining()), capture_output=True, text=True)
        ok = r.returncode == 0
        detail = (r.stdout + r.stderr)[-2000:]
    except subprocess.TimeoutExpired:
        ok, detail = False, "timed out"
    check(ok, f"test homography scoring failed:\n{detail}")
    result["test_homography"] = json.loads((out / "test_eval" / "metrics.json").read_text())
    # 2) standard pose metrics on the same test frames
    pose_json = out / "test_pose_metrics.json"
    args = dict(mode="val", model=str(best), data=str(yaml), split="test", imgsz=cfg.imgsz, batch=cfg.batch,
                device=cfg.device, project=str(out / "runs"), name="test", exist_ok=True, plots=False, out_json=str(pose_json))
    code, _, _ = run_child(args, out / "test_val_output.txt", timeout_s=min(20 * 60, deadline.remaining()))
    if code == 0 and pose_json.exists():
        result["test_pose"] = json.loads(pose_json.read_text())
    else:   # secondary metric: report, don't fail the run over it
        log.warning("Ultralytics test-split val did not finish (exit %s):\n%s", code, tail(out / "test_val_output.txt"))
    return result


def run(cfg: Config) -> None:
    deadline = Deadline(cfg.deadline_hours)
    out = Path(cfg.out)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    t = phase("environment")
    check_environment(cfg)
    t = phase("data")
    yaml, pre_yaml, counts = assemble_data(cfg)
    log.info("data done in %.1f min", (time.time() - t) / 60)
    t = phase("preflight (1 epoch on %.0f%% of the training frames)" % (100 * cfg.preflight_fraction))
    preflight(cfg, pre_yaml, counts["train"], deadline)
    t = phase("training")
    best, summary = train(cfg, yaml, deadline)
    log.info("training summary %s", json.dumps(summary))
    t = phase("test scoring")
    result = {"model": cfg.model, "imgsz": cfg.imgsz, "training": summary, **score_test(cfg, best, yaml, deadline)}
    final = out / f"pitch_{Path(cfg.model).stem}_{cfg.imgsz}.pt"
    shutil.copy(best, final)
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    log.info("RESULT %s", json.dumps(result, indent=2))
    log.info("weights: %s", final)


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        return child_main(json.loads(sys.argv[2]))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-root", required=True)
    ap.add_argument("--labels-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default=Config.work)
    ap.add_argument("--model", default=Config.model)
    ap.add_argument("--imgsz", type=int, default=Config.imgsz)
    ap.add_argument("--batch", type=int, default=Config.batch)
    ap.add_argument("--hours", type=float, default=Config.hours)
    ap.add_argument("--deadline-hours", type=float, default=Config.deadline_hours)
    ap.add_argument("--train-on-valid", action="store_true", help="train on valid too, holding out whole clips")
    ap.add_argument("--val-stride", type=int, default=Config.val_stride)
    ap.add_argument("--warmup-epochs", type=float, default=Config.warmup_epochs)
    ap.add_argument("--min-epochs", type=int, default=Config.min_epochs)
    ap.add_argument("--smoke", action="store_true", help="CPU, tiny model and images: exercises every code path")
    a = ap.parse_args()
    cfg = Config(images_root=a.images_root, labels_root=a.labels_root, out=a.out, work=a.work, model=a.model,
                 imgsz=a.imgsz, batch=a.batch, hours=a.hours, deadline_hours=a.deadline_hours,
                 train_on_valid=a.train_on_valid, holdout_every=2 if a.smoke else Config.holdout_every,
                 val_stride=a.val_stride, warmup_epochs=a.warmup_epochs, min_epochs=a.min_epochs, smoke=a.smoke)
    if cfg.smoke:
        cfg.device, cfg.workers, cfg.preflight_fraction, cfg.test_stride, cfg.val_stride = "cpu", 0, 0.5, 1, 1
        cfg.watch_every_s = 10.0
    Path(cfg.out).mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(Path(cfg.out) / "train.log")):
        handler.setFormatter(fmt)
        log.addHandler(handler)
    log.setLevel(logging.INFO)
    try:
        run(cfg)
    except RunError as e:
        log.error("STOPPED: %s", e)
        return 1
    except Exception:
        log.exception("STOPPED: unexpected error")
        return 1
    log.info("finished OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

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
import shutil
import signal
import subprocess
import sys
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
    batch: int = 16
    workers: int = 4
    hours: float = 3.0               # training budget (Ultralytics `time`: it fits the LR schedule to it)
    deadline_hours: float = 3.7      # the whole script must finish within this
    val_stride: int = 6              # validate each epoch on every 6th valid frame (~1,300)
    test_stride: int = 8             # homography scoring on every 8th test frame (the 828 Roboflow was scored on)
    preflight_fraction: float = 0.05
    min_epochs: int = 4              # stop after the preflight if fewer epochs fit in the budget
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


def projected_epochs(train_images: int, preflight_images: int, preflight_seconds: float, hours: float) -> float:
    """Epochs that fit in the budget at the preflight's speed (pessimistic: the preflight includes start-up)."""
    return hours * 3600 / (preflight_seconds * train_images / max(preflight_images, 1))


# ------------------------------------------------------------------------------------------------- child process
def run_child(args: dict, log_path: Path, timeout_s: float) -> tuple[int | None, float]:
    """Run Ultralytics train/val in a separate process with a hard timeout. A hang is killed (with its data-loader
    workers), never waited on. Returns (exit code or None on timeout, seconds)."""
    cmd = [sys.executable, str(Path(__file__).resolve()), "--child", json.dumps(args)]
    t = time.time()
    with open(log_path, "w") as out:
        proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, start_new_session=os.name != "nt")
        try:
            return proc.wait(timeout=timeout_s), time.time() - t
        except subprocess.TimeoutExpired:
            if os.name != "nt":
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
            proc.wait()
            return None, time.time() - t


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
    log.info("=== %s", name)
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
        found = 0
        for lab in labels:
            img = img_dir / f"{lab.stem}.jpg"
            if img.exists():
                link(img, work / split / "images" / img.name)
                shutil.copy(lab, work / split / "labels" / lab.name)
                found += 1
        check(found >= 0.99 * len(labels), f"{split}: only {found} of {len(labels)} labelled frames have an image in {img_dir}")
        counts[split] = found
        log.info("%s: %d frames (images linked, labels copied)", split, found)

    val_list = sorted(str(p) for p in (work / "valid" / "images").glob("*.jpg"))[:: cfg.val_stride]
    (work / "val_subset.txt").write_text("\n".join(val_list))
    head = f"kpt_shape: [{len(names)}, 3]\nflip_idx: {flip_idx}\nnames: {{0: pitch}}\n"
    yaml = work / "pitch.yaml"
    yaml.write_text(f"train: {work}/train/images\nval: {work}/val_subset.txt\ntest: {work}/test/images\n" + head)
    # preflight validates on 32 frames only: it checks that validation works, not how good the model is
    (work / "val_tiny.txt").write_text("\n".join(val_list[:32]))
    pre_yaml = work / "pitch_preflight.yaml"
    pre_yaml.write_text(f"train: {work}/train/images\nval: {work}/val_tiny.txt\ntest: {work}/test/images\n" + head)
    log.info("validation subset: %d frames (every %d-th)", len(val_list), cfg.val_stride)
    return yaml, pre_yaml, counts


def train_args(cfg: Config, data: Path, project: Path, name: str) -> dict:
    return dict(mode="train", model=cfg.model, data=str(data), imgsz=cfg.imgsz, batch=cfg.batch, workers=cfg.workers,
                device=cfg.device, cache=False, mosaic=0.0, fliplr=0.5, scale=0.3, translate=0.1, degrees=0.0,
                project=str(project), name=name, exist_ok=True, plots=not cfg.smoke, verbose=True)


def preflight(cfg: Config, pre_yaml: Path, train_images: int, deadline: Deadline) -> None:
    project = Path(cfg.work) / "preflight"
    args = train_args(cfg, pre_yaml, project, "run") | dict(epochs=1, fraction=cfg.preflight_fraction)
    code, seconds = run_child(args, Path(cfg.out) / "preflight_output.txt", timeout_s=min(20 * 60, deadline.remaining()))
    check(code is not None, f"preflight hung for {seconds / 60:.0f} min: stopping\n{tail(Path(cfg.out) / 'preflight_output.txt')}")
    check(code == 0, f"preflight failed (exit {code}):\n{tail(Path(cfg.out) / 'preflight_output.txt')}")
    rows = read_results(project / "run" / "results.csv")
    check(len(rows) == 1, f"preflight results.csv has {len(rows)} rows, expected 1")
    losses = {k: v for k, v in rows[0].items() if k.startswith("train/")}
    check(all(math.isfinite(v) for v in losses.values()), f"preflight training losses not finite: {losses}")
    pre_images = max(1, round(cfg.preflight_fraction * train_images))
    epochs = projected_epochs(train_images, pre_images, seconds, cfg.hours)
    log.info("preflight OK in %.1f min (%d frames, 1 epoch incl. start-up): losses %s -> about %.0f+ epochs fit in %.1f h",
             seconds / 60, pre_images, {k: round(v, 3) for k, v in losses.items()}, epochs, cfg.hours)
    if not cfg.smoke:
        check(epochs >= cfg.min_epochs, f"only ~{epochs:.1f} epochs fit in {cfg.hours} h: too slow, stopping before training")


def train(cfg: Config, yaml: Path, deadline: Deadline) -> tuple[Path, dict]:
    project = Path(cfg.out) / "runs"
    args = train_args(cfg, yaml, project, "pitch") | dict(epochs=300, time=cfg.hours, patience=300)
    budget = min(cfg.hours * 3600 + 25 * 60, deadline.remaining() - 20 * 60)   # Ultralytics' own stop + margin
    check(budget > cfg.hours * 3600 * 0.8, f"only {budget / 3600:.1f} h left for a {cfg.hours} h training budget")
    log.info("training: %s, %d px, batch %d, %.1f h budget (hard kill after %.0f min)", cfg.model, cfg.imgsz, cfg.batch,
             cfg.hours, budget / 60)
    out_txt = Path(cfg.out) / "training_output.txt"
    code, seconds = run_child(args, out_txt, timeout_s=budget)
    weights = project / "pitch" / "weights"
    best = weights / "best.pt" if (weights / "best.pt").exists() else weights / "last.pt"
    if code != 0:
        reason = "hung and was killed" if code is None else f"exited with code {code}"
        check(best.exists(), f"training {reason} and saved no weights:\n{tail(out_txt)}")
        log.error("training %s after %.0f min; scoring the saved weights %s (no retraining)\n%s", reason, seconds / 60,
                  best.name, tail(out_txt))
    rows = read_results(project / "pitch" / "results.csv")
    check(len(rows) >= 1, "training finished no epoch")
    key = "metrics/mAP50(P)"
    best_row = max(rows, key=lambda r: r.get(key, 0))
    summary = {"epochs": len(rows), "minutes": round(seconds / 60, 1), "best_epoch": int(best_row["epoch"]),
               "val_pose_mAP50": round(best_row.get(key, 0), 4),
               "val_pose_mAP50-95": round(best_row.get("metrics/mAP50-95(P)", 0), 4),
               "completed_normally": code == 0}
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
    # 2) standard pose metrics on the whole test split
    pose_json = out / "test_pose_metrics.json"
    args = dict(mode="val", model=str(best), data=str(yaml), split="test", imgsz=cfg.imgsz, batch=cfg.batch,
                device=cfg.device, project=str(out / "runs"), name="test", exist_ok=True, plots=False, out_json=str(pose_json))
    code, _ = run_child(args, out / "test_val_output.txt", timeout_s=min(20 * 60, deadline.remaining()))
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
    ap.add_argument("--smoke", action="store_true", help="CPU, tiny model and images: exercises every code path")
    a = ap.parse_args()
    cfg = Config(images_root=a.images_root, labels_root=a.labels_root, out=a.out, work=a.work, model=a.model,
                 imgsz=a.imgsz, batch=a.batch, hours=a.hours, deadline_hours=a.deadline_hours, smoke=a.smoke)
    if cfg.smoke:
        cfg.device, cfg.workers, cfg.preflight_fraction, cfg.test_stride, cfg.val_stride = "cpu", 0, 0.5, 1, 1
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

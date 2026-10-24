# %% [markdown]
# # ⚽ Step 7: Pitch keypoint model (for the top-down minimap)
#
# **Goal:** a model that finds **33 pitch landmarks** (corners, penalty box corners, 6-yard box corners, penalty spots,
# where the penalty arc meets the box, halfway line ends, centre circle points) in any broadcast frame.
#
# ## Why: the homography
# The pitch is flat. A flat surface seen through a camera can be "un-warped" with one 3×3 matrix, a **homography**,
# computed from ≥ 4 points whose real positions we know. With it, any pixel on the grass (a player's feet, the ball)
# converts to metres on the pitch: that's the minimap, and the in/out-of-play rules for events.
#
# ## Why train our own
# Roboflow's free pretrained pitch model, scored on 828 of our SoccerNet test frames, put landmarks a median **71 px**
# off, which makes the minimap **7.9 m** wrong (median). Not usable.
#
# ## The labels
# Nobody hand-clicked these. For each SoccerNet frame we fitted a ground-truth homography from the players' known pitch
# positions, refined it so the annotated pitch-line points lie on their lines (median error ~1.4 px), rejected badly
# fitting frames, then **projected** the 33 landmarks into the image (checked visually: they sit on the real line
# crossings).
#
# ## The model: YOLO11s-pose at 1280 px, on ONE GPU
# | Setting | Value | Why |
# |---|---|---|
# | model | YOLO11s-pose | on one T4, ~19 epochs fit in 3 h (measured 17 images/s); resolution and epochs matter more here |
# | `batch` | 4 | 1280 px pose training peaked at 7.6 GB at batch 4; batch 8 ran out of the T4's 16 GB (gradients are
# |  |  | accumulated to Ultralytics' nominal batch of 64, so the optimisation is the same) |
# | `imgsz` | 1280 | landmarks are thin-line crossings: they need pixel-level detail |
# | `mosaic` | 0 | stitching 4 frames together breaks the pitch's global geometry |
# | `fliplr` | 0.5 + `flip_idx` | a mirrored frame is valid if left/right landmarks swap names |
# | GPUs | 1 | multi-GPU (DDP) runs in separate processes that can hang: the first attempt burned 12 h |
#
# ## Fail-fast run
# The script checks every phase (inputs, label format, flip_idx), runs a **preflight** (the exact training command on
# 5% of the data for 1 epoch: catches data/config/memory errors in minutes and measures speed), trains in a child
# process with a hard timeout, and never retrains after a failure: if weights were saved, they are scored.
#
# ## Short sessions
# Two earlier attempts died with **no logs and no outputs at all**: Kaggle killed the whole machine. So training runs
# in **sessions of about 1.3 h**: each one saves its weights and logs, and the next continues from them; the worst
# case is losing one short session. A **watchdog** logs RAM every 2 minutes and, if free RAM drops below 2 GB, stops
# training on purpose (a controlled stop keeps the logs and saved epochs). In session 1 RAM stayed flat (27.5 of
# 32.9 GB free throughout), so running out of RAM is probably not what killed the earlier runs.
#
# Before training, the script reads all images once with 64 threads: Kaggle's input disk answers one file at a time
# slowly (~13 files/s), which cost session 1 ten minutes of Ultralytics scanning files one by one.

# %%
import glob, json, os, subprocess, sys
import torch
print("CUDA:", torch.cuda.is_available(), "| GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=True)
subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/hamdotexe/football.git", "/tmp/football"], check=True)
print(subprocess.run(["git", "-C", "/tmp/football", "log", "--oneline", "-1"], capture_output=True, text=True).stdout)

# %%
images_root = [d for d in glob.glob("/kaggle/input/**/soccernet_yolo", recursive=True) if os.path.isdir(f"{d}/train/images")]
labels_root = [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/keypoints.json", recursive=True)
               if os.path.isdir(os.path.join(os.path.dirname(p), "train", "labels"))]
assert images_root and labels_root, f"inputs not found: images {images_root}, labels {labels_root}"
print("images:", images_root[0], "\nlabels:", labels_root[0])

# Continue from the latest session's weights if they are attached as input, else start from COCO-pose weights.
number = lambda p: int(os.path.basename(p)[len("pitch_session"):-3])
previous = sorted(glob.glob("/kaggle/input/**/pitch_session*.pt", recursive=True), key=number)
start = previous[-1] if previous else "yolo11s-pose.pt"
session = number(previous[-1]) + 1 if previous else 1
print(f"session {session}, starting from {start}")

# %% [markdown]
# ## Train one session and score on the unseen test matches
#
# **Session 1** (train split only, 10 epochs) beat Roboflow on the test clips 116–150 (3.8 m vs 9.0 m) but found
# almost nothing on clips 187–200, filmed in stadiums the 57 training clips never show. So from session 2 on, the model
# also trains on the **valid** split (58 more clips, more stadiums), except every 6th clip, which stays out to validate.
# The **test** split is never trained on: it stays the honest benchmark.
#
# A continuing session starts from trained weights, so it needs only a short learning-rate warmup (0.5 epoch, not 3).

# %%
import shutil
cmd = [sys.executable, "-u", "/tmp/football/scripts/train_pitch.py", "--images-root", images_root[0],
       "--labels-root", labels_root[0], "--out", "/kaggle/working/pitch", "--work", "/tmp/pitch",
       "--model", start, "--imgsz", "1280", "--batch", "4", "--hours", "1.3", "--deadline-hours", "2.1"]
if session > 1:
    cmd += ["--train-on-valid", "--val-stride", "2", "--warmup-epochs", "0.5", "--min-epochs", "3"]
code = subprocess.run(cmd, cwd="/tmp").returncode       # cwd /tmp: downloaded base weights don't land in the outputs
best = sorted(glob.glob("/kaggle/working/pitch/pitch_*.pt")) or glob.glob("/kaggle/working/pitch/runs/pitch/weights/best.pt")
if best:
    shutil.copy(best[0], f"/kaggle/working/pitch_session{session}.pt")
if code != 0:
    print(open("/kaggle/working/pitch/train.log").read()[-4000:])
    raise SystemExit(f"pitch training stopped with exit code {code}: see the ERROR line above")

# %% [markdown]
# ## Results vs Roboflow's pretrained model (same 828 test frames)

# %%
m = json.load(open("/kaggle/working/pitch/metrics.json"))
h = m["test_homography"]
print(json.dumps(m, indent=2))
print(f"\n{'':32s}{'Roboflow':>12s}{'ours':>12s}")
for label, ours, theirs in [("minimap error, median (m)", h["pitch_error_m_median"], 7.93),
                            ("frames within 2 m", h["frames_within_2m"], 0.0821),
                            ("landmarks within 10 px", h["keypoint_recall_at_10px"], 0.0633),
                            ("landmark error, median (px)", h["keypoint_error_px_median"], 71.29)]:
    print(f"{label:32s}{theirs:>12}{ours:>12}")

# %%
import cv2
import matplotlib.pyplot as plt
for path, title in [("/kaggle/working/pitch/runs/pitch/results.png", "training curves"),
                    ("/kaggle/working/pitch/test_eval/examples.jpg",
                     "test frames: green = true landmarks, red = ours, yellow = pitch lines from our homography")]:
    if os.path.exists(path):
        img = cv2.imread(path)
        plt.figure(figsize=(18, 9)); plt.imshow(img[:, :, ::-1]); plt.axis("off"); plt.title(title); plt.show()

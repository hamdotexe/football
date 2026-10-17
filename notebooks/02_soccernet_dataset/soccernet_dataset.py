# %% [markdown]
# # ⚽ Step 2: Build the SoccerNet detection dataset
#
# **Goal:** turn SoccerNet Game State Reconstruction (GSR) into a YOLO training set with our 4 classes:
# `ball`, `goalkeeper`, `player`, `referee`.
#
# ## The data
# SoccerNet GSR has 30-second broadcast clips (25 fps, 1920×1080) with every player, goalkeeper, referee and the ball
# boxed in **every frame**, plus team, jersey number, pitch position and pitch lines.
#
# | Split | Clips | Use |
# |---|---|---|
# | train | 57 | training |
# | valid | 58 | validation while training (tuning, early stopping) |
# | test | 49 | **final score only**, never trained on or tuned against |
#
# ## Three rules for building a video dataset
# 1. **One class list.** SoccerNet's ids are remapped to ours; `other` (staff, ball boys) is dropped, matching Roboflow.
# 2. **Subsample.** 25 consecutive frames per second are almost identical. Every 5th frame (5 fps) keeps variety
#    without thousands of near-duplicates.
# 3. **Split by clip, never by frame.** Neighbouring frames are near-copies; if one lands in train and the other in
#    test, the test score is inflated (**data leakage**). We keep SoccerNet's official clip-level split and also
#    check whether any *match* (`game_id`) appears in more than one split.
#
# This notebook is CPU-only: it downloads, converts and resizes. No GPU needed.

# %%
!pip install -q supervision
import os, shutil, subprocess, sys
print("free disk in /tmp: %.0f GB" % (shutil.disk_usage("/tmp").free / 1e9))
print("free disk in /kaggle/working: %.0f GB" % (shutil.disk_usage("/kaggle/working").free / 1e9))

# %% [markdown]
# ## Get our code
# The conversion logic lives in our repo (`src/footy/soccernet.py`), so the exact same tested code runs locally and here.

# %%
subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/hamdotexe/football.git", "/tmp/football"], check=True)
sys.path.insert(0, "/tmp/football/src")
from footy.soccernet import convert_split, CLASS_NAMES
print(CLASS_NAMES)

# %% [markdown]
# ## Download and convert, one split at a time
# Each split zip is about 9–11 GB. We download one, convert it, then delete it before the next, so disk usage stays low.
#
# Frames are resized from 1920×1080 to **1280×720**: we'll train at `imgsz=1280`, which would shrink them to this size
# anyway, and it halves the storage.

# %%
import json, zipfile, time
from huggingface_hub import hf_hub_download

OUT = "/kaggle/working/soccernet_yolo"
RAW_LABELS = "/kaggle/working/soccernet_raw_labels"
all_stats = {}

for split in ["train", "valid", "test"]:
    t = time.time()
    zip_path = hf_hub_download("SoccerNet/SN-GSR-2025", f"{split}.zip", repo_type="dataset", local_dir="/tmp/sn")
    print(f"{split}: downloaded {os.path.getsize(zip_path) / 1e9:.1f} GB in {time.time() - t:.0f}s", flush=True)
    with zipfile.ZipFile(zip_path) as zf:
        all_stats[split] = convert_split(zf, OUT, split, stride=5, size=(1280, 720), labels_out=RAW_LABELS)
    os.remove(zip_path)
    print(f"{split}: done in {time.time() - t:.0f}s ->", {k: v for k, v in all_stats[split].items() if k != "game_ids"}, flush=True)

# %% [markdown]
# ## Leakage check: does any match appear in more than one split?
# Clips from the *same match* share stadium, kits, lighting and camera. A model can score well on a test clip just because
# it has seen that exact match in training. Ideally each match lives in exactly one split.

# %%
games = {s: set(st["game_ids"]) for s, st in all_stats.items()}
for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]:
    shared = games[a] & games[b]
    print(f"{a} ∩ {b}: {len(shared)} shared game_ids {sorted(shared, key=str)[:10]}")
for s, g in games.items():
    print(f"{s}: {len(g)} distinct matches")

# %% [markdown]
# ## Summary + YOLO config file
# `data.yaml` tells Ultralytics where the images are and what the class names are.

# %%
with open(f"{OUT}/data.yaml", "w") as f:
    f.write("path: .\ntrain: train/images\nval: valid/images\ntest: test/images\n")
    f.write(f"nc: {len(CLASS_NAMES)}\nnames: {CLASS_NAMES}\n")
json.dump(all_stats, open(f"{OUT}/stats.json", "w"), indent=2, default=list)
print(json.dumps(all_stats, indent=2, default=list))

# %% [markdown]
# ## Visual check: always look at your labels after converting them

# %%
import cv2, glob, random
import numpy as np
import supervision as sv

random.seed(0)
palette = sv.ColorPalette.from_hex(["#FFD700", "#FF1493", "#00BFFF", "#FF6347"])
tiles = []
for path in random.sample(sorted(glob.glob(f"{OUT}/train/images/*.jpg")), 4):
    img = cv2.imread(path)
    h, w = img.shape[:2]
    rows = np.loadtxt(path.replace("/images/", "/labels/").replace(".jpg", ".txt"), ndmin=2)
    cls = rows[:, 0].astype(int)
    cx, cy, bw, bh = rows[:, 1] * w, rows[:, 2] * h, rows[:, 3] * w, rows[:, 4] * h
    det = sv.Detections(xyxy=np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1), class_id=cls)
    img = sv.BoxAnnotator(color=palette, thickness=2).annotate(img, det)
    img = sv.LabelAnnotator(color=palette, text_scale=0.4, text_padding=2).annotate(img, det, [CLASS_NAMES[c] for c in cls])
    tiles.append(img)
grid = cv2.vconcat([cv2.hconcat(tiles[:2]), cv2.hconcat(tiles[2:])])
cv2.imwrite("/kaggle/working/label_check_grid.jpg", grid)
sv.plot_image(grid, (16, 9))

# %%
total = subprocess.run(["du", "-sh", OUT], capture_output=True, text=True).stdout
print("dataset size:", total)

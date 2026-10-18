# %% [markdown]
# # ⚽ Step 3: Fine-tune YOLO11m on football
#
# **Goal:** teach a COCO-pretrained YOLO11m our 4 classes (`ball`, `goalkeeper`, `player`, `referee`) using
# SoccerNet (~8.5k training frames from 57 clips) plus Roboflow (372 frames).
#
# ## Fine-tuning (transfer learning)
# We don't start from random weights. YOLO11m already knows edges, textures, human shapes and round objects from COCO.
# Fine-tuning keeps that knowledge and adjusts the network to our classes and our camera view, which needs far less
# data and time than training from scratch.
#
# ## Key training settings
# | Setting | Value | Why |
# |---|---|---|
# | `imgsz` | 1280 | the ball is ~5–15 px wide; at 640 it would shrink to a few pixels |
# | `batch` | 16 (8 per GPU) | fits a 16 GB T4 at 1280 px |
# | `device` | `[0, 1]` | **DDP**: each GPU holds a copy of the model and processes half the batch |
# | `time` | 6 h | hard time budget; Ultralytics fits the learning-rate schedule into it |
# | `patience` | 10 | stop early if validation mAP hasn't improved for 10 epochs |
#
# ## Augmentation (on by default)
# Each training image is randomly altered so the model sees "new" images every epoch:
# **mosaic** (4 images stitched into one), **scale/translate**, **horizontal flip** (a mirrored pitch is still a pitch),
# and **HSV** colour jitter (different lighting and kit colours). The last 10 epochs switch mosaic off so the model
# finishes on realistic, unstitched frames.
#
# ## Evaluation
# - **Validation** (during training): a subset of SoccerNet *valid*, used to pick the best epoch.
# - **Test** (once, at the end): SoccerNet *test*, 49 clips never seen or tuned against. **This is the honest score.**

# %%
import glob, json, os, shutil, subprocess, sys, zipfile
import torch
print("CUDA:", torch.cuda.is_available(), "| GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

# %%
!pip install -q ultralytics supervision gdown
subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/hamdotexe/football.git", "/tmp/football"], check=True)
sys.path.insert(0, "/tmp/football/src")

# %% [markdown]
# ## Locate the attached datasets
# - **SoccerNet**: output of our Step 2 notebook (`soccernet-yolo-dataset`)
# - **Roboflow**: our uploaded Kaggle dataset (`football-players-roboflow-1280`)

# %%
def find_split_dirs(root):
    """All .../<split>/images folders under root (extracting any uploaded zips first)."""
    for z in glob.glob(f"{root}/**/*.zip", recursive=True):
        dest = "/tmp/unzipped/" + os.path.basename(z)[:-4]
        if not os.path.exists(dest):
            zipfile.ZipFile(z).extractall(dest)
    dirs = glob.glob(f"{root}/**/images", recursive=True) + glob.glob("/tmp/unzipped/**/images", recursive=True)
    return sorted(set(dirs))

print(subprocess.run(["ls", "-R", "/kaggle/input"], capture_output=True, text=True).stdout[:1500])
sn_dirs = [d for d in find_split_dirs("/kaggle/input") if "soccernet_yolo" in d]
rf_dirs = [d for d in find_split_dirs("/kaggle/input") if "soccernet_yolo" not in d]
SN = {d.split("/")[-2]: d for d in sn_dirs}
print("SoccerNet:", SN)
print("Roboflow :", rf_dirs)
assert {"train", "valid", "test"} <= SN.keys() and rf_dirs, "datasets not found: check notebook inputs"

# %% [markdown]
# ## Build the combined dataset config
# - **train** = SoccerNet train + **all** Roboflow images. Roboflow overlaps our Bundesliga demo clips but not SoccerNet,
#   so it can't leak into the honest SoccerNet test score.
# - **val** = every 3rd SoccerNet *valid* frame (~2.9k images). Validating on all 8.7k every epoch would cost ~2× the time.

# %%
from footy.soccernet import CLASS_NAMES

val_imgs = sorted(glob.glob(f"{SN['valid']}/*.jpg"))[::3]
with open("/kaggle/working/val_subset.txt", "w") as f:
    f.write("\n".join(val_imgs))

with open("/kaggle/working/football.yaml", "w") as f:
    f.write("train:\n" + "".join(f"  - {d}\n" for d in [SN["train"], *rf_dirs]))
    f.write("val: /kaggle/working/val_subset.txt\n")
    f.write(f"test: {SN['test']}\n")
    f.write(f"names: {dict(enumerate(CLASS_NAMES))}\n")
print(open("/kaggle/working/football.yaml").read())
print("train images:", sum(len(os.listdir(d)) for d in [SN["train"], *rf_dirs]), "| val images:", len(val_imgs))

# %% [markdown]
# ## Train
# If two-GPU DDP fails for any reason, we fall back to a single GPU so the run isn't wasted.

# %%
from ultralytics import YOLO

args = dict(data="/kaggle/working/football.yaml", imgsz=1280, epochs=50, time=6.0, patience=10,
            project="/kaggle/working/runs", name="yolo11m_1280", exist_ok=True, workers=4, plots=True)
# Two GPUs (DDP) only if two are present. No automatic retry: a failure must stop the run, never start a second
# full training run. If training crashes after saving weights, the saved weights are used.
gpus = torch.cuda.device_count()
try:
    YOLO("yolo11m.pt").train(device=[0, 1] if gpus > 1 else 0, batch=16 if gpus > 1 else 8, **args)
except Exception as e:
    if not os.path.exists("/kaggle/working/runs/yolo11m_1280/weights/last.pt"):
        raise
    print("training raised after saving weights; continuing with saved weights:", repr(e))

BEST = "/kaggle/working/runs/yolo11m_1280/weights/best.pt"
print("best weights:", BEST, os.path.exists(BEST))

# %% [markdown]
# ## The honest score: SoccerNet test (49 unseen clips)
# - **Precision**: of the boxes we predicted, how many were right?
# - **Recall**: of the real objects, how many did we find?
# - **mAP50**: average precision with a box counted as correct if it overlaps the truth by ≥ 50% (IoU ≥ 0.5)
# - **mAP50-95**: the same, averaged over stricter overlaps 0.50 → 0.95 (rewards tight boxes)

# %%
model = YOLO(BEST)
m = model.val(data="/kaggle/working/football.yaml", split="test", imgsz=1280, batch=16, device=0,
              project="/kaggle/working/runs", name="test_eval", exist_ok=True)
test_metrics = {"all": {"precision": float(m.box.mp), "recall": float(m.box.mr),
                        "mAP50": float(m.box.map50), "mAP50-95": float(m.box.map)}}
for i, c in enumerate(m.box.ap_class_index):
    p, r, ap50, ap = m.box.class_result(i)
    test_metrics[CLASS_NAMES[c]] = {"precision": p, "recall": r, "mAP50": ap50, "mAP50-95": ap}
test_metrics = {k: {kk: round(float(vv), 3) for kk, vv in v.items()} for k, v in test_metrics.items()}
json.dump(test_metrics, open("/kaggle/working/test_metrics.json", "w"), indent=2)
print(json.dumps(test_metrics, indent=2))

# %% [markdown]
# ## Before vs after, on the same unseen test frames
# COCO has no goalkeeper/referee classes, so for a fair comparison both models are scored on two shared groups:
# **ball**, and **person** (player + goalkeeper + referee).

# %%
from footy.evaluate import grouped_map, COCO_TO_GROUP, OURS_TO_GROUP

test_sample = sorted(glob.glob(f"{SN['test']}/*.jpg"))[::12]   # ~600 frames spread over all 49 clips
comparison = {
    "frames": len(test_sample),
    "coco_baseline_yolo11m": grouped_map(YOLO("yolo11m.pt"), test_sample, COCO_TO_GROUP),
    "finetuned_yolo11m": grouped_map(model, test_sample, OURS_TO_GROUP),
}
json.dump(comparison, open("/kaggle/working/comparison.json", "w"), indent=2)
print(json.dumps(comparison, indent=2))

# %% [markdown]
# ## Demo video: same Bundesliga clip as Step 1, now with football classes
# Colours: **player** blue, **goalkeeper** pink, **referee** red ellipses at the feet; **ball** yellow triangle.

# %%
import supervision as sv

subprocess.run(["gdown", "-q", "-O", "/tmp/08fd33_0.mp4", "https://drive.google.com/uc?id=1OG8K6wqUw9t7lp9ms1M48DxRhwTYciK-"])
info = sv.VideoInfo.from_video_path("/tmp/08fd33_0.mp4")
palette = sv.ColorPalette.from_hex(["#FFD700", "#FF1493", "#00BFFF", "#FF6347"])
ellipse = sv.EllipseAnnotator(color=palette, thickness=2)
triangle = sv.TriangleAnnotator(color=sv.Color.from_hex("#FFD700"), base=20, height=17)
label = sv.LabelAnnotator(color=palette, text_scale=0.4, text_padding=2, text_position=sv.Position.BOTTOM_CENTER)

with sv.VideoSink("/tmp/demo_raw.mp4", info) as sink:
    for i, frame in enumerate(sv.get_video_frames_generator("/tmp/08fd33_0.mp4", end=int(info.fps * 10))):
        det = sv.Detections.from_ultralytics(model(frame, imgsz=1280, conf=0.3, verbose=False)[0])
        people, ball = det[det.class_id != 0], det[det.class_id == 0]
        out = ellipse.annotate(frame.copy(), people)
        out = label.annotate(out, people, [CLASS_NAMES[c] for c in people.class_id])
        out = triangle.annotate(out, ball)
        sink.write_frame(out)
        if i == int(info.fps * 5):
            import cv2; cv2.imwrite("/kaggle/working/step3_sample_frame.jpg", out)

subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", "/tmp/demo_raw.mp4", "-vcodec", "libx264", "-crf", "26",
                "/kaggle/working/step3_finetuned_demo.mp4"], check=True)
from IPython.display import Video
Video("/kaggle/working/step3_finetuned_demo.mp4", embed=True, width=960)

# %% [markdown]
# ## Keep the outputs small
# Only the best weights, metrics, plots and demo are kept; the base COCO weights downloaded into the working dir are removed.

# %%
for f in glob.glob("/kaggle/working/*.pt"):
    os.remove(f)
shutil.copy(BEST, "/kaggle/working/football_yolo11m_1280.pt")
print(subprocess.run(["du", "-sh", "/kaggle/working"], capture_output=True, text=True).stdout)

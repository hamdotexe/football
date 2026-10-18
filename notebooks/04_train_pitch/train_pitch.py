# %% [markdown]
# # ⚽ Step 5: Pitch keypoint model (for the top-down minimap)
#
# **Goal:** a model that finds **33 pitch landmarks** (corners, penalty box corners, 6-yard box corners, penalty spots,
# where the penalty arc meets the box, halfway line ends, centre circle points) in any broadcast frame.
#
# ## Why: the homography
# The pitch is flat. A flat surface seen through a camera can be "un-warped" with one 3×3 matrix, a **homography**,
# computed from ≥ 4 points whose real positions we know. Once we have it, any pixel on the grass (e.g. a player's feet)
# converts to metres on the pitch, which is how the minimap is drawn.
#
# ## The labels
# No one hand-clicked these keypoints. For each SoccerNet frame we fitted a ground-truth homography from the players'
# known pitch positions, refined it so the annotated pitch-line points lie exactly on their lines (median error ~1.4 px),
# rejected badly-fitting frames, then **projected** the 33 landmarks into the image.
#
# ## The model: YOLO-pose
# Same YOLO family as the detector, but each detection also predicts a list of keypoints, each with a confidence.
# It's usually used for human skeletons; here there's one "object", the pitch, and its "skeleton" is the 33 landmarks.
#
# | Setting | Value | Why |
# |---|---|---|
# | `mosaic` | 0 | stitching 4 frames together breaks the pitch's global geometry |
# | `fliplr` | 0.5 + `flip_idx` | a mirrored frame is valid if left/right landmarks swap names |
# | `imgsz` | 1280 | keypoints need pixel-level precision |
# | `time` | 4 h | time budget |

# %%
import glob, json, os, shutil, subprocess, sys, zipfile
import numpy as np
import torch
print("CUDA:", torch.cuda.is_available(), "| GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

# %%
!pip install -q ultralytics supervision
subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/hamdotexe/football.git", "/tmp/football"], check=True)
sys.path.insert(0, "/tmp/football/src")
from footy.pitch import KEYPOINT_NAMES, homography_from_keypoints, project, pitch_polylines
from footy.minimap import Minimap

# %% [markdown]
# ## Assemble the dataset
# Images come from our SoccerNet detection dataset (Step 2 output); keypoint labels from the
# `soccernet-pitch-keypoints` dataset. We symlink the images (no copying 5 GB) next to the pose labels, because YOLO finds
# each label by swapping `/images/` for `/labels/` in the image path.

# %%
for z in glob.glob("/kaggle/input/**/*.zip", recursive=True):
    zipfile.ZipFile(z).extractall("/tmp/unzipped/" + os.path.basename(z)[:-4])

img_root = glob.glob("/kaggle/input/**/soccernet_yolo", recursive=True)[0]
raw_root = glob.glob("/kaggle/input/**/soccernet_raw_labels", recursive=True)[0]
meta_path = (glob.glob("/kaggle/input/**/keypoints.json", recursive=True) + glob.glob("/tmp/unzipped/**/keypoints.json", recursive=True))[0]
meta = json.load(open(meta_path))
label_dirs = {d.split("/")[-2]: d for d in glob.glob("/kaggle/input/**/labels", recursive=True) + glob.glob("/tmp/unzipped/**/labels", recursive=True)
              if "soccernet_yolo" not in d and d.split("/")[-2] in ("train", "valid", "test")}
print("images:", img_root, "\nraw labels:", raw_root, "\npose labels:", label_dirs)

DATA = "/tmp/pitch"
for split, lab_dir in label_dirs.items():
    os.makedirs(f"{DATA}/{split}/images", exist_ok=True)
    shutil.copytree(lab_dir, f"{DATA}/{split}/labels", dirs_exist_ok=True)
    for lab in os.listdir(lab_dir):
        src = f"{img_root}/{split}/images/{lab[:-4]}.jpg"
        dst = f"{DATA}/{split}/images/{lab[:-4]}.jpg"
        if os.path.exists(src) and not os.path.exists(dst):
            os.symlink(src, dst)
    print(split, len(os.listdir(f"{DATA}/{split}/images")), "images")

val_subset = sorted(glob.glob(f"{DATA}/valid/images/*.jpg"))[::3]
open(f"{DATA}/val_subset.txt", "w").write("\n".join(val_subset))
with open(f"{DATA}/pitch.yaml", "w") as f:
    f.write(f"train: {DATA}/train/images\nval: {DATA}/val_subset.txt\ntest: {DATA}/test/images\n")
    f.write(f"kpt_shape: [{len(meta['keypoints'])}, 3]\nflip_idx: {meta['flip_idx']}\nnames: {{0: pitch}}\n")
print(open(f"{DATA}/pitch.yaml").read())

# %% [markdown]
# ## Train

# %%
from ultralytics import YOLO

args = dict(data=f"{DATA}/pitch.yaml", imgsz=1280, epochs=60, time=4.0, patience=15, mosaic=0.0, fliplr=0.5,
            scale=0.3, translate=0.1, degrees=0.0, project="/kaggle/working/runs", name="pitch_pose", exist_ok=True,
            workers=4, plots=True)
# Two GPUs (DDP) only if two are present. No automatic retry: a failure must stop the run, never start a second
# full training run. If training crashes after saving weights, the saved weights are used.
gpus = torch.cuda.device_count()
try:
    YOLO("yolo11m-pose.pt").train(device=[0, 1] if gpus > 1 else 0, batch=16 if gpus > 1 else 8, **args)
except Exception as e:
    if not os.path.exists("/kaggle/working/runs/pitch_pose/weights/last.pt"):
        raise
    print("training raised after saving weights; continuing with saved weights:", repr(e))
BEST = "/kaggle/working/runs/pitch_pose/weights/best.pt"
model = YOLO(BEST)

# %% [markdown]
# ## The metric that matters: how far off are players on the minimap?
# For test frames (3 matches never seen in training) we:
# 1. predict the 33 keypoints → homography (needs ≥ 4 confident keypoints),
# 2. map every player's feet from pixels to metres with it,
# 3. compare with SoccerNet's ground-truth pitch position.
#
# This tests the whole chain (keypoints → homography → position) in units everyone understands: **metres**.

# %%
SCALE = 1280 / 1920   # raw labels are in 1920x1080 pixels; our frames are 1280x720
test_imgs = sorted(glob.glob(f"{DATA}/test/images/*.jpg"))[::4]
raw_cache, errors, frames_ok, per_frame = {}, [], 0, []
for path in test_imgs:
    clip, frame = os.path.basename(path)[:-4].split("_")
    if clip not in raw_cache:
        d = json.load(open(f"{raw_root}/test/{clip}.json"))
        by = {}
        for a in d["annotations"]:
            by.setdefault(a["image_id"], []).append(a)
        raw_cache[clip] = ({im["file_name"][:-4]: im["image_id"] for im in d["images"]}, by)
    ids, by = raw_cache[clip]
    r = model(path, imgsz=1280, conf=0.25, verbose=False)[0]
    if r.keypoints is None or len(r.keypoints) == 0:
        continue
    H, _ = homography_from_keypoints(r.keypoints.xy[0].cpu().numpy(), r.keypoints.conf[0].cpu().numpy())
    if H is None:
        continue
    frames_ok += 1
    people = [a for a in by.get(ids[frame], []) if a.get("category_id") in (1, 2, 3) and a.get("bbox_pitch")]
    if not people:
        continue
    feet = np.float64([[(a["bbox_image"]["x"] + a["bbox_image"]["w"] / 2) * SCALE, (a["bbox_image"]["y"] + a["bbox_image"]["h"]) * SCALE] for a in people])
    truth = np.float64([[a["bbox_pitch"]["x_bottom_middle"], a["bbox_pitch"]["y_bottom_middle"]] for a in people])
    e = np.linalg.norm(project(H, feet) - truth, axis=1)
    errors += e.tolist()
    per_frame.append((path, float(np.median(e))))

errors = np.array(errors)
position_metrics = {
    "test_frames": len(test_imgs), "homography_found_pct": round(100 * frames_ok / len(test_imgs), 1),
    "player_position_error_m": {"median": round(float(np.median(errors)), 2), "mean": round(float(errors.mean()), 2),
                                "p90": round(float(np.percentile(errors, 90)), 2)},
    "players_within_1m_pct": round(100 * float((errors < 1).mean()), 1),
    "players_within_2m_pct": round(100 * float((errors < 2).mean()), 1),
}
json.dump(position_metrics, open("/kaggle/working/pitch_position_metrics.json", "w"), indent=2)
print(json.dumps(position_metrics, indent=2))

# %% [markdown]
# ## Standard pose metrics on the test split (for completeness)

# %%
m = model.val(data=f"{DATA}/pitch.yaml", split="test", imgsz=1280, batch=16, device=0,
              project="/kaggle/working/runs", name="pitch_test_eval", exist_ok=True)
pose_metrics = {"pose_mAP50": round(float(m.pose.map50), 3), "pose_mAP50-95": round(float(m.pose.map), 3)}
json.dump(pose_metrics, open("/kaggle/working/pitch_pose_metrics.json", "w"), indent=2)
print(pose_metrics)

# %% [markdown]
# ## Look at it: pitch lines re-projected from the predicted homography + minimap (truth white, ours red)

# %%
import cv2
mm = Minimap(px_per_m=6)
rows = []
for path, _ in sorted(per_frame, key=lambda t: t[1])[len(per_frame) // 2::max(1, len(per_frame) // 8)][:4]:
    img = cv2.imread(path)
    r = model(img, imgsz=1280, conf=0.25, verbose=False)[0]
    H, _ = homography_from_keypoints(r.keypoints.xy[0].cpu().numpy(), r.keypoints.conf[0].cpu().numpy())
    H_inv = np.linalg.inv(H)
    for line in pitch_polylines():
        cv2.polylines(img, [project(H_inv, line).astype(np.int32)], False, (0, 0, 255), 2, cv2.LINE_AA)
    clip, frame = os.path.basename(path)[:-4].split("_")
    ids, by = raw_cache[clip]
    people = [a for a in by.get(ids[frame], []) if a.get("category_id") in (1, 2, 3) and a.get("bbox_pitch")]
    feet = np.float64([[(a["bbox_image"]["x"] + a["bbox_image"]["w"] / 2) * SCALE, (a["bbox_image"]["y"] + a["bbox_image"]["h"]) * SCALE] for a in people])
    truth = np.float64([[a["bbox_pitch"]["x_bottom_middle"], a["bbox_pitch"]["y_bottom_middle"]] for a in people])
    radar = mm.draw(truth, (255, 255, 255), radius=6)
    radar = mm.draw(project(H, feet), (0, 0, 255), radius=4, canvas=radar)
    radar = cv2.resize(radar, (int(radar.shape[1] * 720 / radar.shape[0]), 720))
    rows.append(cv2.hconcat([img, radar]))
grid = cv2.vconcat(rows)
cv2.imwrite("/kaggle/working/pitch_examples.jpg", grid)
import supervision as sv
sv.plot_image(grid, (16, 20))

# %%
for f in glob.glob("/kaggle/working/*.pt"):
    os.remove(f)
shutil.copy(BEST, "/kaggle/working/pitch_yolo11m_pose_1280.pt")
print(subprocess.run(["du", "-sh", "/kaggle/working"], capture_output=True, text=True).stdout)

# %% [markdown]
# # ⚽ Step 1: Baseline detection with an off-the-shelf model
#
# **Goal:** measure how well a *generic* object detector handles football footage **before** we build anything
# football-specific. This gives us a **baseline**, the "before" number every later improvement is compared to.
#
# ## How an object detector works (YOLO)
# A detector takes an image and returns a list of `[box (x1, y1, x2, y2), class, confidence]`.
# **YOLO** ("You Only Look Once") does this in a single pass through a neural network:
#
# 1. **Backbone:** a convolutional neural network (CNN) turns the image into *feature maps*, grids where each cell
#    describes "what's here" (edges → textures → body parts → whole people).
# 2. **Head:** for every grid cell, predicts whether an object is centred there, its box, and its class.
# 3. **NMS (non-max suppression):** many cells fire for the same object; NMS keeps the best box and removes duplicates.
#
# ## Pretrained on COCO
# `yolo11m.pt` (**m** = medium) was trained on **COCO**, a dataset of 80 everyday classes. Two are relevant:
# `person` (id 0) and `sports ball` (id 32).
#
# ## What we expect to go wrong
# - It can't separate **player / goalkeeper / referee**; to COCO they're all just `person`.
# - It will box **crowd, coaches and staff** too.
# - It will often **miss the ball**: tiny, motion-blurred, and unlike COCO's close-up balls.
#
# Measuring these failures is what justifies **fine-tuning** in the next step.

# %%
# Is the GPU there? (always check first)
import torch
print("PyTorch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {p.name}, {p.total_memory / 1e9:.1f} GB VRAM")

# %% [markdown]
# ## Install libraries
# - **ultralytics:** YOLO models (training + inference)
# - **supervision:** detection utilities: drawing boxes, trackers, video reading/writing
# - **gdown:** downloads files from Google Drive

# %%
!pip install -q ultralytics supervision gdown

# %% [markdown]
# ## Get test footage
# Five 30-second Bundesliga clips (DFL), re-hosted by Roboflow's open-source `sports` project.
#
# Kaggle folders:
# - `/kaggle/working/`: **outputs we keep** (saved with the notebook version)
# - `/tmp/`: scratch space, wiped when the session ends; good for re-downloadable files

# %%
import os, glob, subprocess

os.makedirs("/tmp/clips", exist_ok=True)
CLIPS = {
    "0bfacc_0.mp4": "12TqauVZ9tLAv8kWxTTBFWtgt2hNQ4_ZF",
    "2e57b9_0.mp4": "19PGw55V8aA6GZu5-Aac5_9mCy3fNxmEf",
    "08fd33_0.mp4": "1OG8K6wqUw9t7lp9ms1M48DxRhwTYciK-",
    "573e61_0.mp4": "1yYPKuXbHsCxqjA9G-S6aeR2Kcnos8RPU",
    "121364_0.mp4": "1vVwjW1dE1drIdd4ZSILfbCGPD4weoNiu",
}
for name, file_id in CLIPS.items():
    path = f"/tmp/clips/{name}"
    if not os.path.exists(path):
        subprocess.run(["gdown", "-q", "-O", path, f"https://drive.google.com/uc?id={file_id}"])

videos = sorted(glob.glob("/tmp/clips/*.mp4"))
for v in videos:
    print(f"{os.path.getsize(v) / 1e6:6.1f} MB  {v}")
assert videos, "No clips downloaded: Google Drive may be rate-limiting. Retry later."

# %% [markdown]
# ## Run the COCO-pretrained detector on 10 seconds of video
#
# Key settings:
# - **`imgsz=1280`:** each frame is resized to 1280 px before entering the network. Bigger keeps more pixels on
#   small objects (the ball!) but is slower. At 640 px a 1080p frame is shrunk 3x and the ball nearly vanishes.
# - **`conf=0.25`:** discard predictions the model is less than 25% confident about.
#
# We record two baseline numbers:
# 1. **% of frames where the ball was detected** (it's on screen almost always, so ideally close to 100%)
# 2. **average `person` boxes per frame** (at most 22 players + 3 referees on the pitch; a higher number means
#    crowd/staff false positives)

# %%
import json
import supervision as sv
from ultralytics import YOLO

VIDEO = videos[0]
info = sv.VideoInfo.from_video_path(VIDEO)
print(info)

model = YOLO("yolo11m.pt")
PERSON, BALL = 0, 32

box_annotator = sv.BoxAnnotator(color=sv.Color.from_hex("#00BFFF"), thickness=2)
ball_annotator = sv.TriangleAnnotator(color=sv.Color.from_hex("#FFD700"), base=20, height=17)

N = int(info.fps * 10)
frames_with_ball, people_counts = 0, []
OUT = "/kaggle/working/step1_baseline_raw.mp4"

with sv.VideoSink(OUT, info) as sink:
    for i, frame in enumerate(sv.get_video_frames_generator(VIDEO, end=N)):
        result = model(frame, imgsz=1280, conf=0.25, verbose=False)[0]
        det = sv.Detections.from_ultralytics(result)

        people = det[det.class_id == PERSON]
        ball = det[det.class_id == BALL]
        people_counts.append(len(people))
        frames_with_ball += len(ball) > 0

        annotated = box_annotator.annotate(frame.copy(), people)
        annotated = ball_annotator.annotate(annotated, ball)
        sink.write_frame(annotated)
        if i == N // 2:
            sample = annotated

stats = {
    "video": os.path.basename(VIDEO),
    "frames": N,
    "ball_detected_pct": round(100 * frames_with_ball / N, 1),
    "avg_person_boxes": round(sum(people_counts) / len(people_counts), 1),
    "max_person_boxes": max(people_counts),
}
json.dump(stats, open("/kaggle/working/step1_stats.json", "w"), indent=2)
print(json.dumps(stats, indent=2))

import cv2
cv2.imwrite("/kaggle/working/step1_sample_frame.jpg", sample)
sv.plot_image(sample, (16, 9))

# %% [markdown]
# ## Watch the result
# OpenCV writes video with the `mp4v` codec, which browsers can't play. We re-encode to **H.264**, the web standard.

# %%
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", OUT, "-vcodec", "libx264", "-crf", "28",
                "/kaggle/working/step1_baseline.mp4"], check=True)
os.remove(OUT)
from IPython.display import Video
Video("/kaggle/working/step1_baseline.mp4", embed=True, width=960)

# %% [markdown]
# ## What to look for
# - Are the **crowd / coaches** boxed? (inflated person count)
# - Does the **ball** flicker on and off, or vanish entirely?
# - Are overlapping players merged into one box or lost?
# - Is the referee indistinguishable from players? (always, as COCO has no "referee" class)
#
# **Next step:** build a football-specific dataset (SoccerNet + Roboflow) and **fine-tune** YOLO with classes
# `player`, `goalkeeper`, `referee`, `ball`, then compare against these baseline numbers.

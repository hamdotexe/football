# %% [markdown]
# # ⚽ Showcase: the whole pipeline on 14 clips no model was trained on
#
# Every piece runs the full pipeline as it is today:
# - **detection** of players, goalkeepers, referees and the ball (YOLO11m fine-tuned at 1280 px)
# - **tracking** (ByteTrack) with a role vote per track, so roles don't flicker
# - **teams** from shirt colours (SigLIP embeddings → UMAP → 2 clusters), fitted per piece because every match differs
# - **ball** marker and trail (the YOLO ball detections; BallNet-L isn't in the pipeline yet)
# - **minimap** (top left) from the pitch keypoint model → homography, held steady over time
#
# The pieces are 10 s each: every 4th of SoccerNet's 49 **test** clips (never used to train any model here) plus one
# Bundesliga clip (another league entirely). A caption names each source.

# %%
import glob, json, os, subprocess, sys
SMOKE = False   # True: CPU, 1 s pieces of 2 clips (checks installs, inputs and every step without using GPU quota)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "supervision==0.31.0", "trackers==2.6.1",
                "umap-learn", "sentencepiece"], check=True)
subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/hamdotexe/football.git", "/tmp/football"], check=True)
print(subprocess.run(["git", "-C", "/tmp/football", "log", "--oneline", "-1"], capture_output=True, text=True).stdout)
sys.path.insert(0, "/tmp/football/src")
import footy.pipeline, footy.calibration   # every pipeline import works before the 9 GB download starts

inputs = {name: glob.glob(f"/kaggle/input/**/{name}", recursive=True)
          for name in ("football_yolo11m_1280.pt", "pitch_keypoints.pt", "0bfacc_0.mp4")}
assert all(inputs.values()), f"inputs not found: {inputs}"
print(inputs)

# %%
from huggingface_hub import hf_hub_download
test_zip = hf_hub_download("SoccerNet/SN-GSR-2025", "test.zip", repo_type="dataset", local_dir="/tmp/zips")
print(test_zip, round(os.path.getsize(test_zip) / 1e9, 1), "GB")

# %%
cmd = [sys.executable, "-u", "/tmp/football/scripts/make_showcase.py",
       "--weights", inputs["football_yolo11m_1280.pt"][0], "--pitch-weights", inputs["pitch_keypoints.pt"][0],
       "--soccernet-zip", test_zip, "--video", inputs["0bfacc_0.mp4"][0], "--out", "/kaggle/working/showcase"]
cmd += (["--every", "25", "--seconds", "1", "--device", "cpu", "--max-minutes", "40"] if SMOKE else
        ["--every", "4", "--seconds", "10", "--device", "0", "--max-minutes", "40"])
code = subprocess.run(cmd).returncode
if code != 0:
    raise SystemExit(f"showcase stopped with exit code {code}: see the ERROR line above")

# %% [markdown]
# ## How often each piece got a minimap

# %%
for s in json.load(open("/kaggle/working/showcase/stats.json")):
    print(f"{s['minimap_shown']:6.0%} minimap | {s['own_homography']:6.0%} own estimate | "
          f"{s['frames'] / s['seconds']:5.1f} frames/s | {s['caption']}")

# %%
import cv2
import matplotlib.pyplot as plt
cap = cv2.VideoCapture("/kaggle/working/showcase/showcase.mp4")
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fig, axes = plt.subplots(2, 2, figsize=(20, 11.5))
for ax, i in zip(axes.ravel(), [n // 8, 3 * n // 8, 5 * n // 8, 7 * n // 8]):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, f = cap.read()
    ax.imshow(f[:, :, ::-1]); ax.axis("off"); ax.set_title(f"frame {i}")
plt.tight_layout(); plt.show()

# %% [markdown]
# # ⚽ Step 7a: is Roboflow's free pitch model good enough? (so we may not need to train our own)
#
# The minimap needs the **homography**: the mapping between image pixels and pitch metres. It comes from pitch
# landmarks found in each frame (corners, box corners, centre circle points...). Training our own landmark model costs
# ~4 GPU hours, so first we score Roboflow's free pretrained one (YOLOv8x-pose, 32 landmarks) on our SoccerNet test
# frames. It runs on CPU here: no GPU quota used.
#
# Their landmarks are matched to ours by physical position (checked with a unit test against their geometry).
# Two scores:
# - **keypoints**: share of visible landmarks found within 10 px (at 1280x720)
# - **homography**: fit pixels->pitch from the predicted landmarks, map the *true* landmarks onto the pitch, and
#   measure how many **metres** off they land. That is the error you'd see on the minimap.

# %%
import glob, hashlib, json, os, subprocess, sys, zipfile
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "gdown"], check=True)
subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/hamdotexe/football.git", "/tmp/football"], check=True)
print(subprocess.run(["git", "-C", "/tmp/football", "log", "--oneline", "-1"], capture_output=True, text=True).stdout)
sys.path.insert(0, "/tmp/football/src")
from footy.roboflow_pitch import WEIGHTS_GDRIVE_ID, WEIGHTS_SHA256

# %%
weights = "/tmp/football-pitch-detection.pt"
subprocess.run([sys.executable, "-m", "gdown", "-q", "-O", weights, f"https://drive.google.com/uc?id={WEIGHTS_GDRIVE_ID}"], check=True)
digest = hashlib.sha256(open(weights, "rb").read()).hexdigest()
assert digest == WEIGHTS_SHA256, f"downloaded weights differ from the file checked locally: {digest}"
print("weights OK:", os.path.getsize(weights) // 2**20, "MB, sha256 matches")

for z in glob.glob("/kaggle/input/**/*.zip", recursive=True):
    zipfile.ZipFile(z).extractall("/tmp/unzipped/" + os.path.basename(z)[:-4])
images = glob.glob("/kaggle/input/**/soccernet_yolo/test/images", recursive=True)
labels = [d for d in glob.glob("/kaggle/input/**/test/labels", recursive=True) + glob.glob("/tmp/unzipped/**/test/labels", recursive=True)
          if "soccernet_yolo" not in d]
assert images and labels, f"inputs not found: images {images}, labels {labels}"
print("images:", images[0], "| labels:", labels[0])

# %%
cmd = [sys.executable, "-u", "/tmp/football/scripts/eval_pitch_keypoints.py", "--weights", weights, "--mapping", "roboflow",
       "--images", images[0], "--labels", labels[0], "--out", "/kaggle/working/pitch_eval", "--stride", "8",
       "--max-minutes", "25"]
code = subprocess.run(cmd).returncode
if code != 0:
    raise SystemExit(f"evaluation stopped with exit code {code}: see the ERROR line above")

# %% [markdown]
# ## Results

# %%
print(json.dumps(json.load(open("/kaggle/working/pitch_eval/metrics.json")), indent=2))

# %%
import cv2
import matplotlib.pyplot as plt
img = cv2.imread("/kaggle/working/pitch_eval/examples.jpg")
plt.figure(figsize=(18, 7)); plt.imshow(img[:, :, ::-1]); plt.axis("off")
plt.title("green circles = true landmarks, red crosses = predicted, yellow = pitch lines from the predicted homography"); plt.show()

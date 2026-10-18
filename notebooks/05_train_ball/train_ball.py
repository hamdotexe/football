# %% [markdown]
# # ⚽ Step 6: BallNet, a ball model that watches 3 frames at once
#
# The detector misses about 1 in 4 balls: the ball is tiny (5–10 px), blurred when kicked, hidden by legs, and easy to
# confuse with boots, heads or the penalty spot. Watching a few frames resolves most of that, because the ball is
# **the small thing that moves like a ball**.
#
# ## Model (TrackNet-style)
# ```
# frames t-1, t, t+1 ──stack──▶ 9-channel image ──U-Net──▶ heatmap ──peak──▶ ball (x, y) or "no ball"
# ```
# - **Input:** 3 consecutive frames stacked, so motion is directly visible to the first layer.
# - **Output:** a heatmap (probability of "ball centre here" for every pixel) instead of a box: better for a 6 px object.
# - **U-Net:** downsample for context (where are the players, where is play going), upsample back to full resolution
#   for pixel precision; skip connections carry fine detail from encoder to decoder.
# - **Focal loss:** ~921,000 pixels per frame are "not ball" and ~1 is "ball". Focal loss down-weights the easy
#   background pixels so the model can't win by predicting "no ball" everywhere.
#
# ## Evaluation
# A prediction counts as correct if it's within **5 px** (at 1280×720) of the true ball centre, less than a ball width.
# The confidence threshold is picked on **validation** matches and applied unchanged to the **test** matches (3 matches
# never used for training). The fine-tuned YOLO detector is scored on the same test frames with the same rule.
#
# ## Fail-fast run
# The training script checks every phase (GPU, downloads, labels, extracted frames, first batch, loss, throughput,
# validation) and **stops with a logged reason** on any failure. It also enforces its own deadline, so it always
# finishes and saves its outputs well before Kaggle's time limit.

# %%
import glob, json, subprocess, sys
import torch
print("CUDA:", torch.cuda.is_available(), "| GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

# %%
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "opencv-python-headless", "huggingface_hub"], check=True)
subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/hamdotexe/football.git", "/tmp/football"], check=True)
print(subprocess.run(["git", "-C", "/tmp/football", "log", "--oneline", "-1"], capture_output=True, text=True).stdout)

# %% [markdown]
# ## Train
# Run 2: continue from run 1's epoch-2 weights (val F1 0.625). Budget: 2.5 h of training inside a 3.5 h deadline
# (download + frame extraction + test evaluation in the rest). If training stops early, the best checkpoint is
# still scored on test before the error is raised.
# The log streams below and is also saved to `ballnet/train.log`.

# %%
detector = (glob.glob("/kaggle/input/**/football_yolo11m_1280.pt", recursive=True) or [None])[0]
init = glob.glob("/kaggle/input/**/ballnet_run1_epoch2.pt", recursive=True)
assert init, "run-1 weights dataset (hamzadotexe/ballnet-run1-weights) is not attached"
print("detector weights for comparison:", detector, "| init:", init[0])
cmd = [sys.executable, "-u", "/tmp/football/scripts/train_ballnet.py", "--out", "/kaggle/working/ballnet",
       "--work", "/tmp/ballnet", "--hours", "2.5", "--deadline-hours", "3.5", "--init", init[0]]
if detector:
    cmd += ["--detector", detector]
code = subprocess.run(cmd).returncode
if code != 0:
    print(open("/kaggle/working/ballnet/train.log").read()[-3000:])
    raise SystemExit(f"training script stopped with exit code {code}: see the ERROR line above")

# %% [markdown]
# ## Results on the unseen test matches

# %%
metrics = json.load(open("/kaggle/working/ballnet/metrics.json"))
print(json.dumps(metrics, indent=2))

# %%
import cv2, csv
import matplotlib.pyplot as plt
rows = list(csv.DictReader(open("/kaggle/working/ballnet/history.csv")))
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
ax[0].plot([float(r["minutes"]) for r in rows], [float(r["train_loss"]) for r in rows]); ax[0].set_title("train loss"); ax[0].set_xlabel("minutes")
ax[1].plot([float(r["minutes"]) for r in rows], [float(r["val_f1"]) for r in rows], label="F1")
ax[1].plot([float(r["minutes"]) for r in rows], [float(r["val_precision"]) for r in rows], label="precision")
ax[1].plot([float(r["minutes"]) for r in rows], [float(r["val_recall"]) for r in rows], label="recall")
ax[1].set_title("validation"); ax[1].legend(); ax[1].set_xlabel("minutes")
plt.savefig("/kaggle/working/ballnet/curves.png", dpi=100, bbox_inches="tight"); plt.show()

# %% [markdown]
# ## Examples (zoomed): green circle = true ball, red cross = BallNet

# %%
img = cv2.imread("/kaggle/working/ballnet/examples.jpg")
plt.figure(figsize=(16, 9)); plt.imshow(img[:, :, ::-1]); plt.axis("off"); plt.show()

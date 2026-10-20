# %% [markdown]
# # ⚽ Step 6b: score BallNet-L on the unseen test matches
#
# BallNet-L (ImageNet-pretrained ResNet-34 U-Net, 24.5M params) trained for 2.5 h on 109 SoccerNet clips and reached
# validation F1 0.83. The training run was then killed while scoring the test split (the machine ran out of RAM), so
# this notebook scores the saved best checkpoint on its own. Nothing is retrained.
#
# Same rules as every earlier score: the 7,331 test frames (every 5th frame of the 49 test clips), the confidence
# threshold chosen on validation (stored in the checkpoint), and a hit = within 5 px of the true ball at 1280x720.

# %%
import glob, json, subprocess, sys
import torch
print("CUDA:", torch.cuda.is_available(), "| GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

# %%
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "opencv-python-headless", "huggingface_hub"], check=True)
subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/hamdotexe/football.git", "/tmp/football"], check=True)
print(subprocess.run(["git", "-C", "/tmp/football", "log", "--oneline", "-1"], capture_output=True, text=True).stdout)

# %%
ckpt = glob.glob("/kaggle/input/**/ballnet_l_run8.pt", recursive=True)
assert ckpt, "BallNet-L weights dataset (hamzadotexe/ballnet-l-weights) is not attached"
cmd = [sys.executable, "-u", "/tmp/football/scripts/train_ballnet.py", "--eval-only", ckpt[0],
       "--out", "/kaggle/working/ballnet", "--work", "/tmp/ballnet", "--deadline-hours", "0.75"]
code = subprocess.run(cmd).returncode
if code != 0:
    print(open("/kaggle/working/ballnet/train.log").read()[-3000:])
    raise SystemExit(f"evaluation stopped with exit code {code}: see the ERROR line above")

# %% [markdown]
# ## Results

# %%
print(json.dumps(json.load(open("/kaggle/working/ballnet/metrics.json")), indent=2))

# %%
import cv2
import matplotlib.pyplot as plt
img = cv2.imread("/kaggle/working/ballnet/examples.jpg")
plt.figure(figsize=(16, 9)); plt.imshow(img[:, :, ::-1]); plt.axis("off"); plt.show()

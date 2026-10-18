#!/usr/bin/env bash
# One-command BallNet training on a rented GPU box (Vast.ai, PyTorch template).
#
#   curl -sL https://raw.githubusercontent.com/hamdotexe/football/main/scripts/vast_run.sh | bash -s -- 2.0
#
# Argument: training time budget in hours (default 2.0). Runs in the background; follow with:
#   tail -f /workspace/ballnet.log
# Results land in /workspace/football/outputs/ballnet/ (weights, metrics.json, history.csv).
set -euo pipefail
HOURS="${1:-2.0}"
cd /workspace 2>/dev/null || cd ~

if [ ! -d football ]; then git clone -q --depth 1 https://github.com/hamdotexe/football.git; fi
cd football && git pull -q

# torch comes with the template; only add what's missing (headless OpenCV: no display libraries on servers)
pip install -q opencv-python-headless huggingface_hub numpy

python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('torch', torch.__version__, '|', torch.cuda.get_device_name(0))"
df -h . | tail -1
nproc

nohup python -u scripts/train_ballnet.py --hours "$HOURS" --data /workspace/soccernet_zips --out outputs/ballnet \
    > /workspace/ballnet.log 2>&1 &
echo "Training started (PID $!). Follow with: tail -f /workspace/ballnet.log"

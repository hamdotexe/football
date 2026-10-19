# ⚽ Football Match Analysis

A multi-model computer vision pipeline that watches a football (soccer) broadcast and produces an annotated video:

- **Players, goalkeepers, referees** detected and tracked with persistent IDs
- **Team assignment** by unsupervised jersey-colour clustering
- **Ball detection and tracking**
- **Top-down minimap** via pitch keypoints and homography
- **Match events**: goals, corners, throw-ins, goal kicks

> 🚧 Work in progress, built step by step.

## Pipeline

```
video → detector (players/GK/refs) ─┐
      → ball detector (multi-frame) ─┼→ tracker → team classifier → pitch homography
                                     │                                   ↓
                                     └──────────────→ game state → event engine → annotated video + minimap
```

## Results so far

### Player / referee / ball detector (YOLO11m, 1280 px, fine-tuned)
Evaluated on the **SoccerNet GSR test split**: 49 clips from 3 matches never used for training or tuning
(7,350 frames, every 5th frame).

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| player | 0.906 | 0.949 | **0.958** | 0.615 |
| referee | 0.921 | 0.791 | **0.887** | 0.547 |
| goalkeeper | 0.911 | 0.609 | 0.749 | 0.477 |
| ball | 0.637 | 0.477 | 0.457 | 0.163 |
| **all** | 0.844 | 0.707 | **0.763** | 0.450 |

**Before vs after** (same 613 unseen test frames; classes grouped so the generic COCO model can be compared):

| Model | ball AP50 | person AP50 | mAP50-95 |
|---|---|---|---|
| COCO-pretrained YOLO11m (baseline) | 0.282 | 0.901 | 0.308 |
| Fine-tuned on SoccerNet + Roboflow | **0.477** (+69%) | **0.963** | **0.396** |

Training: 8,550 SoccerNet frames (57 clips, 3 matches) + 372 Roboflow frames, 19 epochs / 5.8 h on 2× Tesla T4.

Known limitations:
- **Ball** remains hardest (tiny, motion-blurred, often occluded): see the temporal ball model below.
- **Goalkeepers** are confused with players 42% of the time: they're rare (4% of boxes) and the training
  set contains only 3 matches' worth of keeper kits. The pipeline also uses position and tracking to label them.
- Training data covers only 3 matches, so validation mAP plateaus early: more match diversity would help more than more epochs.

### BallNet: 3-frame ball heatmap model (TrackNet-style U-Net, 1.95M params)

Input: 3 consecutive frames stacked (9 channels); output: a per-pixel "ball centre" heatmap trained with focal loss.
A prediction is correct if it lands within **5 px** of the true centre at 1280×720. The confidence threshold is chosen
on validation and applied unchanged to the test split (7,331 frames from 49 clips never used for training).

| Test split (7,331 frames) | precision | recall | F1 | median error |
|---|---|---|---|---|
| Fine-tuned YOLO11m (single frame, 20M params) | **0.799** | 0.637 | **0.709** | 1.12 px |
| BallNet (3 frames, 1.95M params) | 0.710 | **0.651** | 0.679 | 1.16 px |

On validation BallNet scored higher (F1 0.80 vs 0.76), but validation is only 6 clips and was also used to pick the
best epoch and threshold, so its score is optimistic; the 49-clip test split is the number to trust. BallNet finds
slightly more balls (recall) but makes more false detections. Training: ~3.4 h total on 2× Tesla T4 (run 1 + a
warm-started run 2), fp16.

## Project layout

| Folder | Contents |
|---|---|
| `src/footy/` | Pipeline code |
| `notebooks/` | Training notebooks (run on Kaggle GPUs) |
| `configs/` | Settings |
| `scripts/` | Entry points |
| `data/`, `models/`, `outputs/` | Videos, weights, results (not committed) |

## Data

- [SoccerNet](https://www.soccer-net.org/): Game State Reconstruction and Tracking (research, non-commercial use)
- [Roboflow Universe](https://universe.roboflow.com/): football player detection
- Bundesliga sample clips (DFL), used as held-out test footage

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

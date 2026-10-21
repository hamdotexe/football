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

### BallNet: 3-frame ball heatmap models (TrackNet-style U-Nets)

Input: 3 consecutive frames stacked (9 channels); output: a per-pixel "ball centre" heatmap trained with focal loss.
A prediction is correct if it lands within **5 px** of the true centre at 1280×720. The confidence threshold is chosen
on validation and applied unchanged to the test split (7,331 frames from 49 clips never used for training).

| Test split (7,331 frames) | params | precision | recall | F1 | median error |
|---|---|---|---|---|---|
| Fine-tuned YOLO11m (single frame) | 20M | **0.799** | 0.637 | 0.709 | 1.12 px |
| BallNet (from scratch) | 1.95M | 0.710 | 0.651 | 0.679 | 1.16 px |
| **BallNet-L** (ImageNet-pretrained ResNet-34 encoder) | 24.5M | 0.789 | **0.672** | **0.726** | 1.17 px |

- **BallNet** (small, trained from scratch on 57 clips) memorised its training clips: train loss kept falling while
  validation stayed flat, and it lost to YOLO on test.
- **BallNet-L** fixes the two causes: an ImageNet-pretrained encoder (general visual features that transfer to
  unseen stadiums) and ~2× the data (52 labelled clips of SoccerNet's `valid` split that were unused; 6 are kept for
  validation). It finds 236 more balls than YOLO on the test frames (+3.5 recall points) at almost the same precision,
  and catches motion-blurred balls during fast camera pans.
- Fair-comparison caveat: BallNet-L trained on 109 clips, the YOLO detector on 57 (+ Roboflow), so part of the gain
  is data, not architecture.
- Validation scores are optimistic (6 clips, also used to pick the epoch and threshold): BallNet-L 0.83 val → 0.73 test.
- Training: 2.5 h on 2× Tesla T4 (fp16, DataParallel). On T4s, cuDNN crashed with channels-last memory layout under
  DataParallel, so the script probes GPU setups at startup and fell back to the standard layout (59 samples/s).

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

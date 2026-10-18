"""BallNet: a TrackNet-style U-Net that finds the ball from 3 consecutive frames.

Input:  (B, 9, H, W)  frames t-1, t, t+1 stacked on the channel axis, values 0-1
Output: (B, 1, H, W)  heatmap logits for frame t; sigmoid > threshold at the peak = "ball here"
H and W must be divisible by 8 (e.g. 720 x 1280).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
    )


class BallNet(nn.Module):
    def __init__(self, in_frames=3, widths=(32, 64, 128, 256)):
        super().__init__()
        w1, w2, w3, w4 = widths
        self.enc1 = conv_block(3 * in_frames, w1)   # full resolution: fine detail
        self.enc2 = conv_block(w1, w2)              # 1/2
        self.enc3 = conv_block(w2, w3)              # 1/4
        self.bottleneck = conv_block(w3, w4)        # 1/8: wide context (where are players, which way is play going)
        self.dec3 = conv_block(w4 + w3, w3)
        self.dec2 = conv_block(w3 + w2, w2)
        self.dec1 = conv_block(w2 + w1, w1)
        self.head = nn.Conv2d(w1, 1, 1)
        nn.init.constant_(self.head.bias, -4.0)     # start by predicting "almost no ball anywhere" (stable training)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        b = self.bottleneck(F.max_pool2d(e3, 2))
        # decoder: upsample, then concatenate the encoder features of the same resolution ("skip connections")
        d3 = self.dec3(torch.cat([F.interpolate(b, scale_factor=2), e3], 1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, scale_factor=2), e2], 1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, scale_factor=2), e1], 1))
        return self.head(d1)


def gaussian_heatmap(h, w, xy, sigma=2.5):
    """Target heatmap: a Gaussian blob peaking at exactly 1.0 on the (rounded) ball pixel; all zeros if no ball."""
    hm = np.zeros((h, w), np.float32)
    if xy is None:
        return hm
    cx, cy = int(round(xy[0])), int(round(xy[1]))
    if not (0 <= cx < w and 0 <= cy < h):
        return hm
    r = int(3 * sigma)
    ys, xs = np.mgrid[max(0, cy - r):min(h, cy + r + 1), max(0, cx - r):min(w, cx + r + 1)]
    hm[ys, xs] = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
    return hm


def focal_loss(logits, target, alpha=2, beta=4):
    """CenterNet-style focal loss for heatmaps.

    Almost every pixel is "not ball", so plain BCE would learn to predict zero everywhere. This loss
    (1) focuses on pixels the model gets wrong (alpha) and (2) barely punishes predictions right next to the ball (beta).
    """
    p = torch.sigmoid(logits.float()).clamp(1e-6, 1 - 1e-6)
    pos = target.eq(1).float()
    neg = 1 - pos
    pos_loss = -torch.log(p) * (1 - p) ** alpha * pos
    neg_loss = -torch.log(1 - p) * p ** alpha * (1 - target) ** beta * neg
    n_pos = pos.sum().clamp(min=1)
    return (pos_loss.sum() + neg_loss.sum()) / n_pos


@torch.no_grad()
def decode(logits, threshold=0.5):
    """Heatmap logits (B,1,H,W) -> list of (x, y, score) or None, using a 3x3 weighted centroid at the peak."""
    heat = torch.sigmoid(logits.float())[:, 0]
    out = []
    for hm in heat:
        score, idx = hm.flatten().max(0)
        if score < threshold:
            out.append(None)
            continue
        y, x = divmod(int(idx), hm.shape[1])
        y0, y1, x0, x1 = max(0, y - 1), min(hm.shape[0], y + 2), max(0, x - 1), min(hm.shape[1], x + 2)
        patch = hm[y0:y1, x0:x1]
        ys, xs = torch.meshgrid(torch.arange(y0, y1, device=hm.device), torch.arange(x0, x1, device=hm.device), indexing="ij")
        out.append((float((xs * patch).sum() / patch.sum()), float((ys * patch).sum() / patch.sum()), float(score)))
    return out

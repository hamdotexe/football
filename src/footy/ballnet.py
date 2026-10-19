"""BallNet: TrackNet-style U-Nets that find the ball from 3 consecutive frames.

Input:  (B, 9, H, W)  frames t-1, t, t+1 (BGR, as OpenCV reads them) stacked on the channel axis, values 0-1
Output: (B, 1, H, W)  heatmap logits for frame t; sigmoid > threshold at the peak = "ball here"

Two sizes ("arch"):
- "small":    BallNet, 1.95M params, trained from scratch. H and W must be divisible by 8 (e.g. 720 x 1280).
- "resnet34": BallNetL, ~24.5M params, ImageNet-pretrained ResNet-34 encoder. Any H, W (padded internally).
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


def _up(x):
    return F.interpolate(x, scale_factor=2)


class BallNetL(nn.Module):
    """U-Net with an ImageNet-pretrained ResNet-34 encoder.

    The small BallNet learns its features from 3 matches only, so it memorises those stadiums and kits. This encoder
    starts from features learned on 1.2M ImageNet photos (edges, textures, shapes, "what is an object"), which carry
    over to unseen matches much better. It is 12x bigger yet cheaper to run: ResNet downsamples early, so the heavy
    layers work at low resolution, while a light full-resolution branch keeps the pixel-accurate ball position.
    """

    def __init__(self, in_frames=3, pretrained=True):
        super().__init__()
        from torchvision.models import ResNet34_Weights, resnet34
        r = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        conv1 = nn.Conv2d(3 * in_frames, 64, 7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            # ImageNet weights expect one RGB image; we feed 3 BGR frames. The middle frame gets the pretrained filters
            # (so at the start the model sees frame t exactly like the ImageNet model would); the neighbouring frames
            # start at zero and the model learns how to use them (motion) during training.
            conv1.weight.zero_()
            mid = in_frames // 2
            conv1.weight[:, 3 * mid:3 * mid + 3] = r.conv1.weight[:, [2, 1, 0]]
        self.stem = nn.Sequential(conv1, r.bn1, r.relu)                     # 1/2,  64 ch
        self.pool = r.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4
        # layer1: 1/4 64 ch | layer2: 1/8 128 | layer3: 1/16 256 | layer4: 1/32 512 (r.fc, the classifier, is dropped)
        self.full = conv_block(3 * in_frames, 16)                           # full-resolution detail branch
        self.dec4 = conv_block(512 + 256, 256)
        self.dec3 = conv_block(256 + 128, 128)
        self.dec2 = conv_block(128 + 64, 64)
        self.dec1 = conv_block(64 + 64, 32)
        self.dec0 = conv_block(32 + 16, 16)
        self.head = nn.Conv2d(16, 1, 1)
        nn.init.constant_(self.head.bias, -4.0)
        bgr_mean, bgr_std = torch.tensor([0.406, 0.456, 0.485]), torch.tensor([0.225, 0.224, 0.229])  # ImageNet stats
        self.register_buffer("mean", bgr_mean.repeat(in_frames).view(1, -1, 1, 1), persistent=False)
        self.register_buffer("std", bgr_std.repeat(in_frames).view(1, -1, 1, 1), persistent=False)

    def pretrained_parameters(self):
        """Encoder parameters (trained with a lower learning rate so the ImageNet features aren't wrecked early)."""
        for m in (self.stem, self.layer1, self.layer2, self.layer3, self.layer4):
            yield from m.parameters()

    def forward(self, x):
        h, w = x.shape[-2:]
        x = (x - self.mean) / self.std
        x = F.pad(x, (0, -w % 32, 0, -h % 32))      # ResNet needs sizes divisible by 32 (720 -> 736); cropped below
        f0 = self.full(x)
        s1 = self.stem(x)
        s2 = self.layer1(self.pool(s1))
        s3 = self.layer2(s2)
        s4 = self.layer3(s3)
        s5 = self.layer4(s4)
        d = self.dec4(torch.cat([_up(s5), s4], 1))
        d = self.dec3(torch.cat([_up(d), s3], 1))
        d = self.dec2(torch.cat([_up(d), s2], 1))
        d = self.dec1(torch.cat([_up(d), s1], 1))
        d = self.dec0(torch.cat([_up(d), f0], 1))
        return self.head(d)[..., :h, :w]


ARCHS = ("small", "resnet34")


def build_ballnet(arch="small", widths=(32, 64, 128, 256), pretrained=False):
    if arch == "small":
        return BallNet(widths=widths)
    if arch == "resnet34":
        return BallNetL(pretrained=pretrained)
    raise ValueError(f"unknown BallNet arch {arch!r}, expected one of {ARCHS}")


def load_ballnet(path, device="cpu"):
    """Model + checkpoint dict from a training checkpoint (checkpoints from before 'arch' existed are 'small')."""
    ckpt = torch.load(path, map_location=device)
    model = build_ballnet(ckpt.get("arch", "small"), tuple(ckpt.get("widths", (32, 64, 128, 256))))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt


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

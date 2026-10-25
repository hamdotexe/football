"""Top-down pitch drawing (the 'radar' / minimap)."""
import cv2
import numpy as np

from footy.pitch import HL, HW, pitch_polylines

GRASS, LINE = (60, 120, 40), (235, 235, 235)


class Minimap:
    def __init__(self, px_per_m=8, margin_m=4):
        self.s, self.m = px_per_m, margin_m
        self.size = (int((2 * HL + 2 * margin_m) * px_per_m), int((2 * HW + 2 * margin_m) * px_per_m))  # (w, h)
        self.base = np.full((self.size[1], self.size[0], 3), GRASS, np.uint8)
        for line in pitch_polylines():
            cv2.polylines(self.base, [self.to_px(line)], False, LINE, 2, cv2.LINE_AA)

    def to_px(self, xy_m):
        """Pitch metres (origin at centre) -> minimap pixels."""
        xy = (np.asarray(xy_m, dtype=np.float64).reshape(-1, 2) + [HL + self.m, HW + self.m]) * self.s
        return xy.astype(np.int32)

    def draw(self, points_m, colors, radius=7, canvas=None, outline=(0, 0, 0)):
        """Draw dots at pitch positions. `colors` is one BGR tuple or a list (one per point)."""
        img = self.base.copy() if canvas is None else canvas
        if not isinstance(colors, list):
            colors = [colors] * len(points_m)
        for (x, y), c in zip(self.to_px(points_m), colors):
            cv2.circle(img, (int(x), int(y)), radius, tuple(int(v) for v in c), -1, cv2.LINE_AA)
            cv2.circle(img, (int(x), int(y)), radius, outline, 1, cv2.LINE_AA)
        return img

    @staticmethod
    def overlay(frame, img, width_frac=0.24, alpha=0.9, margin=12):
        """Blend the minimap into the top-left corner of a video frame (in place)."""
        w = int(frame.shape[1] * width_frac)
        img = cv2.resize(img, (w, int(img.shape[0] * w / img.shape[1])), interpolation=cv2.INTER_AREA)
        y0, x0 = margin, margin
        roi = frame[y0:y0 + img.shape[0], x0:x0 + w]
        roi[:] = cv2.addWeighted(img, alpha, roi, 1 - alpha, 0)
        return frame

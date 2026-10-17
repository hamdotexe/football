"""Pitch geometry: a metric pitch model, homographies, and keypoints.

Coordinates follow SoccerNet GSR: metres, origin at the centre spot, x along the length (-52.5 .. 52.5),
y across the width (-34 .. 34).
"""
import cv2
import numpy as np

LENGTH, WIDTH = 105.0, 68.0
HL, HW = LENGTH / 2, WIDTH / 2
PENALTY_DEPTH, PENALTY_HALF_WIDTH = 16.5, 20.16
GOAL_AREA_DEPTH, GOAL_AREA_HALF_WIDTH = 5.5, 9.16
CIRCLE_R, PENALTY_SPOT = 9.15, 11.0


def _segments():
    """Straight pitch lines as ((x1, y1), (x2, y2)) in metres."""
    s = [((-HL, -HW), (HL, -HW)), ((-HL, HW), (HL, HW)),       # touchlines
         ((-HL, -HW), (-HL, HW)), ((HL, -HW), (HL, HW)),       # goal lines
         ((0, -HW), (0, HW))]                                  # halfway line
    for sign in (-1, 1):
        gx = sign * HL
        for depth, half in ((PENALTY_DEPTH, PENALTY_HALF_WIDTH), (GOAL_AREA_DEPTH, GOAL_AREA_HALF_WIDTH)):
            bx = gx - sign * depth
            s += [((gx, -half), (bx, -half)), ((bx, -half), (bx, half)), ((bx, half), (gx, half))]
    return s


def _arc(cx, cy, r, a0=0, a1=2 * np.pi, n=64):
    t = np.linspace(a0, a1, n)
    return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], 1)


def pitch_polylines():
    """Every pitch marking as a dense polyline in metres (for drawing / projection)."""
    lines = [np.linspace(a, b, 50) for a, b in _segments()]
    lines.append(_arc(0, 0, CIRCLE_R))
    for sign in (-1, 1):  # penalty arcs: the part of the circle around the spot that lies outside the box
        spot = sign * (HL - PENALTY_SPOT)
        a = np.arccos((PENALTY_DEPTH - PENALTY_SPOT) / CIRCLE_R)
        lines.append(_arc(spot, 0, CIRCLE_R, -a, a) if sign < 0 else _arc(spot, 0, CIRCLE_R, np.pi - a, np.pi + a))
    return lines


# Keypoints we teach the model to find. Name -> pitch coordinate (metres).
def _keypoints():
    k = {}
    for side, sign in (("L", -1), ("R", 1)):
        gx = sign * HL
        pb, ga = gx - sign * PENALTY_DEPTH, gx - sign * GOAL_AREA_DEPTH
        k |= {f"{side}_corner_top": (gx, -HW), f"{side}_corner_bottom": (gx, HW),
              f"{side}_box_goal_top": (gx, -PENALTY_HALF_WIDTH), f"{side}_box_goal_bottom": (gx, PENALTY_HALF_WIDTH),
              f"{side}_box_top": (pb, -PENALTY_HALF_WIDTH), f"{side}_box_bottom": (pb, PENALTY_HALF_WIDTH),
              f"{side}_6yd_goal_top": (gx, -GOAL_AREA_HALF_WIDTH), f"{side}_6yd_goal_bottom": (gx, GOAL_AREA_HALF_WIDTH),
              f"{side}_6yd_top": (ga, -GOAL_AREA_HALF_WIDTH), f"{side}_6yd_bottom": (ga, GOAL_AREA_HALF_WIDTH),
              f"{side}_penalty_spot": (sign * (HL - PENALTY_SPOT), 0.0)}
        a = np.arccos((PENALTY_DEPTH - PENALTY_SPOT) / CIRCLE_R)   # where the penalty arc meets the box line
        k |= {f"{side}_arc_top": (pb, -CIRCLE_R * np.sin(a)), f"{side}_arc_bottom": (pb, CIRCLE_R * np.sin(a))}
    k |= {"mid_top": (0.0, -HW), "mid_bottom": (0.0, HW), "centre_spot": (0.0, 0.0),
          "circle_top": (0.0, -CIRCLE_R), "circle_bottom": (0.0, CIRCLE_R),
          "circle_left": (-CIRCLE_R, 0.0), "circle_right": (CIRCLE_R, 0.0)}
    return k


KEYPOINTS = _keypoints()
KEYPOINT_NAMES = list(KEYPOINTS)
KEYPOINT_XY = np.array(list(KEYPOINTS.values()), dtype=np.float64)


def project(H, pts):
    """Apply homography H to Nx2 points."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def homography_from_players(annotations, min_points=6, ransac_px=6.0):
    """Image->pitch homography from people's feet: bbox_image bottom corners/middle <-> bbox_pitch.

    Returns (H_img_to_pitch, n_inliers, median_reprojection_error_px) or (None, 0, inf).
    """
    img_pts, pitch_pts = [], []
    for a in annotations:
        b, p = a.get("bbox_image"), a.get("bbox_pitch")
        if not b or not p or a.get("category_id") not in (1, 2, 3):
            continue
        y = b["y"] + b["h"]
        for x, key in ((b["x"], "bottom_left"), (b["x"] + b["w"] / 2, "bottom_middle"), (b["x"] + b["w"], "bottom_right")):
            img_pts.append((x, y))
            pitch_pts.append((p[f"x_{key}"], p[f"y_{key}"]))
    if len(img_pts) < min_points:
        return None, 0, np.inf
    img_pts, pitch_pts = np.float64(img_pts), np.float64(pitch_pts)
    H, mask = cv2.findHomography(pitch_pts, img_pts, cv2.RANSAC, ransac_px)   # pitch -> image
    if H is None:
        return None, 0, np.inf
    err = np.linalg.norm(project(H, pitch_pts) - img_pts, axis=1)
    return np.linalg.inv(H), int(mask.sum()), float(np.median(err))


# --- Calibration from SoccerNet line annotations ------------------------------------------------------------------
# SoccerNet line name -> ("x", value) for a line x = value, ("y", value) for y = value, or ("circle", cx, r).
# Goal posts / crossbars are skipped: they are not on the ground plane.
def _line_geometry():
    g = {"Side line top": ("y", -HW), "Side line bottom": ("y", HW), "Side line left": ("x", -HL),
         "Side line right": ("x", HL), "Middle line": ("x", 0.0), "Circle central": ("circle", 0.0, CIRCLE_R)}
    for side, sign in (("left", -1), ("right", 1)):
        g[f"Big rect. {side} main"] = ("x", sign * (HL - PENALTY_DEPTH))
        g[f"Big rect. {side} top"] = ("y", -PENALTY_HALF_WIDTH)
        g[f"Big rect. {side} bottom"] = ("y", PENALTY_HALF_WIDTH)
        g[f"Small rect. {side} main"] = ("x", sign * (HL - GOAL_AREA_DEPTH))
        g[f"Small rect. {side} top"] = ("y", -GOAL_AREA_HALF_WIDTH)
        g[f"Small rect. {side} bottom"] = ("y", GOAL_AREA_HALF_WIDTH)
        g[f"Circle {side}"] = ("circle", sign * (HL - PENALTY_SPOT), CIRCLE_R)
    return g


LINE_GEOMETRY = _line_geometry()
def _pixels_per_metre(H, pitch_pts):
    """Local image scale (px per metre) at each pitch point, used to express pitch-space errors in pixels."""
    return np.linalg.norm(project(H, pitch_pts + [1.0, 0.0]) - project(H, pitch_pts), axis=1)


def _line_residuals(H, observations, circle_scales=None):
    """Pixel distance of each annotated point to its projected template line/circle.

    Straight lines: exact point-to-line distance in the image (lines stay lines under a homography).
    Circles: distance |P - centre| - r on the pitch (P = point mapped back to the pitch), times the local px/m scale.
    """
    res, H_inv, k = [], None, 0
    for geom, pts in observations:
        if geom[0] == "circle":
            H_inv = np.linalg.inv(H) if H_inv is None else H_inv
            P = project(H_inv, pts)
            scale = circle_scales[k] if circle_scales is not None else _pixels_per_metre(H, P)
            k += 1
            res.append(np.abs(np.linalg.norm(P - [geom[1], 0.0], axis=1) - geom[2]) * scale)
        else:
            ends = [(geom[1], -100.0), (geom[1], 100.0)] if geom[0] == "x" else [(-100.0, geom[1]), (100.0, geom[1])]
            a, b = project(H, ends)
            d = b - a
            res.append(np.abs(d[0] * (pts[:, 1] - a[1]) - d[1] * (pts[:, 0] - a[0])) / (np.linalg.norm(d) + 1e-9))
    return np.concatenate(res) if res else np.zeros(0)


def line_observations(pitch_annotation, width, height):
    """[(geometry, Nx2 pixel points)] for every usable annotated line in a frame."""
    obs = []
    for name, pts in pitch_annotation.get("lines", {}).items():
        if name in LINE_GEOMETRY and pts:
            obs.append((LINE_GEOMETRY[name], np.float64([(p["x"] * width, p["y"] * height) for p in pts])))
    return obs


def refine_homography(H_pitch_to_img, observations, player_pairs=None, player_weight=0.3):
    """Least-squares refinement of a pitch->image homography so annotated line points lie on their lines."""
    from scipy.optimize import least_squares

    H0 = H_pitch_to_img / H_pitch_to_img[2, 2]
    H0_inv = np.linalg.inv(H0)
    scales = [_pixels_per_metre(H0, project(H0_inv, pts)) for g, pts in observations if g[0] == "circle"]

    def residuals(h):
        H = np.append(h, 1.0).reshape(3, 3)
        r = _line_residuals(H, observations, scales)
        if player_pairs is not None:
            img_pts, pitch_pts = player_pairs
            r = np.concatenate([r, player_weight * np.linalg.norm(project(H, pitch_pts) - img_pts, axis=1)])
        return r

    fit = least_squares(residuals, H0.ravel()[:8], loss="soft_l1", f_scale=3.0, max_nfev=200)
    H = np.append(fit.x, 1.0).reshape(3, 3)
    line_err = _line_residuals(H, observations)
    return H, line_err

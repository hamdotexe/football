"""Roboflow's pretrained pitch keypoint model (github.com/roboflow/sports, YOLOv8x-pose, 32 keypoints) in our frame.

Their keypoints are numbered differently from ours and their config draws the pitch at 120 x 70 m, so each of their
points is matched to ours by its physical landmark (corner, box corner, circle point, ...), not by number.
Their #11/#12 and #19/#20 sit on the penalty-box line: checked on video frames, they mark where the penalty arc meets
the box line (median 9 px from that, 19 px from their config's 9.16 m position). They have no centre spot.
"""
import numpy as np

from footy.pitch import KEYPOINT_NAMES

WEIGHTS_GDRIVE_ID = "1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf"     # from roboflow/sports examples/soccer/setup.sh
WEIGHTS_SHA256 = "28f68f7c4056d6d9b137efd2e7ab5f3c494039380c63831649126ced25628b36"   # the file downloaded on 2026-10-21

# their keypoint index (0-based) -> our keypoint name
ROBOFLOW_TO_OURS = [
    "L_corner_top", "L_box_goal_top", "L_6yd_goal_top", "L_6yd_goal_bottom", "L_box_goal_bottom", "L_corner_bottom",
    "L_6yd_top", "L_6yd_bottom", "L_penalty_spot", "L_box_top", "L_arc_top", "L_arc_bottom", "L_box_bottom",
    "mid_top", "circle_top", "circle_bottom", "mid_bottom",
    "R_box_top", "R_arc_top", "R_arc_bottom", "R_box_bottom", "R_penalty_spot", "R_6yd_top", "R_6yd_bottom",
    "R_corner_top", "R_box_goal_top", "R_6yd_goal_top", "R_6yd_goal_bottom", "R_box_goal_bottom", "R_corner_bottom",
    "circle_left", "circle_right",
]
OUR_INDEX = np.array([KEYPOINT_NAMES.index(n) for n in ROBOFLOW_TO_OURS])


def to_our_keypoints(xy: np.ndarray, conf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """32 Roboflow keypoints (xy 32x2, conf 32) -> our 33-keypoint layout (missing ones: conf 0)."""
    out_xy = np.zeros((len(KEYPOINT_NAMES), 2))
    out_conf = np.zeros(len(KEYPOINT_NAMES))
    out_xy[OUR_INDEX] = xy
    out_conf[OUR_INDEX] = conf
    return out_xy, out_conf

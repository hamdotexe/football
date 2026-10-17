"""Stack two videos top/bottom with a title bar on each (for before/after comparisons).

Usage: python scripts/stack_videos.py top.mp4 "COCO baseline" bottom.mp4 "Fine-tuned on SoccerNet" out.mp4
"""
import sys

import cv2
import supervision as sv

top, top_title, bottom, bottom_title, target = sys.argv[1:6]
W, H, BAR = 1280, 720, 56


def titled(frame, title):
    frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
    bar = cv2.copyMakeBorder(frame, BAR, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    cv2.putText(bar, title, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    return bar


info = sv.VideoInfo.from_video_path(top)
out_info = sv.VideoInfo(width=W, height=2 * (H + BAR), fps=info.fps)
with sv.VideoSink(target, out_info) as sink:
    for a, b in zip(sv.get_video_frames_generator(top), sv.get_video_frames_generator(bottom)):
        sink.write_frame(cv2.vconcat([titled(a, top_title), titled(b, bottom_title)]))
print("wrote", target)

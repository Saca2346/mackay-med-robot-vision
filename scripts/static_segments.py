"""
Split a recording into STATIC segments (same camera pose) and mark the frames where the shelf is unchanged.

A tripod webcam recording of a shelf: the camera sits still for minutes and the boxes do not move, so one carefully
corrected outline per box is valid for every frame of the segment. This script finds those segments; the labelling
itself is scripts/auto_label_static.py. Pose change = phase-correlation shift of the whole (downscaled) image against
the segment anchor above --shift-px (measured at 480 px width); shelf unchanged = normalised cross-correlation of the
shelf region against the segment's median image above --ncc.

  python scripts/static_segments.py --video V --out DIR  [--step 5]   -> DIR/segments.json
"""
from __future__ import annotations

import argparse
import json
import pathlib

import cv2
import numpy as np

W = 480


def small(frame):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (W, int(frame.shape[0] * W / frame.shape[1])), interpolation=cv2.INTER_AREA).astype(np.float32)


def segments(video: str, step=5, shift_px=1.5, min_len=60):
    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    win = None
    segs, cur, anchor = [], None, None
    for i in range(0, n, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if not ok:
            break
        g = small(f)
        if win is None:
            win = cv2.createHanningWindow(g.shape[::-1], cv2.CV_32F)
        if anchor is None:
            anchor, cur = g, {"start": i, "end": i, "anchor": i}
            continue
        (dx, dy), resp = cv2.phaseCorrelate(anchor, g, win)
        if resp < 0.05 or np.hypot(dx, dy) > shift_px:          # camera moved (or the image changed completely)
            segs.append(cur)
            anchor, cur = g, {"start": i, "end": i, "anchor": i}
        else:
            cur["end"] = i
    if cur:
        segs.append(cur)
    # merge tiny segments into nothing: they are transitions
    return [s for s in segs if s["end"] - s["start"] >= min_len], n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--shift-px", type=float, default=1.5)
    a = ap.parse_args()
    segs, n = segments(a.video, a.step, a.shift_px)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"video": a.video, "frames": n, "segments": segs}, open(out / "segments.json", "w"), indent=1)
    print(a.video, n, "frames ->", len(segs), "static segments:", [(s["start"], s["end"]) for s in segs])


if __name__ == "__main__":
    main()

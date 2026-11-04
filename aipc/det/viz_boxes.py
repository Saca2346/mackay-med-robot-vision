"""Deck figure: outlines on one held-out 28-Sep photo. Usage: python viz_boxes.py PRED_NAME PHOTO_BASENAME OUT.jpg [SNAP_JSON]

green = owner-reviewed truth, red = detector (cached predictions pred/PRED_NAME), blue = after the edge correction
chosen on the dev photos (eval/snap_*.json "chosen_on_dev"). Left: whole photo; right: zoom on both ends and the
middle of the box whose detector outline is furthest from the truth and of a typical box."""
import json
import os
import sys

import cv2
import numpy as np

import tightlib as tl

E = os.path.dirname(os.path.abspath(__file__))
name, base, out = sys.argv[1:4]
snapcfg = json.load(open(sys.argv[4]))["chosen_on_dev"] if len(sys.argv) > 4 else None
path = os.path.join(tl.M, base)
img = cv2.imread(path)
gts = tl.truth(path, img.shape)
prs = list(np.load(os.path.join(E, "pred", name, base + ".npy")))
gx, gy = tl.gradients(img, snapcfg["blur"]) if snapcfg else (None, None)
sn = [tl.snap(q, gx, gy, snapcfg["end"], snapcfg["side"], snapcfg["lam"], side_bias=snapcfg.get("bias", 0.0))
      for q in prs] if snapcfg else []
vis = img.copy()
t = max(2, img.shape[1] // 900)
for g in gts:
    cv2.polylines(vis, [np.rint(g).astype(np.int32)], True, (60, 200, 0), t)
for q in prs:
    cv2.polylines(vis, [np.rint(q).astype(np.int32)], True, (40, 40, 230), t)
for q in sn:
    cv2.polylines(vis, [np.rint(q).astype(np.int32)], True, (230, 120, 20), t)
m = np.array([[tl.iou(g, q) for q in prs] for g in gts])
pairs = tl.greedy(m)
worst = min(pairs, key=lambda p: m[p])
typical = sorted(pairs, key=lambda p: m[p])[len(pairs) // 2]
tiles = []
for gi, pi in (worst, typical):
    c, u, v, L, W = tl.frame_of(gts[gi])
    for f in (-1.0, 0.0, 1.0):
        p = c + f * L * u
        r = int(2.2 * W)
        x0, y0 = int(max(0, p[0] - r)), int(max(0, p[1] - r))
        crop = vis[y0:y0 + 2 * r, x0:x0 + 2 * r]
        crop = cv2.resize(crop, (360, 360), interpolation=cv2.INTER_NEAREST)
        cv2.putText(crop, f"IoU {m[gi, pi]:.2f} {'end' if f else 'middle'}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)
        tiles.append(crop)
grid = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
s = grid.shape[0] / vis.shape[0]
left = cv2.resize(vis, (int(vis.shape[1] * s), grid.shape[0]), interpolation=cv2.INTER_AREA)
cv2.imwrite(out, np.hstack([left, grid]), [cv2.IMWRITE_JPEG_QUALITY, 88])
print(out, "pairs", len(pairs), "worst IoU", round(float(m[worst]), 3), "typical", round(float(m[typical]), 3))

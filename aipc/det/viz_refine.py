"""Deck figure: zooms on held-out 28-Sep boxes. Usage: python viz_refine.py PRED_NAME OUT.jpg [N_BOXES]

green = owner outline, red = detector, blue = the pipeline's snap, magenta = the learned refiner. Each row is one box:
its two ends and its middle. Boxes: spread from bad to typical by the snap's end error (the 5th to the 60th percentile
from the worst; pairs above 15% are left out, there owner and detector outline different extents), chosen by the snap,
never by the refiner's own result."""
import os
import sys

import cv2
import numpy as np

import refine_lib as rl
import tightlib as tl

E = os.path.dirname(os.path.abspath(__file__))
name, out = sys.argv[1:3]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 4
SNAP = dict(blur=1.5, end=0.02, lam=0.6, bias=-0.02)
ref = rl.Refiner(os.path.join(E, "weights", "refiner.pt"))
cands = []
for p in tl.photos("test"):
    img = cv2.imread(p)
    gts = tl.truth(p, img.shape)
    prs = list(np.load(os.path.join(E, "pred", name, os.path.basename(p) + ".npy")))
    if not gts or not prs:
        continue
    gx, gy = tl.gradients(img, SNAP["blur"])
    m = np.array([[tl.iou(g, q) for q in prs] for g in gts])
    for i, j in tl.greedy(m):
        s = tl.snap(prs[j], gx, gy, SNAP["end"], 0.0, SNAP["lam"], side_bias=SNAP["bias"])
        e, _, L, _ = tl.errors(gts[i], s)
        cands.append((max(abs(x) for x in e) / L, p, gts[i], prs[j], s))
cands.sort(key=lambda c: -c[0])
cands = [c for c in cands if c[0] < 0.15]      # > 15% = the owner and detector outlines are not the same box extent
chosen = [cands[int(q * (len(cands) - 1))] for q in np.linspace(0.05, 0.6, N)]   # bad to typical, by the snap's error
rows = []
for err, p, g, q, s in chosen:
    img = cv2.imread(p)
    r = ref(img, q)
    vis = img.copy()
    t = max(2, img.shape[1] // 1500)
    for poly, col in ((g, (60, 200, 0)), (q, (40, 40, 230)), (s, (230, 120, 20)), (r, (220, 40, 220))):
        cv2.polylines(vis, [np.rint(poly).astype(np.int32)], True, col, t, cv2.LINE_AA)
    c, u, v, L, W = tl.frame_of(g)
    tiles = []
    for f in (-1.0, 0.0, 1.0):
        pt = c + f * L * u
        rr = int(1.6 * W)
        x0, y0 = int(max(0, pt[0] - rr)), int(max(0, pt[1] - rr))
        crop = cv2.resize(vis[y0:y0 + 2 * rr, x0:x0 + 2 * rr], (400, 400), interpolation=cv2.INTER_AREA)
        tiles.append(crop)
    ge, gs, GL, GW = tl.errors(g, s)
    re_, rs_, _, _ = tl.errors(g, r)
    lab = f"{os.path.basename(p)[:19]}  snap ends {ge[0] / GL:+.1%} {ge[1] / GL:+.1%}  refiner {re_[0] / GL:+.1%} {re_[1] / GL:+.1%}"
    row = np.hstack(tiles)
    cv2.rectangle(row, (0, 0), (row.shape[1], 34), (20, 20, 20), -1)
    cv2.putText(row, lab, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    rows.append(row)
cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
print(out, len(rows))

"""Deck figure: the PICKed box of a task photo with the refiner's rectangle (blue), tilted ends (magenta) and tilted ends
and sides (orange). Usage: python viz_quad_task.py TIME_JSON PHOTO OUT.jpg  (TIME_JSON from aipc_tasks.py time)."""
import json
import sys

import cv2
import numpy as np

import refine_lib as rl
import tightlib as tl

tj, photo, out = sys.argv[1:4]
d = json.load(open(tj))
run = next(r for r in d["raw"] if r["pick"])
q = np.array(run["polys"][run["pick"][0]], np.float64)
img = cv2.imread(photo)
ref = rl.Refiner(__import__("os").path.join(__import__("os").path.dirname(__file__) or ".", "weights", "refiner.pt"), "cuda")
outs = [(ref(img, q), (230, 120, 20)), (ref.quad(img, q), (220, 40, 220)), (ref.quad(img, q, sides=True), (0, 140, 255))]
c, u, v, L, W = tl.frame_of(q)
tiles = []
for poly, col in outs:
    z = img.copy()
    cv2.polylines(z, [np.rint(poly).astype(np.int32)], True, col, max(1, img.shape[1] // 900), cv2.LINE_AA)
    x0, y0 = np.maximum(np.floor(poly.min(0)).astype(int) - 20, 0)
    x1, y1 = np.minimum(np.ceil(poly.max(0)).astype(int) + 20, [img.shape[1], img.shape[0]])
    t = z[y0:y1, x0:x1]
    f = 900 / t.shape[0]
    tiles.append(cv2.resize(t, (int(t.shape[1] * f), 900), interpolation=cv2.INTER_AREA))
cv2.imwrite(out, np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 90])
print(out)

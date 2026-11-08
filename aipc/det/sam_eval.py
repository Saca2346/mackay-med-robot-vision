"""Outline from Segment Anything (SAM 2.1) prompted by the detector (6 Nov 2026), against the owner outlines.

  python sam_eval.py PRED_NAME [SAM_WEIGHTS]

No training: SAM 2.1 (Meta, via Ultralytics) gets the detector's box as the prompt (the axis-aligned bounds of the
oriented box, 3 points on its centre line, and 2 negative points in the neighbouring boxes beside its long sides), its
mask is turned into the smallest enclosing rectangle (cv2.minAreaRect). A mask that does not fit (rectangle IoU with
the detector outline < 0.6, or the mask fills < 80% of its rectangle) falls back to the refiner (weights/refiner.pt).
The prompt (with or without the negative points, with or without the bounds) is chosen on the dev photos (one in 3,
as there is no training), then the 28-Sep test once. Writes eval/sam_<PRED_NAME>.json.
"""
import json
import os
import sys
import time

import cv2
import numpy as np

import refine_lib as rl
import tightlib as tl

E = os.path.dirname(os.path.abspath(__file__))
name = sys.argv[1]
weights = sys.argv[2] if len(sys.argv) > 2 else "sam2.1_b.pt"
from ultralytics import SAM  # noqa: E402

sam = SAM(weights)
ref = rl.Refiner(os.path.join(E, "weights", "refiner.pt"), "cuda")


def prompt(q, neg, bounds):
    c, u, v, L, W = tl.frame_of(np.asarray(q, np.float64))
    pts = [c + t * L * u for t in (-0.6, 0.0, 0.6)]
    lab = [1, 1, 1]
    if neg:
        pts += [c + 2.0 * W * v, c - 2.0 * W * v]
        lab += [0, 0]
    kw = {"points": [np.array(pts).tolist()], "labels": [lab]}
    if bounds:
        x0, y0 = q.min(0) - 0.03 * (q.max(0) - q.min(0))
        x1, y1 = q.max(0) + 0.03 * (q.max(0) - q.min(0))
        kw["bboxes"] = [[float(x0), float(y0), float(x1), float(y1)]]
    return kw


def sam_outline(img, q, neg, bounds):
    r = sam(img, verbose=False, **prompt(q, neg, bounds))[0]
    if r.masks is None or not len(r.masks.data):
        return None
    m = r.masks.data[0].cpu().numpy().astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.nonzero(lab == k)
    rect = cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32))
    poly = cv2.boxPoints(rect)
    if rect[1][0] * rect[1][1] <= 0 or st[k, cv2.CC_STAT_AREA] / (rect[1][0] * rect[1][1]) < 0.8 or tl.iou(poly, q) < 0.6:
        return None
    return poly


def run(paths, settings):
    acc = {k: ([], 0, 0, 0) for k in settings}
    secs = []
    for p in paths:
        img = cv2.imread(p)
        gts = tl.truth(p, img.shape)
        prs = list(np.load(os.path.join(E, "pred", name, os.path.basename(p) + ".npy")))
        for key in settings:
            outs, fb = [], 0
            for q in prs:
                if key == "refiner":
                    outs.append(ref(img, q))
                    continue
                t = time.perf_counter()
                o = sam_outline(img, np.asarray(q, np.float64), *key)
                secs.append(time.perf_counter() - t)
                if o is None:
                    fb += 1
                    o = ref(img, q)
                outs.append(o)
            rows, ngt, npr, nfb = acc[key]
            acc[key] = (rows + tl.match_rows(gts, outs, os.path.basename(p)), ngt + len(gts), npr + len(prs), nfb + fb)
    out = {}
    for key, (rows, ngt, npr, nfb) in acc.items():
        out[str(key)] = dict(tl.summary(rows, ngt, npr), fallback_to_refiner=nfb)
    return out, (round(float(np.median(secs)), 4) if secs else None)


dev, test = tl.photos("dev")[::3], tl.photos("test")
grid = [(True, True), (True, False), (False, True), (False, False)]
dev_res, _ = run(dev, grid + ["refiner"])
for k, v in dev_res.items():
    print("dev", k, v["tight_1_5"], v["tight_2_10"], v["end_short_1pct"], v["side_over_5pct"], v["fallback_to_refiner"], flush=True)
best = max(grid, key=lambda k: (dev_res[str(k)]["tight_1_5"], dev_res[str(k)]["tight_2_10"]))
test_res, sec = run(test, [best, "refiner"])
for k, v in test_res.items():
    print("test", k, {x: v[x] for x in ("tight_1_5", "tight_2_10", "end_short_1pct", "end_over_1pct", "side_over_5pct",
                                        "side_short_5pct", "iou_median", "fallback_to_refiner")}, flush=True)
json.dump({"pred": name, "sam": weights, "dev_photos": len(dev), "dev": dev_res, "chosen_on_dev": str(best),
           "test": test_res, "sam_s_per_box_median": sec}, open(os.path.join(E, "eval", f"sam_{name}.json"), "w"), indent=1)
print("SAM_DONE chosen", best, "s/box", sec, flush=True)

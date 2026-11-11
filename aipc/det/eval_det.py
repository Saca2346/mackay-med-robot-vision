"""Box outline accuracy on the held-out 28-Sep photo session (87 human-reviewed photos). Usage:
  python eval_det.py WEIGHTS IMGSZ [NAME] [--conf 0.35]          IMGSZ = 1280 or 544x960 etc.

Truth = the owner-reviewed OBB labels, class 0 (box). Predictions = class 0 at conf >= 0.35 (production threshold).
Matching: rotated-polygon IoU, greedy one-to-one, pairs with IoU >= 0.5. For every matched pair the prediction is measured in
the truth box's own frame (pixels of the original 12 MP photo):
  end error  = how far the prediction goes past (+) or stops short of (-) each END of the box (long axis)
  side error = the same for each SIDE (short axis)
"tight" = both ends within 1% of the box length AND both sides within 5% of the box width (also 2% / 10% reported).
Also: recall, precision, photos with the exact box count, median time per photo (GPU, preprocess+inference+postprocess).
Writes eval/<NAME>.json. Nothing here is used for training or checkpoint choice.
"""
import json
import os
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO

E = os.path.dirname(os.path.abspath(__file__))
opts, args, it = {"--conf": "0.35", "--device": "0", "--test": "test_photos.txt"}, [], iter(sys.argv[1:])
for a in it:
    if a in opts:
        opts[a] = next(it)
    else:
        args.append(a)
weights, size = args[0], args[1]
name = args[2] if len(args) > 2 else os.path.basename(os.path.dirname(os.path.dirname(weights))) or "model"
conf, device = float(opts["--conf"]), opts["--device"]
imgsz = [int(v) for v in size.split("x")] if "x" in size else int(size)
photos = open(os.path.join(E, opts["--test"])).read().split()


def truth(path):
    lab = path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
    h, w = cv2.imread(path).shape[:2]
    out = []
    for line in open(lab):
        v = line.split()
        if v and v[0] == "0":
            out.append(np.array(v[1:9], float).reshape(4, 2) * [w, h])
    return out


def iou(a, b):
    a, b = a.astype(np.float32), b.astype(np.float32)
    inter, _ = cv2.intersectConvexConvex(cv2.convexHull(a), cv2.convexHull(b))
    union = cv2.contourArea(cv2.convexHull(a)) + cv2.contourArea(cv2.convexHull(b)) - inter
    return inter / union if union > 0 else 0.0


def greedy(m, thr=0.5):
    """One-to-one pairs, highest IoU first (boxes on a shelf barely overlap, so this equals optimal matching)."""
    pairs, used_g, used_p = [], set(), set()
    for k in np.argsort(-m, axis=None):
        i, j = np.unravel_index(k, m.shape)
        if m[i, j] < thr:
            break
        if i not in used_g and j not in used_p:
            pairs.append((i, j))
            used_g.add(i)
            used_p.add(j)
    return pairs


def frame_of(poly):
    """centre, unit long axis, unit short axis, half length, half width of a 4-point box."""
    c = poly.mean(0)
    e1, e2 = poly[1] - poly[0], poly[2] - poly[1]
    long_, short_ = (e1, e2) if np.linalg.norm(e1) >= np.linalg.norm(e2) else (e2, e1)
    u = long_ / np.linalg.norm(long_)
    v = np.array([-u[1], u[0]])
    du, dv = (poly - c) @ u, (poly - c) @ v
    return c, u, v, (du.max() - du.min()) / 2, (dv.max() - dv.min()) / 2


def errors(gt, pr):
    c, u, v, L, W = frame_of(gt)
    pu, pv = (pr - c) @ u, (pr - c) @ v
    return [pu.max() - L, -pu.min() - L], [pv.max() - W, -pv.min() - W], 2 * L, 2 * W


model = YOLO(weights, task="obb")
model.predict(photos[0], imgsz=imgsz, conf=conf, device=device, verbose=False)          # warm-up
rows, times, exact, n_gt, n_pr = [], [], 0, 0, 0
for p in photos:
    gts = truth(p)
    t0 = time.perf_counter()
    r = model.predict(p, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
    times.append(time.perf_counter() - t0)
    o = r.obb
    prs = [q for q, k in zip(o.xyxyxyxy.cpu().numpy(), o.cls.cpu().numpy()) if int(k) == 0]
    n_gt += len(gts)
    n_pr += len(prs)
    exact += len(gts) == len(prs)
    if not gts or not prs:
        continue
    m = np.array([[iou(g, q) for q in prs] for g in gts])
    for i, j in greedy(m):
            ends, sides, length, width = errors(gts[i], prs[j])
            rows.append({"photo": os.path.basename(p), "iou": float(m[i, j]), "ends": ends, "sides": sides,
                         "length": float(length), "width": float(width)})
ends = np.array([r["ends"] for r in rows])
sides = np.array([r["sides"] for r in rows])
length = np.array([r["length"] for r in rows])[:, None]
width = np.array([r["width"] for r in rows])[:, None]
ious = np.array([r["iou"] for r in rows])


def pct(x):
    return round(100 * float(np.mean(x)), 1)


res = {"name": name, "weights": weights, "imgsz": size, "conf": conf, "device": device, "photos": len(photos), "truth_boxes": n_gt,
       "pred_boxes": n_pr, "matched": len(rows), "recall": round(len(rows) / n_gt, 4),
       "precision": round(len(rows) / max(n_pr, 1), 4), "photos_exact_count": exact,
       "iou_median": round(float(np.median(ious)), 4), "iou_ge_090": pct(ious >= 0.90), "iou_ge_095": pct(ious >= 0.95),
       "end_abs_px_median": round(float(np.median(np.abs(ends))), 1),
       "end_abs_pct_median": round(float(np.median(np.abs(ends) / length)) * 100, 2),
       "end_over_1pct": pct((ends / length > 0.01).any(1)), "end_short_1pct": pct((ends / length < -0.01).any(1)),
       "side_abs_px_median": round(float(np.median(np.abs(sides))), 1),
       "side_over_5pct": pct((sides / width > 0.05).any(1)), "side_short_5pct": pct((sides / width < -0.05).any(1)),
       "tight_1_5": pct((np.abs(ends) / length <= 0.01).all(1) & (np.abs(sides) / width <= 0.05).all(1)),
       "tight_2_10": pct((np.abs(ends) / length <= 0.02).all(1) & (np.abs(sides) / width <= 0.10).all(1)),
       "box_length_px_median": round(float(np.median(length)), 0),
       "box_width_px_median": round(float(np.median(width)), 0),
       "sec_per_photo_median": round(float(np.median(times)), 4)}
os.makedirs(os.path.join(E, "eval"), exist_ok=True)
json.dump({"summary": res, "rows": rows}, open(os.path.join(E, "eval", f"{name}.json"), "w"), indent=1)
print(json.dumps(res), flush=True)

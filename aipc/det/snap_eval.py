"""Edge snap after the detector (4 Nov 2026). Usage: python snap_eval.py WEIGHTS IMGSZ NAME [--device cpu|0]

1. Predictions of WEIGHTS on all 196 reviewed photos are cached (pred/<NAME>/<photo>.npy, class 0, conf >= 0.35).
2. The snap settings are chosen on the DEV photos (23 Sep + 1 Oct) only, by the share of tight boxes (1% / 5%).
3. The chosen settings are applied once to the TEST photos (28 Sep) and reported next to the unsnapped outlines.
Note: models trained on all 196 photos (model B) have seen the dev AND the test photos; their numbers are optimistic.
Writes eval/snap_<NAME>.json.
"""
import itertools
import json
import os
import sys

import cv2
import numpy as np

import tightlib as tl

E = os.path.dirname(os.path.abspath(__file__))
weights, size, name = sys.argv[1:4]
device = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "0"
imgsz = [int(v) for v in size.split("x")] if "x" in size else int(size)
P = os.path.join(E, "pred", name)
os.makedirs(P, exist_ok=True)
# round 3 (side bias): the end search chosen in round 2 on dev (blur 1.5, end 0.02, lam 0.6, no side search), plus a
# constant side correction; rounds 1-2 are in eval/snap_*_v1.json / _v2.json
GRID = [dict(blur=1.5, end=e, side=0.0, lam=0.6, bias=b) for e, b in itertools.product(
    [0.0, 0.02], [0.0, -0.005, -0.01, -0.015, -0.02, -0.03, -0.04])]


def preds(paths):
    todo = [p for p in paths if not os.path.exists(os.path.join(P, os.path.basename(p) + ".npy"))]
    if todo:
        from ultralytics import YOLO
        model = YOLO(weights, task="obb")
        for p in todo:
            r = model.predict(p, imgsz=imgsz, conf=0.35, device=device, verbose=False)[0].obb
            q = r.xyxyxyxy.cpu().numpy()[r.cls.cpu().numpy().astype(int) == 0]
            np.save(os.path.join(P, os.path.basename(p) + ".npy"), q)
    return {p: list(np.load(os.path.join(P, os.path.basename(p) + ".npy"))) for p in paths}


def run(paths, pr, settings):
    """settings: list of snap settings (None = no snap) -> one summary per setting; each photo read once."""
    acc = [([], 0, 0) for _ in settings]
    for p in paths:
        img = cv2.imread(p)
        gts = tl.truth(p, img.shape)
        g = {b: tl.gradients(img, b) for b in {s["blur"] for s in settings if s}}
        for k, s in enumerate(settings):
            prs = pr[p] if s is None else [tl.snap(q, *g[s["blur"]], s["end"], s["side"], s["lam"], side_bias=s.get("bias", 0.0)) for q in pr[p]]
            rows, ngt, npr = acc[k]
            acc[k] = (rows + tl.match_rows(gts, prs, os.path.basename(p)), ngt + len(gts), npr + len(prs))
    return [tl.summary(*a) for a in acc]


dev, test = tl.photos("dev"), tl.photos("test")
pr = preds(dev + test)
dev_res = run(dev, pr, [None] + GRID)
for s, r in zip(GRID, dev_res[1:]):
    print("dev", s, r["tight_1_5"], r["tight_2_10"], r["iou_median"], flush=True)
k = int(np.argmax([r["tight_1_5"] for r in dev_res[1:]]))
none_test, snap_test = run(test, pr, [None, GRID[k]])
results = {"none_dev": dev_res[0], "chosen_on_dev": GRID[k], "snap_dev": dev_res[1 + k], "none_test": none_test,
           "snap_test": snap_test, "grid_dev": [dict(s, **r) for s, r in zip(GRID, dev_res[1:])]}
os.makedirs(os.path.join(E, "eval"), exist_ok=True)
json.dump(results, open(os.path.join(E, "eval", f"snap_{name}.json"), "w"), indent=1)
print(json.dumps({k: results[k] for k in ("none_dev", "chosen_on_dev", "snap_dev", "none_test", "snap_test")}), flush=True)

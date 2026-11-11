"""The edge refiner on the 28-Sep TEST photos, once: detector outlines as they are, after the snap of the pipeline
(--snap 1), and after the refiner, for each detector named.

  python eval_refine.py PRED_NAME [PRED_NAME ...]     cached outlines in pred/<PRED_NAME>/ (snap_eval.py)

Model B has seen these photos in its own training (its outlines are better here than on new photos); the refiner has
not (it is trained on the dev photos only). yolo26s-obb_1280_hold has seen neither. Writes eval/refine_<PRED_NAME>.json.
"""
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

import refine_lib as rl
import tightlib as tl

E = os.path.dirname(os.path.abspath(__file__))
SNAP = dict(blur=1.5, end=0.02, lam=0.6, bias=-0.02)        # the pipeline's snap (aipc_tasks.SNAP)
ref = rl.Refiner(os.path.join(E, "weights", "refiner.pt"), "cuda" if torch.cuda.is_available() else "cpu")
test = tl.photos("test")
for name in sys.argv[1:]:
    acc = {k: ([], 0, 0) for k in ("none", "snap", "refine")}
    secs = []
    for p in test:
        img = cv2.imread(p)
        gts = tl.truth(p, img.shape)
        prs = list(np.load(os.path.join(E, "pred", name, os.path.basename(p) + ".npy")))
        gx, gy = tl.gradients(img, SNAP["blur"])
        t = time.perf_counter()
        refined = [ref(img, q) for q in prs]
        secs.append((time.perf_counter() - t) / max(len(prs), 1))
        outs = {"none": prs, "snap": [tl.snap(q, gx, gy, SNAP["end"], 0.0, SNAP["lam"], side_bias=SNAP["bias"]) for q in prs],
                "refine": refined}
        for k, q in outs.items():
            rows, ngt, npr = acc[k]
            acc[k] = (rows + tl.match_rows(gts, q, os.path.basename(p)), ngt + len(gts), npr + len(prs))
    res = {k: tl.summary(*a) for k, a in acc.items()}
    res["refine_s_per_box_median"] = round(float(np.median(secs)), 4)
    res["rows_refine"] = acc["refine"][0]
    json.dump(res, open(os.path.join(E, "eval", f"refine_{name}.json"), "w"), indent=1)
    keys = ("iou_median", "end_abs_px_median", "end_over_1pct", "end_short_1pct", "side_abs_px_median", "side_over_5pct",
            "side_short_5pct", "tight_1_5", "tight_2_10")
    for k in ("none", "snap", "refine"):
        print(name, k, {x: res[k][x] for x in keys}, flush=True)
    print(name, "refine s/box", res["refine_s_per_box_median"], flush=True)

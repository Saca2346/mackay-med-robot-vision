"""Tilted edges (refine_lib.Refiner.quad) against the refiner's rectangle (6 Nov 2026).

  python quad_eval.py PRED_NAME

Chosen on the dev photos with the cross-validation nets of cv_refine.py (each dev photo refined by the fold net that
never trained on it; the set-up stored with the final weights), then the 28-Sep test once with the final weights.
Set-ups: rect (pipeline now), quad ends (each end on its two halves), quad ends + sides (sides by the side net on
their halves). The owner outlines are rectangles, so a tilted edge is scored against the owner's rectangle (the
per-edge error uses the corner positions). Writes eval/quad_<PRED_NAME>.json.
"""
import json
import os
import sys

import cv2
import numpy as np
import torch

import refine_lib as rl
import tightlib as tl

E = os.path.dirname(os.path.abspath(__file__))
name = sys.argv[1]
dev = tl.photos("dev")
folds = [[p for k, p in enumerate(dev) if k % 5 == f] for f in range(5)]
final = torch.load(os.path.join(E, "weights", "refiner.pt"), map_location="cpu", weights_only=False)
SETUPS = {"rect": lambda r, img, q: r(img, q), "quad_ends": lambda r, img, q: r.quad(img, q),
          "quad_ends_sides": lambda r, img, q: r.quad(img, q, sides=True)}


def fold_refiner(f):
    ck = torch.load(os.path.join(E, "weights", "cv", f"refiner_{final['source']}_f{f}.pt"), map_location="cpu",
                    weights_only=False)
    ck.update({k: final[k] for k in ("iters", "side_bias", "end_bias")})
    path = os.path.join(E, "weights", "cv", f"_quad_f{f}.pt")
    torch.save(ck, path)
    return rl.Refiner(path, "cuda")


def score(pairs):
    acc = {k: ([], 0, 0) for k in SETUPS}
    for ref, paths in pairs:
        for p in paths:
            img = cv2.imread(p)
            gts = tl.truth(p, img.shape)
            prs = list(np.load(os.path.join(E, "pred", name, os.path.basename(p) + ".npy")))
            for k, fn in SETUPS.items():
                rows, ngt, npr = acc[k]
                acc[k] = (rows + tl.match_rows(gts, [fn(ref, img, q) for q in prs], os.path.basename(p)),
                          ngt + len(gts), npr + len(prs))
    return {k: tl.summary(*a) for k, a in acc.items()}


keys = ("tight_1_5", "tight_2_10", "end_short_1pct", "end_over_1pct", "side_over_5pct", "side_short_5pct", "iou_median")
dev_res = score([(fold_refiner(f), folds[f]) for f in range(5)])
for k, v in dev_res.items():
    print("dev", k, {x: v[x] for x in keys}, flush=True)
best = max(SETUPS, key=lambda k: (dev_res[k]["tight_1_5"], dev_res[k]["tight_2_10"]))
test_res = score([(rl.Refiner(os.path.join(E, "weights", "refiner.pt"), "cuda"), tl.photos("test"))])
for k, v in test_res.items():
    print("test", k, {x: v[x] for x in keys}, flush=True)
json.dump({"pred": name, "dev": dev_res, "chosen_on_dev": best, "test": test_res},
          open(os.path.join(E, "eval", f"quad_{name}.json"), "w"), indent=1)
print("QUAD_DONE chosen", best, flush=True)

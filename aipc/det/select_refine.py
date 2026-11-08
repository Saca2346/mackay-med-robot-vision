"""Choose the refiner set-up on the dev VALIDATION photos only (the 28-Sep test photos are not read), then save it.

  python select_refine.py PRED_NAME TAG [TAG ...]

Candidates: each trained net (weights/refiner_<TAG>.pt) x 1 or 2 passes x sides by the side net or by a constant
(each side moved by b x the box width from the detector's, b in 0 ... -5%). Chosen by the share of tight boxes (ends
<= 1%, sides <= 5%), the same criterion as the snap. Writes weights/refiner.pt (the chosen one) and
eval/refine_select.json; eval_refine.py then tests it once.
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
name, tags = sys.argv[1], sys.argv[2:]
dev_ = "cuda" if torch.cuda.is_available() else "cpu"
val = tl.photos("dev")[::7]                       # the same validation photos as train_refiner.py
SNAP = dict(blur=1.5, end=0.02, lam=0.6, bias=-0.02)
SIDES = [None, 0.0, -0.01, -0.02, -0.03, -0.04, -0.05]
data = []
for p in val:
    img = cv2.imread(p)
    data.append((p, img, tl.truth(p, img.shape), list(np.load(os.path.join(E, "pred", name, os.path.basename(p) + ".npy")))))


def score(fn):
    rows, ngt, npr = [], 0, 0
    for p, img, gts, prs in data:
        rows += tl.match_rows(gts, [fn(img, q) for q in prs], os.path.basename(p))
        ngt, npr = ngt + len(gts), npr + len(prs)
    return tl.summary(rows, ngt, npr)


def snap(img, q):
    gx, gy = tl.gradients(img, SNAP["blur"])
    return tl.snap(q, gx, gy, SNAP["end"], 0.0, SNAP["lam"], side_bias=SNAP["bias"])


res = {"none": score(lambda img, q: q), "snap": score(snap)}
print("val none", res["none"]["tight_1_5"], "snap", res["snap"]["tight_1_5"], flush=True)
cands = []
for tag in tags:
    ref = rl.Refiner(os.path.join(E, "weights", f"refiner_{tag}.pt"), dev_)
    for iters in (1, 2):
        for b in SIDES:
            s = score(lambda img, q: ref(img, q, iters, b))
            cands.append({"tag": tag, "iters": iters, "side_bias": b, **s})
            print("val", tag, iters, b, s["tight_1_5"], s["tight_2_10"], s["end_short_1pct"], s["side_over_5pct"], flush=True)
for c in cands:
    c["end_bias"] = 0.0
# round 2 (6 Nov): the ends of round 1 stay short more often than past (val 30% vs 20%), so a constant end correction
# on top of the 6 best set-ups of round 1, again chosen on the validation photos
for c0 in sorted(cands, key=lambda c: -c["tight_1_5"])[:6]:
    ref = rl.Refiner(os.path.join(E, "weights", f"refiner_{c0['tag']}.pt"), dev_)
    for eb in (0.0025, 0.005, 0.0075):
        s = score(lambda img, q: ref(img, q, c0["iters"], c0["side_bias"], eb))
        cands.append({"tag": c0["tag"], "iters": c0["iters"], "side_bias": c0["side_bias"], "end_bias": eb, **s})
        print("val", c0["tag"], c0["iters"], c0["side_bias"], "end", eb, s["tight_1_5"], s["tight_2_10"],
              s["end_short_1pct"], s["end_over_1pct"], flush=True)
best = max(cands, key=lambda c: (c["tight_1_5"], c["tight_2_10"]))
ck = torch.load(os.path.join(E, "weights", f"refiner_{best['tag']}.pt"), map_location="cpu", weights_only=False)
ck.update(iters=best["iters"], side_bias=best["side_bias"], end_bias=best["end_bias"], tag=best["tag"])
torch.save(ck, os.path.join(E, "weights", "refiner.pt"))
json.dump({"pred": name, "val_photos": len(val), "baseline": res, "candidates": cands,
           "chosen": {k: best[k] for k in ("tag", "iters", "side_bias", "end_bias")}},
          open(os.path.join(E, "eval", "refine_select.json"), "w"), indent=1)
print("CHOSEN", {k: best[k] for k in ("tag", "iters", "side_bias", "end_bias", "tight_1_5", "tight_2_10")}, flush=True)

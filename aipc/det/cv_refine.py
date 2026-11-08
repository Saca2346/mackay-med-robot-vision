"""Choose the refiner set-up by 5-fold cross-validation on all 109 DEV photos, then train the final net on all of them.
The 28-Sep test photos are not read.

  python cv_refine.py PRED_NAME [EPOCHS]

Why (6 Nov): choosing on the 16 validation photos of train_refiner.py (about 150 boxes) was noise: round 1 chose
v1 / 2 passes / sides -3% (test: 20.2% tight), round 2 with more options chose v2 / side net / ends +0.25% (val 22.8%,
test 12.4%). Here every dev photo is a validation photo once (about 1,100 boxes). Folds by photo (index mod 5); each
fold's nets are trained on the other 4 folds for EPOCHS epochs, last epoch kept (no choice on the held fold).
Set-ups: SOURCE (truth / pred jitter) x passes (1, 2) x sides (side net, or constant b) x ends (constant eb). Chosen by
the pooled share of tight boxes, ties by the 2% / 10% share. Writes eval/refine_cv.json, weights/refiner.pt.
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
import train_refiner as trf

E = os.path.dirname(os.path.abspath(__file__))
name = sys.argv[1]
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 30
dev_ = "cuda" if torch.cuda.is_available() else "cpu"
dev = tl.photos("dev")
folds = [[p for k, p in enumerate(dev) if k % 5 == f] for f in range(5)]
SIDES = [None, 0.0, -0.01, -0.02, -0.03, -0.04, -0.05]
ENDS = [0.0, 0.0025, 0.005, 0.0075]
SNAP = dict(blur=1.5, end=0.02, lam=0.6, bias=-0.02)
os.makedirs(os.path.join(E, "weights", "cv"), exist_ok=True)
t0 = time.time()


def setups(net_out, const_out):
    """net_out / const_out: {iters: refined poly} -> {(iters, side, end): poly}."""
    out = {}
    for it in (1, 2):
        for b in SIDES:
            if b is None:
                p = net_out[it]
            else:
                d = b * 2 * rl.frame(const_out[it])[4]
                p = rl.rect_from(const_out[it], [0.0, 0.0, d, d])
            for eb in ENDS:
                out[(it, b, eb)] = p if not eb else rl.rect_from(p, [eb * 2 * rl.frame(p)[3]] * 2 + [0.0, 0.0])
    return out


acc = {}                          # (source, iters, side, end) -> [rows, n_gt, n_pred]
base = {"none": [[], 0, 0], "snap": [[], 0, 0]}
for source in ("truth", "pred"):
    per_photo = {p: trf.build_photo(p, name, source, trf.JITTER) for p in dev}
    print(source, "strips built", f"{time.time() - t0:.0f} s", flush=True)
    for f, held in enumerate(folds):
        data = trf.flat(trf.stack([per_photo[p] for p in dev if p not in held]))
        es, ss, _, _ = trf.train_pair(data, dev_, EPOCHS)
        path = os.path.join(E, "weights", "cv", f"refiner_{source}_f{f}.pt")
        torch.save({"end": es, "side": ss, "iters": 1, "source": source}, path)
        ref = rl.Refiner(path, dev_)
        for p in held:
            img = cv2.imread(p)
            gts = tl.truth(p, img.shape)
            prs = list(np.load(os.path.join(E, "pred", name, os.path.basename(p) + ".npy")))
            outs = [setups({it: ref(img, q, it, None, 0.0) for it in (1, 2)},
                           {it: ref(img, q, it, 0.0, 0.0) for it in (1, 2)}) for q in prs]
            for key in [(source, it, b, eb) for it in (1, 2) for b in SIDES for eb in ENDS]:
                a = acc.setdefault(key, [[], 0, 0])
                if outs:
                    a[0] += tl.match_rows(gts, [o[key[1:]] for o in outs], os.path.basename(p))
                a[1] += len(gts)
                a[2] += len(prs)
            if source == "truth":                   # baselines once
                gx, gy = tl.gradients(img, SNAP["blur"])
                for k, q in (("none", prs), ("snap", [tl.snap(x, gx, gy, SNAP["end"], 0.0, SNAP["lam"],
                                                              side_bias=SNAP["bias"]) for x in prs])):
                    base[k][0] += tl.match_rows(gts, q, os.path.basename(p))
                    base[k][1] += len(gts)
                    base[k][2] += len(prs)
        print(source, "fold", f, "done", f"{time.time() - t0:.0f} s", flush=True)
    del per_photo

res = {k: tl.summary(*v) for k, v in base.items()}
print("cv none", res["none"]["tight_1_5"], "snap", res["snap"]["tight_1_5"], flush=True)
cands = []
for key, (rows, ngt, npr) in acc.items():
    s = tl.summary(rows, ngt, npr)
    cands.append({"source": key[0], "iters": key[1], "side_bias": key[2], "end_bias": key[3], **s})
cands.sort(key=lambda c: (-c["tight_1_5"], -c["tight_2_10"]))
for c in cands[:12]:
    print("cv", c["source"], c["iters"], c["side_bias"], c["end_bias"], c["tight_1_5"], c["tight_2_10"],
          c["end_short_1pct"], c["end_over_1pct"], c["side_over_5pct"], c["side_short_5pct"], flush=True)
best = cands[0]

# final net: the chosen source, trained on all dev photos, same epochs, last epoch
data = trf.flat(trf.stack([trf.build_photo(p, name, best["source"], trf.JITTER) for p in dev]))
es, ss, _, _ = trf.train_pair(data, dev_, EPOCHS)
torch.save({"end": es, "side": ss, "iters": best["iters"], "side_bias": best["side_bias"], "end_bias": best["end_bias"],
            "source": best["source"], "tag": "cv"}, os.path.join(E, "weights", "refiner.pt"))
json.dump({"pred": name, "epochs": EPOCHS, "dev_photos": len(dev), "baseline": res, "candidates": cands,
           "chosen": {k: best[k] for k in ("source", "iters", "side_bias", "end_bias")}, "seconds": round(time.time() - t0)},
          open(os.path.join(E, "eval", "refine_cv.json"), "w"), indent=1)
print("CHOSEN", {k: best[k] for k in ("source", "iters", "side_bias", "end_bias", "tight_1_5", "tight_2_10")},
      f"{time.time() - t0:.0f} s", flush=True)

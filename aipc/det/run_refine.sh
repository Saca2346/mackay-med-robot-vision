#!/bin/bash
# Edge refiner (6 Nov 2026): cache the held-out YOLO26s outlines, train on the dev photos (v1: inputs jittered around
# the owner outline; v2: around the detector outline), choose the set-up on the dev validation photos, test once on 28 Sep.
cd "$(dirname "$0")"
P=~/trse_venv/bin/python
$P - <<'EOF'
import os, numpy as np, tightlib as tl
from ultralytics import YOLO
d = "pred/s1280_hold"; os.makedirs(d, exist_ok=True)
m = YOLO("runs/yolo26s-obb_1280_hold/weights/last.pt", task="obb")
for p in tl.photos("test"):
    f = os.path.join(d, os.path.basename(p) + ".npy")
    if not os.path.exists(f):
        r = m.predict(p, imgsz=1280, conf=0.35, device="0", verbose=False)[0].obb
        np.save(f, r.xyxyxyxy.cpu().numpy()[r.cls.cpu().numpy().astype(int) == 0])
print("cached", len(os.listdir(d)))
EOF
[ -f weights/refiner_v1.pt ] || $P train_refiner.py modelB_544x960_SEEN 40 truth v1 2>&1 | grep -v "^ep "
[ -f weights/refiner_v2.pt ] || $P train_refiner.py modelB_544x960_SEEN 40 pred v2 2>&1 | grep -v "^ep "
$P select_refine.py modelB_544x960_SEEN v1 v2
$P eval_refine.py modelB_544x960_SEEN modelB_1280_SEEN s1280_hold
echo REFINE_DONE

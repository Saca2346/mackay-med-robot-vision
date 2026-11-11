"""10 Nov 2026: every box the FINAL detector finds on Leo's Drive photos, drawn thin and numbered (no OCR, nothing decided).
Same detector and grouping as aipc_tasks.py with --views B:1280:0+B:544x960:0 (MultiDetector, group_labels, drop_wide),
so the count per photo is the run's box count.

  PYTHONPATH=../pylib ~/trse_venv/bin/python det_sheet_mv.py OUT_DIR

Writes OUT_DIR/<photo>.jpg (long side 1600 px) and OUT_DIR/dets.json (box count per photo).
"""
import glob
import json
import os
import sys

import cv2
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(os.path.dirname(H), "code")
sys.path.insert(0, C)
sys.path.insert(0, H)
os.chdir(C)
from scripts.test_webcam_box import drop_wide, group_labels  # noqa: E402
from src import config  # noqa: E402
from multiview import MultiDetector  # noqa: E402

out_dir = os.path.abspath(os.path.join(H, sys.argv[1]))
os.makedirs(out_dir, exist_ok=True)
cfg = config.load_config(config.resolve("config/box_pipeline_config_webcam_eleven.yaml"))
m = cfg["model"]
weights = os.path.join(os.path.dirname(H), "..", "models", "box_y26_maxB.pt")
det = MultiDetector("B:1280:0+B:544x960:0", weights, m["task"], m["conf"], m["iou"], cfg["classes"])
photos = sorted(glob.glob(os.path.join(C, "results/drive_photos_20261029/src_photos/main", "*.jpg")))
res = {}
for p in photos:
    img = cv2.imread(p)
    dets = det.detect(img)[0]
    groups, _ = group_labels(dets, cfg["grouping"]["min_label_overlap"])
    groups = drop_wide(groups)
    vis = img.copy()
    tk = max(3, img.shape[1] // 450)
    for i, g in enumerate(groups):
        poly = np.rint(np.asarray(g.box.poly, np.float64)).astype(np.int32)
        cv2.polylines(vis, [poly], True, (255, 210, 0), tk, cv2.LINE_AA)
        cx, cy = poly.mean(0)
        cv2.putText(vis, str(i + 1), (int(cx) - 12 * tk, int(cy) + 6 * tk), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * tk,
                    (0, 0, 0), 4 * tk)
        cv2.putText(vis, str(i + 1), (int(cx) - 12 * tk, int(cy) + 6 * tk), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * tk,
                    (255, 255, 255), 2 * tk)
    s = 1600 / max(img.shape[:2])
    vis = cv2.resize(vis, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else vis
    name = os.path.basename(p)
    cv2.imwrite(os.path.join(out_dir, name), vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
    res[name] = {"boxes": len(groups), "size": list(img.shape[:2]),
                 "polys": [np.round(np.asarray(g.box.poly, np.float64), 1).tolist() for g in groups]}
    print(name, len(groups), flush=True)
json.dump({"weights": weights, "views": "B:1280:0+B:544x960:0", "photos": res},
          open(os.path.join(out_dir, "dets.json"), "w"), indent=1)
print("total", sum(r["boxes"] for r in res.values()))

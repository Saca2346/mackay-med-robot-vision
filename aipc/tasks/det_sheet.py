"""Diagnosis (7 Nov 2026): every box the detector finds on each photo, drawn thin and numbered, to see by eye which boxes
are missed (for example tilted boxes). Nothing is decided here; no OCR.

  python det_sheet.py OUT_DIR PHOTO... [--det W] [--imgsz 544x960|1280] [--rot 0]

Writes OUT_DIR/<photo>.jpg (long side 1600 px) and OUT_DIR/dets.json (box count, confidences, angle of each box).
"""
import json
import math
import os
import sys

import cv2
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(os.path.dirname(H), "code")
sys.path.insert(0, C)
os.chdir(C)
from src import config  # noqa: E402
from src.box_detector import BoxDetector  # noqa: E402

opts, args, it = {"--det": "", "--imgsz": "544x960"}, [], iter(sys.argv[1:])
for a in it:
    if a in opts:
        opts[a] = next(it)
    else:
        args.append(a)
out_dir, photos = args[0], args[1:]
os.makedirs(out_dir, exist_ok=True)
cfg = config.load_config(config.resolve("config/box_pipeline_config_webcam_eleven.yaml"))
m = cfg["model"]
weights = opts["--det"] or os.path.join(os.path.dirname(H), "..", "models", "box_y26_maxB.pt")
imgsz = [int(v) for v in opts["--imgsz"].split("x")] if "x" in opts["--imgsz"] else int(opts["--imgsz"])
det = BoxDetector(weights, m["task"], imgsz, "0", m["conf"], m["iou"], cfg["classes"], fixed_input=isinstance(imgsz, list))
res = {}
for p in photos:
    img = cv2.imread(p)
    dets, _ = det.detect(img)
    boxes = [d for d in dets if d.cls == "box"]
    vis = img.copy()
    tk = max(2, img.shape[1] // 600)
    info = []
    for i, d in enumerate(boxes):
        poly = np.asarray(d.poly, dtype=np.float32)
        (cx, cy), (w, h), a = cv2.minAreaRect(poly)
        long_angle = a if w >= h else a + 90          # angle of the long side against the image x axis
        tilt = abs(((long_angle + 90) % 180) - 90)    # 0 = long side horizontal, 90 = vertical
        info.append({"conf": round(d.conf, 3), "tilt_deg": round(tilt, 1), "centre": [round(cx), round(cy)]})
        cv2.polylines(vis, [np.rint(poly).astype(np.int32)], True, (255, 200, 0), tk, cv2.LINE_AA)
        cv2.putText(vis, str(i), (int(cx) - 10 * tk, int(cy) + 5 * tk), cv2.FONT_HERSHEY_SIMPLEX, 0.9 * tk, (0, 0, 255),
                    2 * tk)
    for d in dets:
        if d.cls == "size_label":
            cv2.polylines(vis, [np.rint(np.asarray(d.poly)).astype(np.int32)], True, (200, 0, 200), max(1, tk // 2))
    s = 1600 / max(img.shape[:2])
    vis = cv2.resize(vis, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else vis
    name = os.path.basename(p)
    cv2.putText(vis, f"{name}  boxes {len(boxes)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6)
    cv2.putText(vis, f"{name}  boxes {len(boxes)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(out_dir, name), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
    res[name] = {"boxes": len(boxes), "size": list(img.shape[:2]), "dets": info}
    print(name, len(boxes), flush=True)
json.dump({"weights": weights, "imgsz": opts["--imgsz"], "photos": res}, open(os.path.join(out_dir, "dets.json"), "w"),
          indent=1)

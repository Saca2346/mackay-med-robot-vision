"""Diagnosis (8 Nov 2026): which boxes of a task photo get another view's outline (--alt-views), and the reads of the
asked box from each outline. Nothing decided here.

  python dbg_alt.py SET KEY VIEWS        e.g. dbg_alt.py t1_drive t1_drive:3 B:544x960:0+B:1280:0
"""
import json
import os
import sys

H = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(os.path.dirname(H), "code")
sys.path[:0] = [C, H]
os.chdir(C)
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime  # noqa: E402

onnxruntime.preload_dlls()
from scripts.test_webcam_box import drop_wide, group_labels  # noqa: E402
from src import config  # noqa: E402
import multiview as mv  # noqa: E402

name, key, views = sys.argv[1:4]
cfg = config.load_config(config.resolve("config/box_pipeline_config_webcam_eleven.yaml"))
_m = cfg["model"]
weights = os.path.join(os.path.dirname(H), "..", "models", "box_y26_maxB.pt")
T = os.path.join(C, "results/tasks_final_20261102")
DRIVE = os.path.join(C, "results/drive_photos_20261029/src_photos/main")
TASK = os.path.join(C, "results/task_20261015/input")
PHOTO = {"t1_orig": f"{TASK}/task1_braun.jpg", "t1_drive": f"{DRIVE}/DCB_BBraun_SeQuentPleaseNEO.jpg",
         "t2_drive": f"{DRIVE}/BMS_Biosensors_Chroma.jpg"}[name]
det = mv.MultiDetector(views, weights, _m["task"], _m["conf"], _m["iou"], cfg["classes"])
frame = cv2.imread(PHOTO)
dets = det.detect(frame)[0]
groups = drop_wide(group_labels(dets, cfg["grouping"]["min_label_overlap"])[0])
mv.attach_alts(groups, dets.alt, cfg["grouping"]["min_label_overlap"])
reads = json.load(open(os.path.join(T, name, "reads.json")))
centre = next(r["center"] for r in reads if r["key"] == key)
print("boxes", len(groups), "with alts", sum(bool(g.alts) for g in groups), "alt dets", len(dets.alt),
      "alt boxes", sum(d.cls == "box" for d in dets.alt))
for i, g in enumerate(groups):
    if cv2.pointPolygonTest(g.box.poly.astype(np.float32), tuple(map(float, centre)), False) >= 0:
        print("asked box", i, "poly", np.round(g.box.poly).astype(int).tolist(), "labels",
              [np.round(l.poly).astype(int).tolist() for l in g.labels])
        for a in g.alts:
            print("   alt poly", np.round(a.box.poly).astype(int).tolist(), "iou", round(mv._iou(g.box.poly, a.box.poly), 3),
                  "labels", [np.round(l.poly).astype(int).tolist() for l in a.labels])

"""
Laptop speed of each compared YOLO model, the way the demo runs it: OpenVINO export at 544 x 960 on the Intel iGPU,
through src/box_detector.BoxDetector, on D455 shelf frames. Run ON THE CHARGER (battery = 3-8x slower).

  python scripts/yolo_laptop_fps.py --weights DIR_WITH_<model>.pt --frames VIDEO --out laptop_fps.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.box_detector import BoxDetector  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="folder with yolov8s-obb.pt, yolo11s-obb.pt, ... (fold 0)")
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    from ultralytics import YOLO
    cap = cv2.VideoCapture(a.frames)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(0, total - 1, a.n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    out = {}
    for pt in sorted(pathlib.Path(a.weights).glob("*.pt")):
        name = pt.stem
        ov = pt.with_name(f"{name}_960_openvino_model")
        if not ov.exists():
            exported = YOLO(str(pt)).export(format="openvino", imgsz=[544, 960], half=False, dynamic=False)
            pathlib.Path(exported).rename(ov)
        det = BoxDetector(str(ov), "obb", [544, 960], "intel:gpu.0", {"box": 0.35, "size_label": 0.45}, 0.5)
        for f in frames[:5]:
            det.detect(f)                                 # warm-up (first GPU calls compile kernels)
        t = []
        for f in frames:
            t0 = time.perf_counter()
            det.detect(f)
            t.append((time.perf_counter() - t0) * 1000)
        out[name] = {"ms": round(statistics.median(t), 1), "fps": round(1000 / statistics.median(t), 1),
                     "frames": len(t)}
        print(name, out[name], flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

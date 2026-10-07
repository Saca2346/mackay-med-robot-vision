#!/usr/bin/env python3
"""
JALANKAN DI SERVER (hucenrotia-ai), setelah training selesai.

1. Ekspor best.pt -> ONNX (format yang dipakai laptop).
2. Uji ONNX itu di CPU saja (simulasi laptop MX350): waktu per frame dan FPS.

Contoh:
  python scripts/export_box_onnx.py --weights runs/bootstrap_v1/weights/best.pt --image data/uji/images/contoh.jpg
Lalu salin models/box_obb.onnx ke laptop (scp / rsync, lihat docs/BOX_PIPELINE.md).
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.config import resolve


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="path best.pt hasil training")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default="models/box_obb.onnx")
    ap.add_argument("--image", default=None, help="foto untuk uji kecepatan CPU (opsional)")
    ap.add_argument("--runs", type=int, default=20)
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    print(f"[info] task={model.task}  kelas={model.names}")
    onnx_path = model.export(format="onnx", imgsz=args.imgsz, simplify=True, dynamic=False)
    dest = resolve(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(onnx_path, dest)
    print(f"[export] {onnx_path} -> {dest}  ({dest.stat().st_size / 1e6:.1f} MB)")

    cpu_model = YOLO(str(dest), task=model.task)
    if args.image:
        import cv2
        img = cv2.imread(args.image)
    else:
        img = (np.random.rand(720, 1280, 3) * 255).astype(np.uint8)
    cpu_model.predict(img, imgsz=args.imgsz, device="cpu", verbose=False)          # pemanasan
    times = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        r = cpu_model.predict(img, imgsz=args.imgsz, device="cpu", verbose=False)[0]
        times.append((time.perf_counter() - t0) * 1000)
    n = len(r.obb) if getattr(r, "obb", None) is not None else len(r.boxes)
    ms = float(np.median(times))
    print(f"[cpu] median {ms:.1f} ms/frame  -> {1000 / ms:.1f} FPS  (deteksi saja, {n} objek di foto uji)")
    print("       Catatan: CPU server (i9) lebih cepat dari laptop; angka FPS final WAJIB diukur ulang di laptop.")


if __name__ == "__main__":
    main()

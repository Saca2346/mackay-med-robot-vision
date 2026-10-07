#!/usr/bin/env python3
"""
Export a YOLO26n model to ONNX (design doc sections 3 & 4).

Default detector diganti dari YOLO11n ke YOLO26n (rilis Ultralytics Januari 2026):
diuji langsung di proyek ini (lihat catatan "YOLO26" di dokumen desain) -- decoder ONNX
custom kita cocok 5/5 dengan hasil resmi ultralytics.YOLO(...).predict() pada foto
referensi, dan ~15% lebih cepat di CPU pada sandbox pengembangan ini (40.6ms vs
47.6ms/frame) sambil parameternya sedikit lebih kecil (2.4M vs 2.6M) dan mAP COCO
sedikit lebih tinggi (40.9 vs 39.5). Tidak perlu perubahan kode di src/detector.py --
bentuk output ONNX-nya (1, 84, 8400) sama persis dengan YOLO11n.

Butuh ultralytics >= 8.4.0 (YOLO26 belum ada di rilis 8.3.x) -- lihat requirements.txt.

By default exports the COCO-pretrained yolo26n.pt as a placeholder so the whole
pipeline is runnable end-to-end before you have a labeled medicine dataset.
Once you've trained on your own SKUs (scripts/train.py), pass --weights to
export that checkpoint instead.

Usage:
    python scripts/export_model.py                          # COCO-pretrained placeholder
    python scripts/export_model.py --weights runs/train/exp/weights/best.pt
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.config import resolve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default="yolo26n.pt", help=".pt checkpoint to export (default: COCO-pretrained nano)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out-dir", type=str, default="models")
    ap.add_argument("--out-name", type=str, default=None, help="nama file .onnx tujuan (default: sama dengan basename --weights)")
    args = ap.parse_args()

    from ultralytics import YOLO

    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[export] memuat {args.weights} ...")
    model = YOLO(args.weights)

    print(f"[export] mengekspor ke ONNX @ {args.imgsz}px ...")
    exported_path = model.export(format="onnx", imgsz=args.imgsz, simplify=True, opset=12)

    out_name = args.out_name or (pathlib.Path(args.weights).stem + ".onnx")
    dest = out_dir / out_name
    pathlib.Path(exported_path).replace(dest)
    print(f"[export] selesai -> {dest}")
    print(
        "\nCatatan (bagian 1 & 9 dokumen desain): model ini masih COCO-pretrained "
        "(80 kelas umum), BUKAN model obat. Cukup untuk memvalidasi pipeline "
        "kamera->deteksi->tampilan. Latih ulang pada dataset kemasan obat Anda sendiri "
        "(scripts/train.py) sebelum digunakan untuk deteksi obat sungguhan."
    )


if __name__ == "__main__":
    main()

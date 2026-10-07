#!/usr/bin/env python3
"""
Training scaffold for a custom medicine-SKU YOLO26n model (design doc section 4).

You need a labeled dataset first (a few hundred - ~1,500 images per SKU, YOLO
format: images/ + labels/ + a data.yaml naming your classes). This script does
NOT come with that dataset — collect/label your own medicine packaging photos
(e.g. with Roboflow, LabelImg, or CVAT), then point --data at your data.yaml.

Recommended (section 4): run this on a cloud/free GPU (Kaggle/Colab) rather than
the MX350 laptop, then copy the resulting best.pt back and export it with
scripts/export_model.py.

Usage:
    python scripts/train.py --data path/to/your_medicine_dataset/data.yaml --epochs 120
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="path to your YOLO-format data.yaml")
    ap.add_argument("--model", type=str, default="yolo26n.pt", help="starting checkpoint (COCO-pretrained nano)")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8, help="4-8 realistic on a laptop; larger fine on cloud GPU")
    ap.add_argument("--device", type=str, default="0", help="'0' for first GPU, 'cpu' if none")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=30,       # early-stop if val mAP plateaus — saves cloud GPU hours
        workers=4,
    )
    print(
        "\nSelesai training. Cek runs/detect/train*/weights/best.pt lalu jalankan:\n"
        "  python scripts/export_model.py --weights runs/detect/train*/weights/best.pt"
    )


if __name__ == "__main__":
    main()

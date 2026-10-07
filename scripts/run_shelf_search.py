#!/usr/bin/env python3
"""
Demo: "cari paracetamol di rak 4 tingkat ini, ada di tingkat berapa?"

Extends Stage 1 of the design doc (section 7) to a multi-tier shelf. This is
the piece you flagged as missing: detect the shelf, then figure out which of
its tiers holds the target medicine, BEFORE the arm moves in and Stage 2
(wrist camera, close-up) takes over. Picking/grasping is intentionally not
part of this script.

By default runs against the synthetic ground-truth boxes (honest about not
yet having a trained detector — see README). Once you have a trained medicine
model and a real photo, pass --live to run the actual detector instead.

Usage:
    python scripts/run_shelf_search.py --target MED-001
    python scripts/run_shelf_search.py --target MED-001 --far           # simulate a distant/blurry shot
    python scripts/run_shelf_search.py --target MED-001 --live --image path/to/real_shelf_photo.jpg
"""
import argparse
import json
import pathlib
import sys

import cv2

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.config import load_config, resolve
from src.database import MedicineDB
from src.detector import YoloOnnxDetector
from src.pipeline import COCO_CLASSES
from src.shelf_scanner import ShelfScanner
from src.verifier import ObjectVerifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="SKU obat yang dicari, mis. MED-001")
    ap.add_argument("--far", action="store_true", help="pakai gambar sintetis versi blur/jauh")
    ap.add_argument("--live", action="store_true", help="pakai detektor YOLO sungguhan, bukan ground-truth sintetis")
    ap.add_argument("--image", type=str, default=None, help="gambar rak (wajib jika --live)")
    ap.add_argument("--config", type=str, default=None,
                     help="pakai profil config lain, mis. config/book_pipeline_config.yaml (default: "
                          "config/pipeline_config.yaml, domain obat) -- tier rak (config.shelf) dipakai apa "
                          "adanya untuk domain apapun, termasuk 'rak buku', tanpa perlu model/kelas baru.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    db = MedicineDB(str(resolve(cfg["database"]["path"])))
    verifier = ObjectVerifier(db, cfg["verification"])

    if args.live:
        if not args.image:
            print("Error: --live butuh --image path/ke/foto_rak.jpg")
            sys.exit(1)
        onnx_path = resolve(cfg["model"]["onnx_path"])
        detector = YoloOnnxDetector(
            str(onnx_path), input_size=cfg["model"]["input_size"],
            conf_threshold=cfg["model"]["conf_threshold"], class_names=COCO_CLASSES,
        )
        scanner = ShelfScanner(detector=detector, verifier=verifier, cfg=cfg["shelf"])
        frame = cv2.imread(args.image)
        result = scanner.scan(frame, args.target)
    else:
        suffix = "_far_blurry" if args.far else ""
        img_path = resolve(f"data/test_assets/four_tier_shelf{suffix}.png")
        gt_path = resolve(f"data/test_assets/four_tier_shelf{suffix}_groundtruth.json")
        if not img_path.exists():
            print(f"Gambar uji tidak ada di {img_path}. Jalankan scripts/generate_test_assets.py dulu.")
            sys.exit(1)
        frame = cv2.imread(str(img_path))
        gt = json.loads(gt_path.read_text())
        boxes = [tuple(item["box"]) for item in gt["items"]]
        scanner = ShelfScanner(detector=None, verifier=verifier, cfg=cfg["shelf"])
        candidates = scanner.scan_frames_for_ground_truth_boxes(frame, boxes)
        result = scanner.find_target(candidates, args.target)
        print(f"(mode ground-truth sintetis — {'jauh/blur' if args.far else 'dekat/tajam'}; "
              f"pakai --live --image <foto> setelah model dilatih untuk hasil sungguhan)\n")

    print(f"Mencari: {args.target}")
    tiers = result.tier_summary()
    for tier_idx in sorted(tiers):
        items = ", ".join(f"{c.name or 'tidak dikenal'} [{c.match_kind}]" for c in tiers[tier_idx])
        marker = "  <-- TARGET DITEMUKAN DI SINI" if (result.found and tier_idx == result.tier_index) else ""
        print(f"  tingkat {tier_idx}: {items}{marker}")

    if result.found:
        print(f"\n=> {args.target} ada di TINGKAT {result.tier_index}. "
              f"Arahkan lengan/kamera wrist ke tingkat ini untuk konfirmasi jarak-dekat (Stage 2, bagian 7).")
    else:
        print(f"\n=> {args.target} TIDAK ditemukan di rak ini pada pemindaian saat ini. "
              f"Pertimbangkan: ganti sudut kamera, dekatkan, atau rak memang tidak punya stok SKU ini.")

    db.close()


if __name__ == "__main__":
    main()

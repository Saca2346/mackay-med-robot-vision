#!/usr/bin/env python3
"""Create + seed the local medicine database (design doc sections 2 & 5).

Run after generate_test_assets.py so reference crops exist. Replace this seed
data with your real SKU catalogue once you have actual product photos.
"""
import pathlib
import sys

import cv2

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.config import load_config, resolve
from src.database import MedicineDB, MedicineRecord, color_histogram

MEDICINES = [
    ("MED-001", "Paracetamol", "500mg", "paracetamol 500mg"),
    ("MED-002", "Amoxicillin", "250mg", "amoxicillin 250mg"),
    ("MED-003", "Cetirizine", "10mg", "cetirizine 10mg"),
]


def main():
    cfg = load_config()
    db_path = resolve(cfg["database"]["path"])
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = MedicineDB(str(db_path))

    ref_dir = resolve("data/medicine_images")
    bins = cfg["verification"]["color_hist_bins"]

    for sku, name, dose, label_text in MEDICINES:
        ref_path = ref_dir / f"{sku}_ref.png"
        if not ref_path.exists():
            print(f"[skip] referensi tidak ada untuk {sku}: {ref_path} (jalankan generate_test_assets.py)")
            continue
        ref_img = cv2.imread(str(ref_path))
        hist = color_histogram(ref_img, bins=bins)
        db.upsert(
            MedicineRecord(
                sku=sku,
                name=name,
                dose=dose,
                label_text=label_text,
                color_hist=hist,
                ref_image_path=str(ref_path),
                stock_count=20,
            )
        )
        print(f"[db] tersimpan {sku} — {name} {dose}")

    db.close()
    print(f"\nDatabase siap di {db_path}")


if __name__ == "__main__":
    main()

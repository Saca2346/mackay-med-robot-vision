#!/usr/bin/env python3
"""
Component-level smoke tests that do NOT depend on the YOLO detector or any
camera hardware — they validate verifier.py, database.py and depth_estimator.py
directly against the synthetic reference crops, which is what actually proves
those modules work (a COCO-pretrained model won't reliably detect hand-drawn
synthetic packages, so a full-pipeline run alone wouldn't exercise this code).

Run: python tests/test_components.py
(after scripts/generate_test_assets.py and scripts/init_db.py)
"""
import pathlib
import sys

import cv2

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.config import load_config, resolve
from src.database import MedicineDB
from src.depth_estimator import MockDepth
from src.verifier import ObjectVerifier

PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"


def check(label, cond):
    print(f"[{PASS if cond else FAIL}] {label}")
    return cond


def main():
    cfg = load_config()
    ok = True

    # --- database + verifier -------------------------------------------------
    db_path = resolve(cfg["database"]["path"])
    ok &= check(f"database ada di {db_path}", db_path.exists())
    db = MedicineDB(str(db_path))
    records = db.all_records()
    ok &= check(f"database berisi record (>=1): {len(records)} SKU", len(records) >= 1)

    verifier = ObjectVerifier(db, cfg["verification"])

    ref_dir = resolve("data/medicine_images")
    for rec in records:
        crop = cv2.imread(rec.ref_image_path)
        result = verifier.verify(crop)
        matched = result.sku == rec.sku
        ok &= check(
            f"verifikasi {rec.sku} ({rec.name}) pada crop referensinya sendiri -> "
            f"cocok={result.sku} via {result.confidence}",
            matched,
        )

    # a crop that should NOT match anything (plain gray square, no label/color match)
    import numpy as np
    blank = np.full((100, 100, 3), 128, dtype=np.uint8)
    result = verifier.verify(blank)
    ok &= check(
        f"crop kosong tidak salah cocok ke SKU manapun (dapat={result.sku})",
        result.sku is None,
    )

    # --- depth (mock) ----------------------------------------------------------
    depth = MockDepth(cfg["depth"])
    near_box = (500, 300, 620, 380)   # wide box -> should estimate as "closer"
    far_box = (500, 300, 540, 340)    # narrow box -> should estimate as "farther"
    pt_near = depth.estimate_from_box(near_box, (720, 1280, 3))
    pt_far = depth.estimate_from_box(far_box, (720, 1280, 3))
    ok &= check(
        f"kotak lebih lebar -> Z lebih dekat (near.z={pt_near.z:.3f}m < far.z={pt_far.z:.3f}m)",
        pt_near.z < pt_far.z,
    )
    ok &= check(
        f"Z dalam rentang masuk akal jangkauan lengan (0.10-1.20m): near={pt_near.z:.3f}, far={pt_far.z:.3f}",
        0.10 <= pt_near.z <= 1.20 and 0.10 <= pt_far.z <= 1.20,
    )

    db.close()
    print("\n" + ("SEMUA TES LULUS" if ok else "ADA TES YANG GAGAL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

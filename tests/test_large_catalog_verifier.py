#!/usr/bin/env python3
"""
Proves ObjectVerifier's large-catalog path (src/verifier.py, added for the
110,000-SKU target) still finds the right medicine when the catalog is stuffed
with thousands of decoy SKUs -- not just that the vectorized retrieval math is
correct in isolation (tests/test_sku_retrieval.py already covers that), but that
the two-tier verify() pipeline as a whole (color retrieval -> OCR/shape on the
shortlist) gives the SAME answer the small-catalog loop already gives, at a
catalog size where the small-catalog loop would be too slow to use for real.

Run: python tests/test_large_catalog_verifier.py
(after scripts/generate_test_assets.py and scripts/init_db.py)
"""
import pathlib
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.config import load_config, resolve
from src.database import MedicineDB, MedicineRecord, color_histogram
from src.verifier import ObjectVerifier

PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"


def check(label, cond):
    print(f"[{PASS if cond else FAIL}] {label}")
    return cond


def main():
    cfg = load_config()
    db_path = resolve(cfg["database"]["path"])
    real_db = MedicineDB(str(db_path))
    real_records = real_db.all_records()
    if not real_records:
        print("[skip] medicine_db.sqlite kosong -- jalankan scripts/init_db.py dulu")
        return

    # In-memory DB: 3 SKU asli + 4000 SKU pengecoh (histogram warna acak, tanpa
    # label_text/ref_image supaya tidak pernah salah cocok) -- ini yang membuat
    # katalog melewati large_catalog_threshold (default 500) di verifier.py.
    mem_db = MedicineDB(":memory:")
    for rec in real_records:
        mem_db.upsert(rec)

    rng = np.random.default_rng(7)
    n_decoys = 4000
    bins_shape = real_records[0].color_hist.shape[0]
    for i in range(n_decoys):
        mem_db.upsert(
            MedicineRecord(
                sku=f"DECOY-{i:05d}",
                name="",
                dose="",
                label_text="",
                color_hist=rng.random(bins_shape).astype(np.float32),
                ref_image_path=None,
                stock_count=0,
            )
        )

    verifier = ObjectVerifier(mem_db, cfg["verification"])
    ok = True
    ok &= check(
        f"katalog gabungan melewati ambang large-catalog ({len(verifier._records)} > {verifier.large_catalog_threshold})",
        verifier._is_large_catalog(),
    )

    ref_dir = resolve("data/medicine_images")
    for rec in real_records:
        crop_path = ref_dir / f"{rec.sku}_ref.png"
        if not crop_path.exists():
            continue
        crop = cv2.imread(str(crop_path))
        t0 = time.perf_counter()
        result = verifier.verify(crop)
        dt_ms = (time.perf_counter() - t0) * 1000
        ok &= check(
            f"{rec.sku} ({rec.name}) tetap ditemukan benar di katalog {len(verifier._records)} SKU "
            f"via {result.confidence} ({dt_ms:.1f}ms)",
            result.sku == rec.sku,
        )

    # crop kosong / tidak dikenal tetap no_match, bukan salah cocok ke salah satu decoy
    blank = np.zeros((50, 50, 3), dtype=np.uint8)
    result = verifier.verify(blank)
    ok &= check(f"crop kosong di katalog besar -> no_match (dapat: {result.confidence})", result.sku is None)

    print("\n" + ("SEMUA TES LULUS" if ok else "ADA TES YANG GAGAL"))
    real_db.close()
    mem_db.close()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

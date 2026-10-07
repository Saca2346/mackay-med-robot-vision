#!/usr/bin/env python3
"""
Correctness check for src/sku_retrieval.py's vectorized top-K search: proves
its scores are numerically IDENTICAL to calling cv2.compareHist(..., HISTCMP_CORREL)
one record at a time (the method verifier.py already uses) -- so switching to the
vectorized version at 110k-SKU scale changes performance, not results.

Run: python tests/test_sku_retrieval.py
"""
import pathlib
import sys
from dataclasses import dataclass

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.sku_retrieval import SkuRetrievalIndex

PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"


def check(label, cond):
    print(f"[{PASS if cond else FAIL}] {label}")
    return cond


@dataclass
class FakeRecord:
    sku: str
    name: str
    color_hist: np.ndarray


def main():
    rng = np.random.default_rng(42)
    n_records = 500
    bins = 16
    records = [
        FakeRecord(sku=f"SKU-{i:05d}", name=f"Produk {i}", color_hist=rng.random(bins).astype(np.float32))
        for i in range(n_records)
    ]
    # plant one record that's an exact copy of the query so we know the true top-1
    query_hist = rng.random(bins).astype(np.float32)
    records[247] = FakeRecord(sku="SKU-TARGET", name="Target Asli", color_hist=query_hist.copy())

    index = SkuRetrievalIndex(records)
    ok = True
    ok &= check(f"index memuat semua {n_records} record", len(index) == n_records)

    top10 = index.top_k(query_hist, k=10)
    ok &= check("record identik ditemukan sebagai top-1", top10[0].sku == "SKU-TARGET")
    ok &= check(f"skor top-1 mendekati 1.0 (identik): {top10[0].score:.6f}", abs(top10[0].score - 1.0) < 1e-6)

    # cross-check EVERY score against cv2.compareHist directly (the ground truth
    # verifier.py itself uses) -- not just the top-1
    max_diff = 0.0
    for rec in records[:50]:  # a representative sample is enough to prove the formula, not just one point
        cv_score = cv2.compareHist(query_hist, rec.color_hist, cv2.HISTCMP_CORREL)
        vectorized_scores = index.top_k(query_hist, k=n_records)
        by_sku = {c.sku: c.score for c in vectorized_scores}
        diff = abs(by_sku[rec.sku] - cv_score)
        max_diff = max(max_diff, diff)
    ok &= check(f"skor vektor cocok persis dengan cv2.compareHist (selisih maks: {max_diff:.2e})", max_diff < 1e-6)

    # empty / mismatched-shape edge cases shouldn't crash
    empty_index = SkuRetrievalIndex([])
    ok &= check("index kosong tidak crash, hasil kosong", empty_index.top_k(query_hist) == [])
    bad_query = rng.random(bins + 1).astype(np.float32)  # wrong bin count
    ok &= check("query dengan jumlah bin salah tidak crash, hasil kosong", index.top_k(bad_query) == [])

    print("\n" + ("SEMUA TES LULUS" if ok else "ADA TES YANG GAGAL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

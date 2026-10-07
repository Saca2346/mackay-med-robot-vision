#!/usr/bin/env python3
"""
Measures -- not guesses -- why verifier.py's per-record Python loop (used for
the current 3-SKU medicine_db.sqlite) cannot scale to a ~110,000-SKU catalog,
and confirms src/sku_retrieval.py's vectorized approach can.

Method: build synthetic color histograms for a range of catalog sizes, time
(a) the naive approach (a Python loop calling cv2.compareHist once per record,
    exactly what verifier.py does today) and
(b) SkuRetrievalIndex.top_k (one vectorized numpy operation),
then extrapolate (a)'s measured per-record cost to 110,000 to show the actual
time it would add per detected object, per frame -- and show (b)'s real,
measured time at n=110,000 directly (no extrapolation needed, it's fast enough
to just run).

Run: python scripts/benchmark_sku_scale.py
"""
import pathlib
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.sku_retrieval import SkuRetrievalIndex


@dataclass
class FakeRecord:
    sku: str
    name: str
    color_hist: np.ndarray


def naive_loop_search(records, query_hist):
    """Exactly what verifier.py's color-matching fallback does today: a Python
    for-loop calling cv2.compareHist once per record."""
    best = None
    for rec in records:
        score = cv2.compareHist(query_hist, rec.color_hist, cv2.HISTCMP_CORREL)
        if best is None or score > best[0]:
            best = (score, rec.sku)
    return best


def main():
    rng = np.random.default_rng(0)
    bins = 16

    print("=== Biaya per-record dari pendekatan saat ini (loop Python + cv2.compareHist) ===\n")
    sample_sizes = [100, 1_000, 5_000]
    per_record_costs = []
    for n in sample_sizes:
        records = [
            FakeRecord(f"SKU-{i}", f"Produk {i}", rng.random(bins).astype(np.float32))
            for i in range(n)
        ]
        query = rng.random(bins).astype(np.float32)
        # warmup
        naive_loop_search(records, query)
        t0 = time.perf_counter()
        naive_loop_search(records, query)
        dt = time.perf_counter() - t0
        per_record_us = (dt / n) * 1e6
        per_record_costs.append(per_record_us)
        print(f"  n={n:>6}: {dt*1000:.2f} ms total -> {per_record_us:.2f} us/record")

    avg_per_record_us = sum(per_record_costs) / len(per_record_costs)
    projected_110k_ms = avg_per_record_us * 110_000 / 1000
    print(f"\n  Rata-rata biaya per record: {avg_per_record_us:.2f} us")
    print(f"  EKSTRAPOLASI ke 110.000 SKU: ~{projected_110k_ms:.0f} ms PER OBJEK PER FRAME "
          f"hanya untuk tahap warna -- sebelum shape matching & OCR ditambahkan, dan sebelum "
          f"dikali jumlah objek yang terdeteksi di satu frame. Ini sudah lebih lambat dari "
          f"anggaran waktu satu frame real-time (~33ms @ 30fps).")

    print("\n=== Biaya SkuRetrievalIndex (vektorisasi numpy) pada skala nyata ===\n")
    for n in (1_000, 10_000, 110_000):
        records = [
            FakeRecord(f"SKU-{i:06d}", f"Produk {i}", rng.random(bins).astype(np.float32))
            for i in range(n)
        ]
        query = rng.random(bins).astype(np.float32)
        t0 = time.perf_counter()
        index = SkuRetrievalIndex(records)
        build_ms = (time.perf_counter() - t0) * 1000

        # warmup then measure query time
        index.top_k(query, k=20)
        t0 = time.perf_counter()
        top20 = index.top_k(query, k=20)
        query_ms = (time.perf_counter() - t0) * 1000

        print(f"  n={n:>7}: build index sekali = {build_ms:.1f} ms, "
              f"top-20 per query = {query_ms:.2f} ms  (top-1: {top20[0].sku}, skor {top20[0].score:.3f})")

    print(
        "\nKesimpulan: build index sekali per sesi (atau saat katalog berubah), lalu tiap "
        "query top-K tetap sub-milidetik bahkan pada 110.000 SKU -- karena satu operasi "
        "matriks numpy menggantikan 110.000 pemanggilan cv2.compareHist satu per satu. "
        "Shape matching + OCR substring check (yang jauh lebih mahal per item) baru dijalankan "
        "pada hasil top-K ini (puluhan kandidat), bukan pada seluruh katalog."
    )


if __name__ == "__main__":
    main()

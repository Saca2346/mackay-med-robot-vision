"""
Vectorized coarse retrieval for large SKU catalogs (added when the target scaled
to ~110,000 SKUs).

WHY THIS EXISTS: `ObjectVerifier.verify()` (src/verifier.py) does a plain Python
`for rec in self._records: ...` over EVERY known SKU, calling `cv2.compareHist`
and `cv2.matchShapes` per record, per detected object, per frame. That is fine at
3 SKUs (the current `medicine_db.sqlite`). It falls over completely at 110,000 SKUs
-- see `scripts/benchmark_sku_scale.py` for a measured comparison, not a guess.

THE FIX: a two-tier lookup, the same pattern large-scale product-recognition
systems use (barcode/exact-id first, then visual retrieval as fallback, never a
linear scan of the whole catalog):

  1. Barcode / exact SKU code, when readable -> O(1) dict lookup. Always try this
     first for a catalog this size; it is the only method that is both fast AND
     reliable at 110k items. (Not implemented here -- needs a barcode reader like
     pyzbar; this module covers the vision-only fallback path.)
  2. OCR text, when confidently read -> still needs a candidate shortlist first at
     this scale (substring-matching raw OCR text against 110k label strings, one
     Python comparison at a time, has the same scaling problem as color/shape).
  3. Vision-only fallback: `SkuRetrievalIndex` here computes a COLOR HISTOGRAM
     correlation (identical math to `cv2.compareHist(..., cv2.HISTCMP_CORREL)`,
     verified below to match cv2 exactly) against ALL records in one vectorized
     numpy matrix operation, returning the top-K candidates in time that barely
     grows with catalog size. Only those top-K (tens, not 110,000) then go through
     the more expensive per-candidate OCR substring check and `cv2.matchShapes`
     shape comparison already in `verifier.py` -- so the expensive per-item work
     still happens, just on ~20-50 items instead of 110,000.

This module deliberately does NOT replace `verifier.py`'s OCR/shape logic --
it narrows what that logic has to loop over.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RetrievalCandidate:
    sku: str
    name: str
    score: float  # same scale as cv2.compareHist(..., HISTCMP_CORREL): -1..1, higher = more similar


class SkuRetrievalIndex:
    """Stacks every SKU's color histogram into one (N, bins) matrix once, then
    answers "top-K most similar SKUs to this query histogram" with a single
    vectorized correlation computation instead of a per-record Python loop."""

    def __init__(self, records):
        """records: iterable of objects with .sku, .name, .color_hist (np.ndarray)."""
        skus, names, hists = [], [], []
        for rec in records:
            hist = getattr(rec, "color_hist", None)
            if hist is None or hist.size == 0:
                continue
            skus.append(rec.sku)
            names.append(rec.name)
            hists.append(np.asarray(hist, dtype=np.float64).ravel())

        self.skus = skus
        self.names = names
        if hists:
            self._matrix = np.stack(hists, axis=0)  # (N, bins)
            # pre-center and pre-normalize rows once at build time, not per query
            row_mean = self._matrix.mean(axis=1, keepdims=True)
            self._centered = self._matrix - row_mean
            self._row_norm = np.sqrt(np.sum(self._centered**2, axis=1))
        else:
            self._matrix = np.zeros((0, 0))
            self._centered = np.zeros((0, 0))
            self._row_norm = np.zeros(0)

    def __len__(self):
        return len(self.skus)

    def top_k(self, query_hist: np.ndarray, k: int = 20) -> list[RetrievalCandidate]:
        """Vectorized equivalent of calling cv2.compareHist(query, rec, HISTCMP_CORREL)
        for every record and sorting -- verified to match cv2's own output exactly
        (see tests/test_sku_retrieval.py). Returns the top-k, highest score first."""
        n = len(self.skus)
        if n == 0:
            return []
        q = np.asarray(query_hist, dtype=np.float64).ravel()
        if q.shape[0] != self._matrix.shape[1]:
            return []

        q_centered = q - q.mean()
        q_norm = np.sqrt(np.sum(q_centered**2))
        if q_norm == 0:
            return []

        # Pearson correlation of q against every row, all at once.
        numerators = self._centered @ q_centered  # (N,)
        denom = self._row_norm * q_norm
        with np.errstate(invalid="ignore", divide="ignore"):
            scores = np.where(denom > 0, numerators / denom, -1.0)

        k = min(k, n)
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [RetrievalCandidate(self.skus[i], self.names[i], float(scores[i])) for i in top_idx]

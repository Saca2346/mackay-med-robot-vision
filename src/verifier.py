"""
Stage 2 of the architecture (design doc sections 2 & 5): Object Verification.

Combines OCR + color histogram + contour/shape matching against the local
medicine database, entirely on CPU, running only on the crops the detector
already isolated (never on the full frame).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pytesseract

from src.database import MedicineDB, color_histogram
from src.sku_retrieval import SkuRetrievalIndex


@dataclass
class VerificationResult:
    sku: str | None
    name: str | None
    confidence: str  # "ocr_match" | "color_shape_match" | "no_match"
    ocr_text: str
    color_score: float | None
    shape_score: float | None


def _normalize_text(t: str) -> str:
    return " ".join(t.lower().split())


def _shape_score(crop_gray: np.ndarray, ref_gray: np.ndarray) -> float | None:
    def largest_contour(gray):
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)

    c1, c2 = largest_contour(crop_gray), largest_contour(ref_gray)
    if c1 is None or c2 is None:
        return None
    return cv2.matchShapes(c1, c2, cv2.CONTOURS_MATCH_I1, 0.0)


class ObjectVerifier:
    def __init__(self, db: MedicineDB, cfg: dict):
        self.db = db
        self.ocr_lang = cfg.get("ocr_lang", "eng")
        self.ocr_min_conf = cfg.get("ocr_min_confidence", 40)
        self.color_bins = cfg.get("color_hist_bins", 16)
        self.shape_threshold = cfg.get("shape_match_threshold", 0.25)
        # Katalog kecil (default): loop lengkap seperti semula, sudah teruji.
        # Katalog besar (ratusan-ribu SKU, mis. target 110k): loop penuh per-record
        # tidak lagi masuk akal untuk real-time (lihat scripts/benchmark_sku_scale.py --
        # ~39ms/objek/frame HANYA untuk tahap warna pada 110k SKU, sebelum shape+OCR).
        # Di atas ambang ini, retrieval warna memakai SkuRetrievalIndex (satu operasi
        # matriks numpy, skornya identik dengan cv2.compareHist per-record -- lihat
        # tests/test_sku_retrieval.py) untuk mempersempit ke top-K kandidat SEBELUM
        # OCR/shape matching yang jauh lebih mahal per item dijalankan.
        self.large_catalog_threshold = cfg.get("large_catalog_threshold", 500)
        self.retrieval_top_k = cfg.get("retrieval_top_k", 50)
        self._records = db.all_records()
        self._ref_gray_cache: dict[str, np.ndarray] = {}
        self._retrieval_index: SkuRetrievalIndex | None = None

    def refresh(self):
        self._records = self.db.all_records()
        self._retrieval_index = None  # dibangun ulang malas (lazy) pada verify() berikutnya

    def _is_large_catalog(self) -> bool:
        return len(self._records) > self.large_catalog_threshold

    def _get_retrieval_index(self) -> SkuRetrievalIndex:
        if self._retrieval_index is None:
            self._retrieval_index = SkuRetrievalIndex(self._records)
        return self._retrieval_index

    def _ocr(self, crop_bgr: np.ndarray) -> tuple[str, float]:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        try:
            data = pytesseract.image_to_data(
                gray, lang=self.ocr_lang, output_type=pytesseract.Output.DICT
            )
        except pytesseract.TesseractNotFoundError:
            return "", 0.0
        words, confs = [], []
        for word, conf in zip(data["text"], data["conf"]):
            word = word.strip()
            conf = float(conf) if conf not in ("-1", -1) else -1.0
            if word and conf >= 0:
                words.append(word)
                confs.append(conf)
        text = _normalize_text(" ".join(words))
        avg_conf = float(np.mean(confs)) if confs else 0.0
        return text, avg_conf

    def _shape_score_for(self, crop_gray: np.ndarray, rec) -> float | None:
        if not rec.ref_image_path:
            return None
        ref_gray = self._ref_gray_cache.get(rec.sku)
        if ref_gray is None:
            ref_img = cv2.imread(rec.ref_image_path)
            if ref_img is not None:
                ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
                self._ref_gray_cache[rec.sku] = ref_gray
        if ref_gray is None:
            return None
        return _shape_score(crop_gray, ref_gray)

    def verify(self, crop_bgr: np.ndarray) -> VerificationResult:
        if crop_bgr.size == 0 or not self._records:
            return VerificationResult(None, None, "no_match", "", None, None)

        ocr_text, ocr_conf = self._ocr(crop_bgr)
        crop_hist = color_histogram(crop_bgr, bins=self.color_bins)
        crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        ocr_ok = bool(ocr_text) and ocr_conf >= self.ocr_min_conf

        if self._is_large_catalog():
            # Katalog besar: retrieval warna vektorisasi dulu (top-K, sub-milidetik
            # bahkan di 110k SKU -- lihat scripts/benchmark_sku_scale.py), BARU OCR
            # substring check + shape matching (mahal per item) dijalankan pada
            # kandidat top-K itu saja, bukan seluruh katalog.
            candidates = self._get_retrieval_index().top_k(crop_hist, k=self.retrieval_top_k)
            by_sku = {rec.sku: rec for rec in self._records}

            if ocr_ok:
                for cand in candidates:
                    rec = by_sku.get(cand.sku)
                    if rec and rec.label_text and (rec.label_text in ocr_text or ocr_text in rec.label_text):
                        return VerificationResult(rec.sku, rec.name, "ocr_match", ocr_text, None, None)

            best = None
            for cand in candidates:
                rec = by_sku.get(cand.sku)
                if rec is None:
                    continue
                shape_score = self._shape_score_for(crop_gray, rec)
                shape_ok = shape_score is not None and shape_score <= self.shape_threshold
                candidate_score = cand.score - (shape_score or 0.0)
                if cand.score > 0.6 and (shape_score is None or shape_ok):
                    if best is None or candidate_score > best[0]:
                        best = (candidate_score, rec, cand.score, shape_score)

            if best is not None:
                _, rec, color_score, shape_score = best
                return VerificationResult(rec.sku, rec.name, "color_shape_match", ocr_text, color_score, shape_score)
            return VerificationResult(None, None, "no_match", ocr_text, None, None)

        # Katalog kecil (default, termasuk medicine_db.sqlite saat ini): loop penuh
        # seperti semula -- perilaku persis sama dengan sebelumnya, sudah teruji
        # (tests/test_components.py).

        # 1) OCR match: label text is a substring / high overlap with a known SKU's text
        if ocr_ok:
            for rec in self._records:
                if rec.label_text and (rec.label_text in ocr_text or ocr_text in rec.label_text):
                    return VerificationResult(rec.sku, rec.name, "ocr_match", ocr_text, None, None)

        # 2) Fallback: color histogram + shape matching against every known SKU
        best = None
        for rec in self._records:
            if rec.color_hist.shape != crop_hist.shape:
                continue
            color_score = cv2.compareHist(
                crop_hist.astype(np.float32), rec.color_hist.astype(np.float32), cv2.HISTCMP_CORREL
            )
            shape_score = self._shape_score_for(crop_gray, rec)
            shape_ok = shape_score is not None and shape_score <= self.shape_threshold
            candidate_score = color_score - (shape_score or 0.0)
            if color_score > 0.6 and (shape_score is None or shape_ok):
                if best is None or candidate_score > best[0]:
                    best = (candidate_score, rec, color_score, shape_score)

        if best is not None:
            _, rec, color_score, shape_score = best
            return VerificationResult(
                rec.sku, rec.name, "color_shape_match", ocr_text, color_score, shape_score
            )

        return VerificationResult(None, None, "no_match", ocr_text, None, None)

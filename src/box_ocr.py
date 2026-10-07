"""
Membaca teks di dalam crop size_label.

Engine default: RapidOCR (model PP-OCR milik PaddleOCR yang sudah dikonversi ke
ONNX, jalan di onnxruntime CPU, cukup `pip install rapidocr-onnxruntime`).
Cadangan: Tesseract (pytesseract), yang sudah dipakai modul lama proyek ini.

OCR hanya dijalankan pada crop size_label (kecil), tidak pada seluruh frame,
dan tidak di setiap frame -- lihat scripts/test_webcam_box.py.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class OcrToken:
    text: str
    conf: float        # 0..1


@dataclass
class OcrResult:
    tokens: list[OcrToken] = field(default_factory=list)
    ms: float = 0.0
    flipped: bool = False

    @property
    def text(self) -> str:
        return " ".join(t.text for t in self.tokens)

    @property
    def mean_conf(self) -> float:
        return float(np.mean([t.conf for t in self.tokens])) if self.tokens else 0.0


class SizeTextReader:
    def __init__(self, engine: str = "rapidocr", retry_flip_below: float = 0.80, det_limit=("max", 1280),
                 threads: int = 0, use_cuda: bool = False):
        """det_limit: ukuran gambar untuk detektor teks RapidOCR. Bawaan RapidOCR ("min", 736) memperbesar crop
        panjang-tipis sampai detektor gagal (tidak ada teks sama sekali) dan 10x lebih lambat; ("max", 1280)
        membaca paling banyak di 89 crop size_label dataset (lihat docs/BOX_PIPELINE.md).
        threads: thread CPU untuk satu mesin OCR (0 = semua core). Beberapa mesin OCR paralel yang masing-masing
        memakai SEMUA core saling berebut (terukur 1-2 detik per kotak di server) -> bagi core per mesin.
        use_cuda: OCR di GPU NVIDIA (butuh paket onnxruntime-gpu); kalau tidak tersedia RapidOCR kembali ke CPU."""
        self.engine_name = engine
        self.retry_flip_below = retry_flip_below
        self._engine = None
        self._api = 1
        if engine == "rapidocr":
            try:
                from rapidocr_onnxruntime import RapidOCR
                self._engine = RapidOCR(det_limit_type=det_limit[0], det_limit_side_len=det_limit[1])
            except ImportError:
                try:
                    # rapidocr_onnxruntime hanya ada untuk Python < 3.13; penerusnya paket `rapidocr` (>= 2)
                    from rapidocr import RapidOCR
                    params = {"Det.limit_type": det_limit[0], "Det.limit_side_len": det_limit[1]}
                    if threads > 0:
                        params["EngineConfig.onnxruntime.intra_op_num_threads"] = int(threads)
                        params["EngineConfig.onnxruntime.inter_op_num_threads"] = 1
                    if use_cuda:
                        params["EngineConfig.onnxruntime.use_cuda"] = True
                    self._engine = RapidOCR(params=params)
                    self._api = 2
                    # tanpa ini setiap crop tanpa teks mencetak "[WARNING] The text detection result is empty"
                    logging.getLogger("RapidOCR").setLevel(logging.ERROR)
                except ImportError as e:
                    # rapidocr >= 2 tidak memasang onnxruntime sendiri -> ImportError saat RapidOCR() dibuat
                    print(f"[ocr] RapidOCR tidak bisa dipakai ({e}) -> coba tesseract. "
                          "Pasang: pip install rapidocr onnxruntime")
                    self.engine_name = "tesseract"
        if self.engine_name == "tesseract":
            try:
                import pytesseract  # noqa: F401
            except ImportError:
                raise SystemExit("OCR tidak tersedia. Pasang: pip install rapidocr onnxruntime "
                                 "(disarankan) atau pytesseract + program Tesseract.")
        # pemanasan: panggilan pertama memuat model OCR (lambat), jangan ikut dihitung
        self._run(np.full((48, 160, 3), 255, dtype=np.uint8))

    def _run(self, img: np.ndarray) -> list[OcrToken]:
        if self.engine_name == "rapidocr":
            if self._api == 1:
                result, _ = self._engine(img)
            else:
                out = self._engine(img)
                result = [] if out.txts is None else list(zip(out.boxes, out.txts, out.scores))
            if not result:
                return []
            # urutkan baca: atas->bawah, kiri->kanan
            result = sorted(result, key=lambda r: (round(np.mean([p[1] for p in r[0]]) / 20), np.mean([p[0] for p in r[0]])))
            return [OcrToken(str(r[1]).strip(), float(r[2])) for r in result if str(r[1]).strip()]
        import pytesseract
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        try:
            d = pytesseract.image_to_data(gray, config="--psm 6", output_type=pytesseract.Output.DICT)
        except pytesseract.TesseractNotFoundError:
            return []
        out = []
        for w, c in zip(d["text"], d["conf"]):
            c = float(c)
            if w.strip() and c >= 0:
                out.append(OcrToken(w.strip(), c / 100.0))
        return out

    def read(self, crop: np.ndarray) -> OcrResult:
        """
        Baca satu crop size_label. Teks ukuran di punggung kemasan berjalan SEPANJANG punggung
        (Terumo, Boston Scientific, iVascular sama), jadi crop yang tegak dibaca dulu setelah diputar
        90 derajat ke kiri, crop yang mendatar dibaca apa adanya; arah lain hanya dicoba kalau hasilnya
        belum jelas. Skor arah = jumlah karakter terbaca x confidence: arah yang salah memecah teks jadi
        potongan pendek ("2", "75", "<L>") padahal arah benar membaca "2.75", "Accuforce".
        Terbalik 180 derajat ditangani classifier arah bawaan RapidOCR.
        """
        t0 = time.perf_counter()
        if crop is None or crop.size == 0 or min(crop.shape[:2]) < 4:
            return OcrResult(ms=0.0)
        tall = crop.shape[0] > 1.5 * crop.shape[1]
        order = ((cv2.ROTATE_90_COUNTERCLOCKWISE, None, cv2.ROTATE_90_CLOCKWISE) if tall
                 else (None, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_90_CLOCKWISE))
        best = None
        for rot in order:
            img = crop if rot is None else cv2.rotate(crop, rot)
            res = OcrResult(tokens=self._run(img), flipped=rot is not None)
            score = sum(t.conf * len(t.text) for t in res.tokens)
            if best is None or score > best[0]:
                best = (score, res)
            b = best[1]
            if b.tokens and b.mean_conf >= self.retry_flip_below and max(len(t.text) for t in b.tokens) >= 4:
                break                                        # jelas terbaca: arah lain tidak perlu dicoba
        res = best[1]
        res.ms = (time.perf_counter() - t0) * 1000.0
        return res

    def read_orient(self, crop: np.ndarray, rot) -> OcrResult:
        """Baca crop dalam SATU arah (rot = None atau konstanta cv2.ROTATE_*)."""
        return self.read_once(crop if rot is None or crop is None or crop.size == 0 else cv2.rotate(crop, rot))

    def read_once(self, img: np.ndarray) -> OcrResult:
        """Satu kali baca tanpa mencoba arah lain (teks panjang yang arahnya sudah diketahui, mis. REF)."""
        t0 = time.perf_counter()
        if img is None or img.size == 0 or min(img.shape[:2]) < 4:
            return OcrResult()
        return OcrResult(tokens=self._run(img), ms=(time.perf_counter() - t0) * 1000.0)

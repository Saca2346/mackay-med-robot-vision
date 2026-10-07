"""
Barcode GS1 di kemasan alat medis (UDI): identitas produk yang PERSIS dan bisa diperiksa mesin.

Isi yang dipakai (GS1 Application Identifier):
  (01) GTIN-14  produk + model + ukuran; digit ke-14 = check digit (salah baca 1 digit selalu ketahuan)
  (17) YYMMDD   tanggal kedaluwarsa (DD = 00 berarti akhir bulan)
  (10) LOT      (21) nomor seri     (11) tanggal produksi
Ditemukan di foto dataset: Data Matrix (Boston Scientific) dan GS1-128 / Code 128 (Terumo).
Teks di bawah barcode ("(01)05413206252510(17)...") juga bisa dibaca OCR -> find_gtins_in_text().
"""
from __future__ import annotations

import calendar
import datetime as dt
import re
from dataclasses import dataclass

import numpy as np

_AI = re.compile(r"\((\d{2,4})\)([^()]*)")


@dataclass
class Gs1Code:
    gtin: str
    expiry: dt.date | None = None
    lot: str | None = None
    serial: str | None = None
    raw: str = ""
    fmt: str = ""
    source: str = "barcode"          # "barcode" (didekode) atau "ocr" (teks HRI dibaca OCR)
    poly: np.ndarray | None = None   # posisi di gambar yang didekode (4, 2)


def gtin_valid(gtin: str) -> bool:
    """GTIN-14 dengan check digit GS1 (bobot 3,1,3,... dari kiri untuk 13 digit pertama)."""
    if not re.fullmatch(r"\d{14}", gtin or ""):
        return False
    d = [int(c) for c in gtin]
    s = sum(v * (3 if i % 2 == 0 else 1) for i, v in enumerate(d[:13]))
    return (10 - s % 10) % 10 == d[13]


def gs1_date(yymmdd: str) -> dt.date | None:
    if not re.fullmatch(r"\d{6}", yymmdd or ""):
        return None
    y, m, d = 2000 + int(yymmdd[:2]), int(yymmdd[2:4]), int(yymmdd[4:])
    if not 1 <= m <= 12:
        return None
    if d == 0:
        d = calendar.monthrange(y, m)[1]
    try:
        return dt.date(y, m, d)
    except ValueError:
        return None


_FIXED = {"00": 18, "01": 14, "02": 14, "11": 6, "12": 6, "13": 6, "15": 6, "16": 6, "17": 6, "20": 2}


def _raw_fields(text: str) -> dict:
    """Element string GS1 TANPA tanda kurung, mis. isi DataMatrix '0108714729896845172904261039718940' atau dengan
    pemisah GS (\\x1d) setelah field panjang-variabel (10 = LOT, 21 = nomor seri, 240/241 = info tambahan)."""
    s, out, i = (text or "").strip().lstrip("\x1d").replace("]d2", "").replace("]C1", ""), {}, 0
    while i + 2 <= len(s):
        ai = s[i:i + 2]
        if ai in _FIXED:
            n = _FIXED[ai]
            if i + 2 + n > len(s):
                break
            out[ai], i = s[i + 2:i + 2 + n], i + 2 + n
        elif ai in ("10", "21", "22", "30", "37", "90") or s[i:i + 3] in ("240", "241", "250", "251"):
            k = 3 if s[i:i + 3] in ("240", "241", "250", "251") else 2
            end = s.find("\x1d", i + k)
            end = len(s) if end < 0 else end
            out[s[i:i + k]], i = s[i + k:end], end + 1
        else:
            break
    return out


def _digital_link_fields(text: str) -> dict:
    """QR 'GS1 Digital Link', mis. https://id.gs1.org/01/08714729896845/10/ABC123?17=290426"""
    out = {}
    for ai, v in re.findall(r"/(01|10|21|17)/([^/?#&]+)", text or ""):
        out[ai] = v
    for ai, v in re.findall(r"[?&](17|10|21)=([^&#]+)", text or ""):
        out.setdefault(ai, v)
    if "01" in out and len(out["01"]) == 13:           # GTIN-13 di URL -> GTIN-14
        out["01"] = "0" + out["01"]
    return out


def parse_gs1(text: str, fmt: str = "", source: str = "barcode") -> Gs1Code | None:
    """'(01)08714729896845(17)290426(10)39718940' (juga tanpa tanda kurung, atau QR GS1 Digital Link) -> Gs1Code,
    atau None kalau bukan GTIN yang sah."""
    fields = {ai: v.strip() for ai, v in _AI.findall(text or "")}
    if "01" not in fields:
        fields = _digital_link_fields(text) if "://" in (text or "") else _raw_fields(text)
    gtin = fields.get("01", "")
    if not gtin_valid(gtin):
        return None
    return Gs1Code(gtin=gtin, expiry=gs1_date(fields.get("17", "")), lot=fields.get("10") or None,
                   serial=fields.get("21") or None, raw=text, fmt=fmt, source=source)


def _variants(img: np.ndarray, max_side: int):
    """Versi gambar untuk percobaan ulang: dipertajam, lalu diperbesar 2x + dipertajam (garis barcode tipis dari
    webcam 1080p hanya 1-2 piksel per modul). -> [(gambar, skala terhadap asli)]."""
    import cv2
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharp = cv2.addWeighted(g, 1.8, cv2.GaussianBlur(g, (0, 0), 2.0), -0.8, 0)
    out = [(sharp, 1.0)]
    if max(g.shape[:2]) * 2 <= max_side:
        up = cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        out.append((cv2.addWeighted(up, 1.8, cv2.GaussianBlur(up, (0, 0), 3.0), -0.8, 0), 2.0))
    return out


def decode_gs1(img: np.ndarray, hard: bool = False, max_side: int = 2400) -> list[Gs1Code]:
    """Semua barcode GS1 dengan GTIN sah di gambar BGR / grayscale (satu GTIN dilaporkan sekali).
    hard=True: kalau tidak ada yang terbaca, coba lagi pada versi dipertajam / diperbesar 2x (hanya kalau sisi
    terpanjang hasil pembesaran <= max_side, supaya tetap cepat) dan dengan binarizer global."""
    try:
        import zxingcpp
    except ImportError:          # pembaca barcode tidak terpasang: verifikasi jalan tanpa barcode
        return []
    out: dict[str, Gs1Code] = {}

    def collect(results, scale):
        for r in results:
            code = parse_gs1(r.text, str(r.format).split(".")[-1])
            if code is None or code.gtin in out:
                continue
            p = r.position
            code.poly = np.array([[p.top_left.x, p.top_left.y], [p.top_right.x, p.top_right.y],
                                  [p.bottom_right.x, p.bottom_right.y], [p.bottom_left.x, p.bottom_left.y]],
                                 dtype=np.float32) / scale
            out[code.gtin] = code

    collect(zxingcpp.read_barcodes(img), 1.0)
    if hard and not out:
        for v, scale in _variants(img, max_side):
            collect(zxingcpp.read_barcodes(v), scale)
            if out:
                break
            collect(zxingcpp.read_barcodes(v, binarizer=zxingcpp.Binarizer.GlobalHistogram), scale)
            if out:
                break
    if hard and not out and max(img.shape[:2]) > 2000:
        import cv2                                     # foto ponsel 12 MP: coba versi diperkecil (dan diputar)
        for sc in (0.5, 0.35, 0.25):
            small = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
            collect(zxingcpp.read_barcodes(small), sc)
            for v, vs in _variants(small, 4000):       # terukur: Sapphire 3 hanya terbaca di skala 0,35 x2 tajam
                if out:
                    break
                collect(zxingcpp.read_barcodes(v), sc * vs)
                collect(zxingcpp.read_barcodes(v, binarizer=zxingcpp.Binarizer.GlobalHistogram), sc * vs)
            if out:
                break
    return list(out.values())


def find_gtins_in_text(text: str) -> list[Gs1Code]:
    """GTIN dari teks OCR, mis. '(01)05413206252510(17)270630(10)250725'. Hanya yang check digit-nya sah;
    O/I/l/S/B yang terbaca di posisi angka diperbaiki dulu."""
    fixed = re.sub(r"[\s]", "", text or "")
    out = {}
    for m in re.finditer(r"\(?0[1lI]\)?([0-9OoIlSB]{14})", fixed):
        digits = m.group(1).translate(str.maketrans("OoIlSB", "001158"))
        if not gtin_valid(digits) or digits in out:
            continue
        tail = fixed[m.end():m.end() + 40]
        exp = re.match(r"\(?1[7]\)?(\d{6})", tail)
        out[digits] = Gs1Code(gtin=digits, expiry=gs1_date(exp.group(1)) if exp else None,
                              raw=m.group(0) + tail, source="ocr")
    return list(out.values())

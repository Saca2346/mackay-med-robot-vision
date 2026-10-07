"""
Kode REF (nomor katalog) di punggung kemasan -> produk + diameter + panjang.

Sumber kedua yang independen dari angka ukuran besar: kodenya dicetak di baris lain, jadi salah baca
angka ukuran jarang terulang persis sama di REF. Pola di bawah HANYA yang sudah dicocokkan dengan
foto dataset (produk, ukuran, REF terbaca di kemasan yang sama):
  iVascular angiolite      SCCDSR14150 250 029  -> 2.50 x 29   (diameter x100, 3 digit; panjang 3 digit)
  iVascular xperience pro  BCPR14N150 250 015   -> 2.50 x 15
  iVascular essential pro  BCDPR14N150 300 040  -> 3.00 x 40
  Terumo accuforce         DC-RM 27 20 HHW      -> 2.75 x 20   (diameter 2 digit: 25 = 2.5, 27 = 2.75)
  Terumo ryurei            DC-RR 20 10 HHW      -> 2.0 x 10    (foto close-up 2026-09-28)
  Terumo ultimaster nagomi DE-RS 30 24 ASM      -> 3.00 x 24
Merek lain (Boston Scientific dll.) belum punya pola -> diverifikasi lewat barcode GS1.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_D = "[0-9OQDISLBZ]"                      # posisi angka: huruf yang sering tertukar dengan angka
_FIX = str.maketrans("OQDISLBZ", "00015182")


@dataclass
class RefHit:
    product: str
    diameter: float
    length: float
    text: str


def _x100(dd: str) -> float | None:
    # ukuran stent/balon selalu kelipatan 0,25 mm: "259" / "409" = satu angka salah baca, bukan produk lain
    # (rekaman 30-09: 4 dari ~1150 bacaan REF iVascular, semuanya jadi bantahan palsu)
    return int(dd) / 100.0 if int(dd) % 25 == 0 else None


def _terumo(dd: str) -> float | None:
    a, b = int(dd[0]), int(dd[1])
    if b in (0, 5):
        return a + b / 10
    if b in (2, 7):
        return a + b / 10 + 0.05
    return None


RULES = [
    ("angiolite", rf"[S5]C{{1,3}}D[S5]R{_D}{{2}}{_D}{{3}}({_D}{{3}})({_D}{{3}})", _x100),
    ("essential pro", rf"[B8]CDPR{_D}{{2}}N{_D}{{3}}({_D}{{3}})({_D}{{3}})", _x100),
    ("xperience pro", rf"[B8]CPR{_D}{{2}}N{_D}{{3}}({_D}{{3}})({_D}{{3}})", _x100),
    ("accuforce", rf"DC[-_.]?RM({_D}{{2}})({_D}{{2}})(?:HHW)?", _terumo),
    ("ryurei", rf"DC[-_.]?RR({_D}{{2}})({_D}{{2}})(?:HHW)?", _terumo),
    ("ultimaster nagomi", rf"DE[-_.]?R[S5]({_D}{{2}})({_D}{{2}})(?:A[S5]M)?", _terumo),
]


def parse_refs(text: str, diameter_range=(1.0, 10.0), length_range=(6.0, 200.0)) -> list[RefHit]:
    """Semua kode REF yang dikenali di teks OCR (spasi diabaikan)."""
    s = re.sub(r"\s+", "", (text or "").upper())
    out: dict[tuple, RefHit] = {}
    for product, pattern, dia in RULES:
        for m in re.finditer(pattern, s):
            dd, ll = (g.translate(_FIX) for g in m.groups())
            if not (dd.isdigit() and ll.isdigit()):
                continue
            d, length = dia(dd), float(int(ll))
            if d is None or not (diameter_range[0] <= d <= diameter_range[1]) \
                    or not (length_range[0] <= length <= length_range[1]):
                continue
            out.setdefault((product, d, length), RefHit(product, d, length, m.group(0)))
    return list(out.values())

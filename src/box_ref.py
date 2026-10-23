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
  Biosensors chroma        BCR-27 33            -> 2.75 x 33   (diameter = first 2 digits: 22 25 27 30 35 40)
  B. Braun sequent please  5023 2 4 3           -> 2.75 x 30   (50232 + length code + diameter code, see _BRAUN_*)
Merek lain (Boston Scientific dll.) belum punya pola -> diverifikasi lewat barcode GS1.
Terumo codes are also checked against the maker's size tables (VALID_SIZES): an impossible size is a misread.
Sources of the two added 15-10: B. Braun brochure "SeQuent Please NEO" (bbraun.com, 6050333, ordering table p. 7),
all 8 REFs on the task photo match it; Chroma: the 13 REFs readable on the task photo (BCR-4014..4029, 3514, 3014,
2736..2714, 2536, 2533) and the IFU excerpt (BCR-2209..2229 = 2.25 mm); lengths 9-36 mm.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_D = "[0-9OQDISLBZ]"                      # posisi angka: huruf yang sering tertukar dengan angka
_FIX = str.maketrans("OQDISLBZT", "000151827")
_DT = "[0-9OQDISLBZT]"                     # Terumo codes only: a 7 is also read as T (live webcam 23-10: "RM2T20" = 2.75 x 20)


@dataclass
class RefHit:
    product: str
    diameter: float
    length: float
    text: str
    alts: tuple = ()                      # parse_ref_hints only: the diameters this misread code can stand for


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


def _chroma(dd: str) -> float | None:
    return {"22": 2.25, "25": 2.5, "27": 2.75, "30": 3.0, "35": 3.5, "40": 4.0}.get(dd)


def _grid(rows: dict) -> set:
    return {(round(d, 2), float(n)) for d, lengths in rows.items() for n in lengths}


# Sizes each product line is made in, from the maker's own ordering table (terumo-europe.com product pages,
# "Specifications and order codes", read 17-10-2026). A REF that decodes to a size outside its table is a misread digit,
# not a product: it counts as "not read" instead of contradicting the size text (like _x100 for iVascular). On the D455
# shelf clip of 16-10 every wrong REF was such a size: Nagomi 3.0 read 1.0, Nagomi 33 read 31 / 13, Accuforce 3.5 x 15
# read 3.5 x 16, Accuforce 2.75 x 20 read 2.75 x 25 (a REF piece "DCRM272" glued to a "5" of the size text).
# A product without a table here is not checked.
_ACCU = (6, 8, 12, 15, 20)
_RYU = (10, 15, 20)
VALID_SIZES = {
    "accuforce": _grid({2.0: _ACCU, 2.25: _ACCU, 2.5: _ACCU + (25, 30), 2.75: _ACCU, 3.0: _ACCU + (25, 30),
                        3.25: _ACCU, 3.5: _ACCU + (25, 30), 3.75: _ACCU, 4.0: _ACCU, 4.5: (6, 8, 12, 15),
                        5.0: (6, 8, 12, 15)}),
    "ryurei": _grid({1.0: (5,), 1.25: (5, 10, 15, 20), 1.5: (5, 10, 15, 20), 2.0: _RYU + (30, 40), 2.25: _RYU,
                     2.5: _RYU + (30, 40), 2.75: _RYU, 3.0: _RYU + (30, 40), 3.25: _RYU, 3.5: _RYU + (30, 40),
                     3.75: _RYU, 4.0: _RYU}),
    "ultimaster nagomi": _grid({d: (9, 12, 15, 18, 21, 24, 28, 33, 38, 44, 50)
                                for d in (2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 4.5)}),
}


def size_exists(product: str, diameter: float, length: float) -> bool:
    """False only when the product has a size table and this size is not in it."""
    table = VALID_SIZES.get(product)
    return table is None or (round(diameter, 2), float(length)) in table


RULES = [
    # RealSense D455 08-10: "SCCDSR" terbaca SCCDSA / SCCD5A / SCCDSN / SCCOSR (R -> A/N, D -> O). Dengan R saja
    # REF kotak angiolite 2.5 x 29 terbaca 5 kali dalam 133 s -> 20; K10 (4 x 19) tetap terbaca 4 x 19, tidak tertukar.
    ("angiolite", rf"[S5]C{{1,3}}[DO0][S5][RAN]{_D}{{2}}{_D}{{3}}({_D}{{3}})({_D}{{3}})", _x100),
    ("essential pro", rf"[B8]CDPR{_D}{{2}}N{_D}{{3}}({_D}{{3}})({_D}{{3}})", _x100),
    ("xperience pro", rf"[B8]CPR{_D}{{2}}N{_D}{{3}}({_D}{{3}})({_D}{{3}})", _x100),
    # Terumo prefix "DC-" / "DE-" lost by the OCR ("26 RM3515HHW", "OC-RM2720HHW": 15 of 5,100 recorded readings, 20-10,
    # every one the real box; K21 never read its REF otherwise): accepted only with the full suffix, so a date or a
    # size glued after a bare "RM" can never be a code.
    ("accuforce", rf"(?:DC[-_.]?RM|RM(?={_DT}{{4}}HHW))({_DT}{{2}})({_DT}{{2}})(?:HHW)?", _terumo),
    ("ryurei", rf"(?:DC[-_.]?RR|RR(?={_DT}{{4}}HHW))({_DT}{{2}})({_DT}{{2}})(?:HHW)?", _terumo),
    ("ultimaster nagomi", rf"(?:DE[-_.]?R[S5]|R[S5](?={_D}{{4}}A[S5]M))({_D}{{2}})({_D}{{2}})(?:A[S5]M)?", _terumo),
]

# Codes matched on the text WITH its spaces and only as a whole code (no digit glued before or after): an OCR digit
# inserted into a short all-digit code gives another valid code. Task 1 photo, 15-10: the 2.75 x 30 box read
# "NEFE50232243" (= 3.0 x 20 without this rule) and the 3.5 x 30 box "NE502324246" (= 2.5 x 30).
# B. Braun SeQuent Please NEO REF = 50232 + length code + diameter code (official ordering table: 5023200 = 2.0 x 10
# ... 5023267 = 4.0 x 40; there is no diameter code 5).
_BRAUN_LEN = {"0": 10.0, "1": 15.0, "2": 20.0, "3": 25.0, "4": 30.0, "5": 35.0, "6": 40.0}
_BRAUN_DIA = {"0": 2.0, "1": 2.25, "2": 2.5, "3": 2.75, "4": 3.0, "6": 3.5, "7": 4.0}
CODED = [
    ("sequent please neo", rf"[5S][0OQD][2Z]3[2Z]({_D})({_D})",
     lambda lc, dc: (_BRAUN_DIA.get(dc), _BRAUN_LEN.get(lc))),
    ("chroma", rf"[B8]CR\s?[-_.]?\s?({_D}{{2}})({_D}{{2}})",
     lambda dd, ll: (_chroma(dd), float(int(ll)))),
]


# printed dates (2028-09-30) and their OCR pieces ("2028-0", "2028-09"): removed BEFORE the spaces are dropped, or
# "DC-RM" + "2028-09-30" becomes "DC-RM2028" = accuforce 2.0 x 28 (stress test 15-10 K07: false contradictions)
_DATE_PIECE = re.compile(r"(?<!\d)(?:19|20)\d\d\s*[-./]\s*\d{0,2}(?:\s*[-./]\s*\d{0,2})?")


def _clean(text: str) -> str:
    return re.sub(r"\s+", "", _DATE_PIECE.sub(" ", (text or "").upper()))


def parse_refs(text: str, diameter_range=(1.0, 10.0), length_range=(6.0, 200.0)) -> list[RefHit]:
    """Semua kode REF yang dikenali di teks OCR (spasi diabaikan)."""
    s = _clean(text)
    out: dict[tuple, RefHit] = {}
    for product, pattern, dia in RULES:
        for m in re.finditer(pattern, s):
            dd, ll = (g.translate(_FIX) for g in m.groups())
            if not (dd.isdigit() and ll.isdigit()):
                continue
            d, length = dia(dd), float(int(ll))
            if d is None or not (diameter_range[0] <= d <= diameter_range[1]) \
                    or not (length_range[0] <= length <= length_range[1]) or not size_exists(product, d, length):
                continue
            out.setdefault((product, d, length), RefHit(product, d, length, m.group(0)))
    spaced = _DATE_PIECE.sub(" ", (text or "").upper())
    for product, pattern, decode in CODED:
        for m in re.finditer(rf"(?<![0-9]){pattern}(?![0-9])", spaced):
            groups = [g.translate(_FIX) for g in m.groups()]
            if not all(g.isdigit() for g in groups):
                continue
            d, length = decode(*groups)
            if d is None or length is None or not (diameter_range[0] <= d <= diameter_range[1]) \
                    or not (length_range[0] <= length <= length_range[1]):
                continue
            out.setdefault((product, d, length), RefHit(product, d, length, m.group(0)))
    return list(out.values())


# Terumo diameter codes end in 0 / 2 / 5 / 7 (x.0, x.25, x.5, x.75). An 8 there is impossible and is the D455 OCR
# reading a 5 or a 0: stress test 15-10, box K07 (REF DC-RM2512HHW) was read DC-RM2812HHW 13 times and correctly once,
# so the PICK took 53.6 s. Such a code may SUPPORT a request for x.5 or x.0 with the same length, never contradict one;
# it is kept apart from parse_refs (inventory keys, scans) because its size is not known.
# Live webcam 23-10: the same K07 (2.5 x 12) was read DC-RM2612HHW in 4 of 13 readings (5 read as 6, correct once in 13), 7.4 s to green.
_TERUMO_AMBIGUOUS = {"8": (5, 0), "6": (5,)}


def parse_ref_hints(text: str, length_range=(6.0, 200.0)) -> list[RefHit]:
    """Terumo REF codes with an impossible diameter digit -> RefHit with `alts` (possible diameters)."""
    s = _clean(text)
    out: dict[tuple, RefHit] = {}
    for product, pattern, dia in RULES:
        if dia is not _terumo:
            continue
        for m in re.finditer(pattern, s):
            dd, ll = (g.translate(_FIX) for g in m.groups())
            if not (dd.isdigit() and ll.isdigit()) or dd[1] not in _TERUMO_AMBIGUOUS:
                continue
            length = float(int(ll))
            if not (length_range[0] <= length <= length_range[1]):
                continue
            alts = tuple(int(dd[0]) + b / 10 for b in _TERUMO_AMBIGUOUS[dd[1]]
                         if size_exists(product, int(dd[0]) + b / 10, length))
            if not alts:
                continue
            out.setdefault((product, alts, length), RefHit(product, alts[0], length, m.group(0), alts))
    return list(out.values())

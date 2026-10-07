"""
Dari bukti yang terbaca -> identitas kotak (produk, diameter, panjang) -> keputusan.

Keputusan per kotak (aturan keselamatan dari mentor: salah ukuran stent bisa fatal):
  MATCH     : cukup bukti independen (lihat decide_multi) bahwa produk DAN diameter DAN panjang
              SAMA PERSIS dengan permintaan, dan tidak ada bukti yang bertentangan
  CANDIDATE : baru satu sumber yang cocok (mis. hanya angka ukuran) -> dekatkan / baca barcode
  EXPIRED   : barcode menunjukkan tanggal kedaluwarsa sudah lewat -> tidak pernah diambil
  IGNORED   : ada bukti yakin yang BERBEDA dari permintaan (pasti bukan target)
  CONFIRM   : tidak ada yang bertentangan, tapi bukti belum cukup / sumber saling bertentangan
              -> sistem berhenti dan minta konfirmasi, TIDAK PERNAH menebak
Di video, keputusan baru dipercaya kalau konsisten di beberapa bacaan berturut-turut.
"""
from __future__ import annotations

import datetime as dt
import re
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from src.box_geometry import inside_fraction, poly_iom, poly_iou
from src.box_ocr import OcrToken

MATCH, IGNORED, CONFIRM, READING = "match", "ignored", "needs_confirmation", "reading"
STILL_JIT = 0.25         # gerak relatif terhadap rak (rata-rata, dalam lebar kotak) di bawah ini = kotak diam di rak
CANDIDATE, EXPIRED = "candidate", "expired"
EMPTY = ("empty",)        # key bacaan yang tidak memuat apa pun (tidak terbaca / terlalu jauh): tidak memutus
                          # rangkaian bacaan yang sepakat, tapi 3 kali berturut-turut -> CONFIRM


# ------------------------------------------------------------------ permintaan
@dataclass
class Request:
    product: str
    diameter: float | None = None          # None untuk barang tanpa ukuran (mis. "permanent sled bag")
    length: float | None = None

    def __str__(self) -> str:
        if self.diameter is None:
            return self.product
        return f"{self.product} {self.diameter:g} x {self.length:g} mm"


PRODUCTS = ["angiolite", "xperience pro", "essential pro", "accuforce", "ultimaster nagomi", "ryurei",
            "conqueror nc pro", "nc emerge", "agent", "sapphire", "scoreflex", "permanent sled bag"]
BRANDS = {"ivascular": {"angiolite", "xperience pro", "essential pro"},
          "terumo": {"accuforce", "ultimaster nagomi", "ryurei"},
          "boston scientific": {"agent", "nc emerge", "permanent sled bag"},
          "orbusneich": {"sapphire", "scoreflex"},
          "apt medical": {"conqueror nc pro"}}
NO_SIZE = {"permanent sled bag"}                 # barang tanpa diameter / panjang
ALIASES = {"nagomi": "ultimaster nagomi", "ultimaster": "ultimaster nagomi", "xperience": "xperience pro",
           "essential": "essential pro", "sled bag": "permanent sled bag", "sled": "permanent sled bag",
           "conqueror": "conqueror nc pro", "emerge": "nc emerge"}
_FILLER = {"pilih", "ambil", "ambilkan", "cari", "carikan", "tolong", "minta", "saya", "yang", "dengan", "dan", "kotak",
           "box", "stent", "balon", "balloon", "kateter", "catheter", "ukuran", "size", "mm", "milimeter", "merek",
           "merk", "brand", "produk", "untuk", "the", "please", "pick", "select", "get", "a", "of"}
_D_WORDS = r"(?:(?<![a-z])(?:diameter|diam|dia|d)|ø|⌀)"
_L_WORDS = r"(?:panjang|pjg|length|len|l|p)"


def brand_of(product: str) -> str:
    return next((b for b, ps in BRANDS.items() if product in ps), "")


def parse_request(s: str, catalog: list[str] | None = None) -> Request:
    """Permintaan dalam kalimat biasa -> Request. Contoh yang dipahami:
      "angiolite,2.5,29"   "angiolite 2.5x29"   "pilih ivascular angiolite diameter 4 mm panjang 19"
      "terumo accuforce 2,75 x 20"   "nagomi 3/24"   "angiolite Ø4 L19"   "permanent sled bag" (tanpa ukuran)
    Merek boleh ditulis, tapi harus cocok dengan produknya. Angka tanpa kata kunci: pertama diameter, kedua panjang."""
    names = list(dict.fromkeys([_norm_words(c) for c in (catalog or [])] + PRODUCTS))
    parts = [p.strip() for p in s.split(",")]
    if len(parts) == 3 and all(re.fullmatch(r"\d+(?:\.\d+)?", p) for p in parts[1:]):   # format lama
        text, dia, length = parts[0], float(parts[1]), float(parts[2])
    else:
        t = s.lower().replace("×", " x ").replace("⌀", " ø ")
        t = re.sub(r"(\d),(\d)", r"\1.\2", t)                                       # 2,75 -> 2.75
        dia = length = None
        m = re.search(_D_WORDS + r"\s*[:=]?\s*(\d+(?:\.\d+)?)", t)
        if m:
            dia, t = float(m.group(1)), t[:m.start()] + " " + t[m.end():]
        m = re.search(r"\b" + _L_WORDS + r"\s*[:=]?\s*(\d+(?:\.\d+)?)", t)
        if m:
            length, t = float(m.group(1)), t[:m.start()] + " " + t[m.end():]
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:mm)?\s*[x*/]\s*(\d+(?:\.\d+)?)", t)
        if m and dia is None and length is None:
            dia, length, t = float(m.group(1)), float(m.group(2)), t[:m.start()] + " " + t[m.end():]
        nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", t)]
        if dia is None and nums:
            dia = nums.pop(0)
        if length is None and nums:
            length = nums.pop(0)
        text = re.sub(r"\d+(?:\.\d+)?", " ", t)
    words = [w for w in _norm_words(text).split() if w not in _FILLER]
    brand = ""
    for b in sorted(BRANDS, key=len, reverse=True):                                 # nama merek dibuang dari teks
        joined = " ".join(words)
        if re.search(rf"\b{b}\b", joined):
            brand, words = b, re.sub(rf"\b{b}\b", " ", joined).split()
    phrase = " ".join(words)
    product, best = None, None
    for cand in names + list(ALIASES):
        target = cand.replace(" ", "")
        k = len(cand.split())
        for i in range(len(words)):
            for n in (k, k + 1):
                if i + n > len(words):
                    continue
                d = levenshtein("".join(words[i:i + n]), target)
                if d <= _tolerance(cand) and (best is None or (d, -len(target)) < best):
                    best, product = (d, -len(target)), ALIASES.get(cand, cand)
    if product is None:
        raise ValueError(f"Produk tidak dikenali dalam {s!r} (terbaca: {phrase!r}). Produk yang dikenal: "
                         + ", ".join(names))
    if brand and brand_of(product) and brand_of(product) != brand:
        raise ValueError(f"{product} bukan produk {brand} (produk {brand}: {', '.join(sorted(BRANDS[brand]))})")
    if product in NO_SIZE:
        return Request(product)
    if dia is None or length is None:
        raise ValueError(f"Ukuran {product} belum lengkap dalam {s!r}: tulis diameter DAN panjang, "
                         f"mis. \"{product} diameter 2.5 panjang 29\"")
    if not (1.0 <= dia <= 10.0) or not (6.0 <= length <= 200.0):
        raise ValueError(f"Ukuran tidak masuk akal: diameter {dia:g} mm, panjang {length:g} mm "
                         "(diameter 1-10 mm, panjang 6-200 mm; mungkin tertukar?)")
    return Request(product, dia, length)


def describe_request(req: Request) -> str:
    b = brand_of(req.product)
    name = f"{req.product}" + (f" ({b})" if b else "")
    return name if req.diameter is None else f"{name}, diameter {req.diameter:g} mm, panjang {req.length:g} mm"


# ------------------------------------------------------------------ parsing teks
@dataclass
class Identity:
    product: str | None = None
    product_ok: bool = False          # terbaca dengan confidence cukup
    diameter: float | None = None
    diameter_ok: bool = False
    length: float | None = None
    length_ok: bool = False
    notes: list[str] = field(default_factory=list)
    text: str = ""
    # diameter bulat tanpa "mm" di sebelahnya, langsung diikuti panjang (label iVascular "4" "19"): BUKAN diameter_ok.
    # Hanya boleh MENYEPAKATI diameter yang diminta (decide_multi / support), tidak pernah membantah: potongan "3,5"
    # yang terbaca "3" tidak boleh membuat kotak 3.5 dianggap bukan target.
    diameter_hint: float | None = None

    def key(self):
        return (self.product, self.diameter, self.length)


def _norm_words(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", s.lower()).split())


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _tolerance(name: str) -> int:
    n = len(name.replace(" ", ""))
    return 0 if n <= 4 else (1 if n <= 8 else 2)


# tanggal lengkap (pemisah SAMA di kedua posisi, jadi "2.75-20" = ukuran Terumo "⌀2.75 - 20" bukan tanggal)
# dan potongan tanggal hasil OCR ("026-06-15", "0-06-14", "028-09-3") yang kalau tidak dibuang jadi angka palsu
_DATE = re.compile(r"\b\d{4}([-/.])\d{1,2}\1\d{1,2}\b|\b\d{1,2}([-/.])\d{1,2}\2\d{2,4}\b|\d*-\d{1,2}-\d*"
                   r"|\b(?:19|20)\d{2}-\d{1,2}\b")
_NUM = re.compile(r"\d+(?:[.,]\d+)?")
_BAD_UNIT = re.compile(r"(atm|bar|kpa|psi|cm|fr|in|inch|\")", re.I)


def _find_product(tokens: list[OcrToken], catalog: list[str], min_conf: float):
    words: list[tuple[str, float]] = []
    for t in tokens:
        for w in re.sub(r"[^a-z]+", " ", t.text.lower()).split():
            words.append((w, t.conf))
    hits: dict[str, tuple[int, float]] = {}
    firsts = [n.split()[0] for n in catalog]
    for name in catalog:
        target = name.replace(" ", "")
        k = len(name.split())
        first = name.split()[0]
        if k > 1 and len(first) >= 7 and firsts.count(first) == 1:
            # nama berkata banyak ("conqueror nc pro"): kata pertama yang khas (>= 7 huruf, tidak dipakai produk lain)
            # sudah cukup - di punggung sempit sering hanya kata itu yang terbaca utuh ("CONQUEROF")
            for w, c in words:
                d = levenshtein(w, first)
                if d <= _tolerance(first) and all(levenshtein(w, o.replace(" ", "")) > d for o in catalog if o != name):
                    if name not in hits or d < hits[name][0]:
                        hits[name] = (d, c)
        for size in (k, k + 1):
            for i in range(0, max(0, len(words) - size + 1)):
                win = words[i:i + size]
                joined = "".join(w for w, _ in win)
                d = levenshtein(joined, target)
                if d <= _tolerance(name):
                    conf = min(c for _, c in win)
                    if name not in hits or d < hits[name][0]:
                        hits[name] = (d, conf)
    if not hits:
        return None, False, "product not read"
    best = min(d for d, _ in hits.values())
    top = [n for n, (d, _) in hits.items() if d == best]
    if len(top) > 1:
        return None, False, "two products read: " + ", ".join(sorted(top))
    name = top[0]
    return name, hits[name][1] >= min_conf, None


_DIA_SYMBOL = re.compile(r"^[8]\s*(?=\d[.,]\d)")          # "8 3.0" / "83.0" <- "⌀ 3.0" (simbol dibaca 8)


def _diameter_symbols(tokens: list[OcrToken]) -> list[OcrToken]:
    """Simbol diameter ⌀ sering terbaca OCR sebagai "8" tepat sebelum angka desimal ("8" "3.0" untuk "⌀ 3.0").
    "8" tunggal yang langsung diikuti angka desimal (x.x / x.xx) diganti penanda "⌀": angka palsu 8 tidak lagi
    membuat diameter ambigu, dan angka sesudahnya mendapat petunjuk diameter."""
    out = []
    for i, t in enumerate(tokens):
        s = t.text.strip()
        nxt = tokens[i + 1].text.strip() if i + 1 < len(tokens) else ""
        if s == "8" and re.fullmatch(r"\d[.,]\d{1,2}", nxt):
            out.append(OcrToken("⌀", t.conf))
            continue
        if _DIA_SYMBOL.match(s) and re.fullmatch(r"8\s*\d[.,]\d{1,2}", s):
            out.append(OcrToken("⌀ " + _DIA_SYMBOL.sub("", s), t.conf))
            continue
        out.append(t)
    return out


def _merge_split_decimals(tokens: list[OcrToken], min_conf: float) -> list[OcrToken]:
    """OCR kadang kehilangan titik desimal: "2" "75" "mm" dari "2.75 mm". Pola satu digit + dua digit
    (+ "mm") dijadikan SATU token "2.75" yang sengaja dibuat RAGU (conf di bawah min_conf): tidak dihitung
    sebagai bukti cocok, tapi juga tidak lagi menghasilkan bacaan salah "diameter 2 / panjang 75"."""
    out, i = [], 0
    while i < len(tokens):
        t = tokens[i]
        one = re.fullmatch(r"(\d)\s+(\d{2})\s*(mm)?", t.text.strip())
        if one:
            out.append(OcrToken(f"{one.group(1)}.{one.group(2)}", min(t.conf, min_conf) * 0.5))
            i += 1
            continue
        if (i + 1 < len(tokens) and re.fullmatch(r"\d", t.text.strip()) and re.fullmatch(r"\d{2}", tokens[i + 1].text.strip())
                and (i + 2 >= len(tokens) or tokens[i + 2].text.strip().lower().startswith("mm"))):
            out.append(OcrToken(f"{t.text.strip()}.{tokens[i + 1].text.strip()}", min(t.conf, min_conf) * 0.5))
            i += 2
            continue
        out.append(t)
        i += 1
    return out


def parse_identity(tokens: list[OcrToken], catalog: list[str], min_conf: float = 0.80,
                   diameter_range=(1.0, 10.0), length_range=(6.0, 200.0)) -> Identity:
    """
    Teks OCR (gabungan semua size_label milik satu box) -> produk, diameter, panjang.
    Aturan angka:
      - buang tanggal (2028-09-30), kode panjang (>= 5 digit: LOT, REF, barcode),
        dan angka bersatuan lain (atm, cm, Fr, inch)
      - diameter: 1-10 mm (angka desimal seperti 2.75 = kandidat kuat)
      - panjang : 6-200 mm, bilangan bulat
      - lebih dari satu kandidat berbeda -> ambigu -> tidak dipakai (jadi CONFIRM)
    """
    ident = Identity(text=" ".join(t.text for t in tokens))
    ident.product, ident.product_ok, note = _find_product(tokens, catalog, min_conf)
    if note:
        ident.notes.append(note)
    tokens = _merge_split_decimals(_diameter_symbols(tokens), min_conf)

    d_cand: dict[float, dict] = {}
    l_cand: dict[float, dict] = {}
    for i, t in enumerate(tokens):
        if len(re.split(r"\s*-\s*", t.text.strip())) >= 3:
            continue                                        # potongan tanggal / LOT rusak ("22-505-2022", "0-60-5702")
        raw = _DATE.sub(" ", t.text)                        # ("2.75-20" Terumo = 2 bagian, tetap dibaca)
        neighbours = " ".join(x.text for x in tokens[max(0, i - 1):i + 2]).lower()
        for m in _NUM.finditer(raw):
            s = m.group(0)
            if len(re.sub(r"\D", "", s)) >= 5:
                continue                                    # LOT / REF / barcode
            after = raw[m.end():m.end() + 4]
            if _BAD_UNIT.match(after.strip()):
                continue                                    # 12 atm, 140 cm, 6 Fr
            if re.match(r"[A-Za-z]", after) and not re.match(r"(m|rn|nm|x)", after, re.I):
                # awal kode huruf-angka, bukan ukuran: REF "SCCDSR1415..." terbaca "5CCD5A1415..." -> diameter palsu 5
                # (replay 30-09: 2 bacaan begini membatalkan AMBIL Angiolite 4x19). "4mm", "19rnm", "2.5x29" tetap.
                continue
            prefix = re.search(r"[A-Za-zØø⌀]*$", raw[:m.start()]).group(0).lower()
            if len(prefix) >= 2:
                continue                                    # bagian kode, mis. "SCC123", "REF45"
            v = float(s.replace(",", "."))
            is_dec = ("." in s or "," in s)
            if is_dec and len(re.split(r"[.,]", s)[1]) > 2:
                continue                                    # "2.512" = "2.5" + "12" menempel: ukuran maks 2 desimal
            if len(s) > 1 and s[0] == "0" and s[1].isdigit():
                continue                                    # "09", "06": bagian tanggal; ukuran tanpa nol di depan
            if v == 1 and not is_dec:
                continue                                    # "1" lepas = simbol ① (STENT/BALLOON), bukan ukuran
            if not is_dec and len(s) == 3 and v > length_range[1] and diameter_range[0] <= v / 100 <= diameter_range[1]:
                # "275" = "2.75" tanpa titik: kandidat diameter yang sengaja RAGU -> diameter lain jadi ambigu
                old = d_cand.get(v / 100)
                if old is None:
                    d_cand[v / 100] = {"conf": min(t.conf, min_conf) * 0.5, "dec": True, "hl": False, "hd": False}
                continue
            hint_l = prefix == "l" or "<l>" in neighbours or " l " in f" {neighbours} "
            hint_d = prefix in ("d", "ø", "⌀") or any(x in neighbours for x in ("ø", "⌀", "<d>")) \
                or " d " in f" {neighbours} "
            info = {"conf": t.conf, "dec": is_dec, "hl": hint_l, "hd": hint_d}
            # diameter bulat ("3", "4" iVascular) hanya kalau ada "mm" / tanda diameter di sebelahnya: angka lepas
            # seperti sisa "2,5" yang terpotong tidak boleh jadi diameter 2
            has_mm = re.search(r"(?<![a-z])mm\b", neighbours) is not None
            dia_ok = is_dec or hint_d or has_mm
            if diameter_range[0] <= v <= diameter_range[1] and dia_ok:
                old = d_cand.get(v)
                d_cand[v] = info if old is None or info["conf"] > old["conf"] else old
            if length_range[0] <= v <= length_range[1] and not is_dec:
                old = l_cand.get(v)
                l_cand[v] = info if old is None or info["conf"] > old["conf"] else old

    # angka yang bisa jadi diameter maupun panjang (6-10): putuskan dari petunjuk / sisa kandidat
    for v in sorted(set(d_cand) & set(l_cand)):
        info = d_cand[v]
        if info["hl"] and not info["hd"]:
            d_cand.pop(v)
        elif info["hd"] and not info["hl"]:
            l_cand.pop(v)
        elif len(set(l_cand) - {v}) >= 1 and len(set(d_cand) - {v}) == 0:
            l_cand.pop(v)
        elif len(set(d_cand) - {v}) >= 1 and len(set(l_cand) - {v}) == 0:
            d_cand.pop(v)
        elif len(set(d_cand) - {v}) == 0 and len(set(l_cand) - {v}) == 0:
            d_cand.pop(v)                                   # satu-satunya angka: tidak bisa diameter DAN panjang
            l_cand.pop(v)
            ident.notes.append(f"angka {v:g} bisa diameter atau panjang")

    if len(d_cand) == 1:
        (ident.diameter, info), = d_cand.items()
        ident.diameter_ok = info["conf"] >= min_conf
    elif len(d_cand) > 1:
        ident.notes.append("several diameters: " + ", ".join(f"{v:g}" for v in sorted(d_cand)))
    else:
        ident.notes.append("diameter not read")
        # label iVascular: "4" lalu "19" (satuan mm di baris lain). Audit rekaman tripod 30-09: "4"/"3" terbaca 1,00
        # tapi tidak pernah dipakai, jadi Angiolite 4x19 dan Essential Pro 3x40 hanya punya REF (1 bukti).
        for a, b in zip(tokens, tokens[1:]):
            # digit itu sendiri TIDAK BOLEH juga jadi kandidat panjang (mis. "6": diameter 2-6 dan panjang mulai
            # 6 mm tumpang tindih) - dua arti sekaligus = ambigu, jangan menebak salah satu (audit 01-10)
            if re.fullmatch(r"[2-6]", a.text.strip()) and a.conf >= min_conf and float(a.text) not in l_cand \
                    and re.fullmatch(r"\d{1,2}", b.text.strip()) and float(b.text.strip()) in l_cand:
                ident.diameter_hint = float(a.text)
                ident.notes.append(f"diameter {a.text} tanpa mm: hanya pendukung")
                break

    if len(l_cand) == 1:
        (ident.length, info), = l_cand.items()
        ident.length_ok = info["conf"] >= min_conf
    elif len(l_cand) > 1:
        ident.notes.append("several lengths: " + ", ".join(f"{v:g}" for v in sorted(l_cand)))
    else:
        ident.notes.append("length not read")
    return ident


# ------------------------------------------------------------------ keputusan satu bacaan
def _same(a, b) -> bool:
    return (a is None and b is None) or (a is not None and b is not None and abs(a - b) < 1e-6)


def decide(ident: Identity, req: Request) -> tuple[str, str]:
    """Satu bacaan OCR teks saja -> (keputusan, alasan singkat). Lihat decide_multi untuk verifikasi penuh."""
    if ident.product and ident.product_ok and ident.product != req.product:
        return IGNORED, f"product {ident.product}"
    if req.diameter is not None:
        if ident.diameter is not None and ident.diameter_ok and not _same(ident.diameter, req.diameter):
            return IGNORED, f"diameter {ident.diameter:g}"
        if ident.length is not None and ident.length_ok and not _same(ident.length, req.length):
            return IGNORED, f"length {ident.length:g}"
    if ident.product_ok and (req.diameter is None or (ident.diameter_ok and ident.length_ok)):
        return MATCH, "product + diameter + length"
    missing = [n for n, ok in (("product", ident.product_ok), ("diameter", ident.diameter_ok),
                               ("length", ident.length_ok)) if not ok]
    return CONFIRM, "unsure: " + ", ".join(missing)


# ------------------------------------------------------------------ verifikasi banyak sumber
W_BARCODE, W_GTIN_TEXT, W_TEXT, W_REF = 2, 1, 1, 1


@dataclass
class Evidence:
    """Semua yang terbaca dari SATU kotak dalam satu kali baca."""
    text: Identity | None = None                        # nama produk + angka ukuran (crop size_label)
    refs: list = field(default_factory=list)            # src.box_ref.RefHit
    codes: list = field(default_factory=list)           # src.box_barcode.Gs1Code di dalam kotak ini
    too_far: bool = False                               # kotak terlalu kecil di gambar untuk dibaca
    support: tuple | None = None                        # (atom bukti, bertentangan?) dari support(), diisi read_evidence
    spine: str = ""                                     # teks OCR seluruh punggung (hanya untuk diagnosa / log bacaan)


def _matches(product, diameter, length, req: Request) -> bool:
    if product != req.product:
        return False
    return req.diameter is None or (_same(diameter, req.diameter) and _same(length, req.length))


def decide_multi(ev: Evidence, req: Request, catalog=None, today: dt.date | None = None, min_score: int = 2,
                 trust_unverified: bool = False) -> tuple[str, str, tuple | None]:
    """
    Satu bacaan lengkap -> (keputusan, alasan, key). Sumber dan bobotnya:
      barcode GS1 -> GTIN di katalog terverifikasi          2   (check digit + koreksi galat bawaan)
      GTIN dari teks OCR di bawah barcode (check digit sah) 1
      nama produk + diameter + panjang dari teks ukuran     1
      kode REF yang memuat ukuran (src/box_ref.py)          1
    MATCH butuh skor >= min_score (default 2: barcode saja, atau dua bacaan teks independen) dan TIDAK
    ada sumber yang bertentangan. Kedaluwarsa (dari barcode) selalu menang: EXPIRED.
    """
    today = today or dt.date.today()
    if ev.too_far:
        return CONFIRM, "terlalu jauh - dekatkan kamera", EMPTY
    for c in ev.codes:
        if c.expiry is not None and c.expiry < today:
            return EXPIRED, f"kedaluwarsa {c.expiry.isoformat()} (GTIN {c.gtin})", ("expired", c.gtin)
    gtins = sorted({c.gtin for c in ev.codes})
    if len(gtins) > 1:
        return CONFIRM, "lebih dari satu GTIN di kotak ini: " + ", ".join(gtins), None
    agree, contra, notes, partial = [], [], [], False
    for c in ev.codes:
        e = catalog.get(c.gtin) if catalog is not None else None
        if e is None:
            notes.append(f"GTIN {c.gtin} belum ada di katalog")
            continue
        if not e.verified and not trust_unverified:
            notes.append(f"GTIN {c.gtin} ({e}) belum diverifikasi di katalog")
            continue
        src, w = ("barcode", W_BARCODE) if c.source == "barcode" else ("teks GTIN", W_GTIN_TEXT)
        (agree if _matches(e.product, e.diameter_mm, e.length_mm, req) else contra).append((src, w, str(e)))
    t = ev.text
    if t is not None:
        bad = []
        if t.product and t.product_ok and t.product != req.product:
            bad.append(f"produk {t.product}")
        if req.diameter is not None:
            if t.diameter is not None and t.diameter_ok and not _same(t.diameter, req.diameter):
                bad.append(f"diameter {t.diameter:g}")
            if t.length is not None and t.length_ok and not _same(t.length, req.length):
                bad.append(f"panjang {t.length:g}")
        d_ok = t.diameter_ok or (req.diameter is not None and t.diameter_hint is not None
                                 and _same(t.diameter_hint, req.diameter))       # petunjuk hanya menyepakati
        if bad:
            contra.append(("teks", W_TEXT, ", ".join(bad)))
        elif t.product_ok and (req.diameter is None or (d_ok and t.length_ok)):
            agree.append(("teks", W_TEXT, t.text[:40]))
        elif t.product_ok or t.diameter_ok or t.length_ok:
            partial = True
    for r in ev.refs:
        detail = f"{r.text} = {r.product} {r.diameter:g} x {r.length:g}"
        (agree if _matches(r.product, r.diameter, r.length, req) else contra).append(("REF", W_REF, detail))

    if agree and contra:
        return CONFIRM, ("bukti bertentangan: " + " | ".join(f"{s} {d}" for s, _, d in agree + contra)), None
    if contra:
        return IGNORED, "; ".join(f"{s}: {d}" for s, _, d in contra), None
    score = sum(w for _, w, _ in agree)
    key = (req.product, req.diameter, req.length, tuple(gtins))
    if score >= min_score:
        return MATCH, " + ".join(s for s, _, _ in agree), key
    if agree or partial:
        got = " + ".join(s for s, _, _ in agree) or "sebagian teks"
        return CANDIDATE, f"baru {got} cocok - perlu bukti kedua (dekatkan / tunjukkan barcode)", key
    if notes:
        return CONFIRM, "; ".join(notes), None
    return CONFIRM, "belum terbaca", EMPTY


COMBINE_WINDOW = 4       # bacaan berisi terakhir yang buktinya boleh digabung (TrackState.combined_match)


def support(ev: Evidence, req: Request, catalog=None, today: dt.date | None = None,
            trust_unverified: bool = False) -> tuple[frozenset, bool]:
    """Bukti SATU bacaan dipecah per sumber, untuk digabung antar bacaan kotak yang sama -> (atom yang cocok dengan
    permintaan, ada yang bertentangan?). Atom: ("bc", gtin) barcode, ("gt", gtin) teks GTIN, "ref", dan per bagian
    teks ukuran "tp" produk / "td" diameter / "tl" panjang. Bertentangan = aturan yang sama dengan decide_multi
    (produk / ukuran / REF / GTIN lain, >1 GTIN, kedaluwarsa)."""
    today = today or dt.date.today()
    if ev.too_far:
        return frozenset(), False
    atoms, contra = set(), False
    if len({c.gtin for c in ev.codes}) > 1 or any(c.expiry is not None and c.expiry < today for c in ev.codes):
        contra = True
    for c in ev.codes:
        e = catalog.get(c.gtin) if catalog is not None else None
        if e is None or (not e.verified and not trust_unverified):
            continue
        if _matches(e.product, e.diameter_mm, e.length_mm, req):
            atoms.add(("bc" if c.source == "barcode" else "gt", c.gtin))
        else:
            contra = True
    t = ev.text
    if t is not None:
        if t.product and t.product_ok:
            if t.product != req.product:
                contra = True
            else:
                atoms.add("tp")
                if req.diameter is None:                  # produk tanpa ukuran (mis. Sled Bag): nama saja = teks lengkap
                    atoms.update(("td", "tl"))
        if req.diameter is not None:
            for val, ok, atom, want in ((t.diameter, t.diameter_ok, "td", req.diameter),
                                        (t.length, t.length_ok, "tl", req.length)):
                if val is not None and ok:
                    if _same(val, want):
                        atoms.add(atom)
                    else:
                        contra = True
            if not t.diameter_ok and t.diameter_hint is not None and _same(t.diameter_hint, req.diameter):
                atoms.add("td")                            # petunjuk diameter: hanya menyepakati, tidak membantah
    for r in ev.refs:
        if _matches(r.product, r.diameter, r.length, req):
            atoms.add("ref")
        else:
            contra = True
    return frozenset(atoms), contra


def combined_score(atoms) -> tuple[int, set]:
    """Skor bukti gabungan (bobot sama dengan decide_multi) -> (skor, GTIN yang terlibat)."""
    gtins = {a[1] for a in atoms if isinstance(a, tuple)}
    score = (W_BARCODE if any(isinstance(a, tuple) and a[0] == "bc" for a in atoms) else 0) \
        + (W_GTIN_TEXT if any(isinstance(a, tuple) and a[0] == "gt" for a in atoms) else 0) \
        + (W_REF if "ref" in atoms else 0) + (W_TEXT if {"tp", "td", "tl"} <= atoms else 0)
    return score, gtins


# ------------------------------------------------------------------ konsistensi antar bacaan
@dataclass
class TrackState:
    history: deque = field(default_factory=lambda: deque(maxlen=5))   # (keputusan, key)
    support: deque = field(default_factory=lambda: deque(maxlen=5))   # (atom bukti, bertentangan?) sejajar history
    last_text: str = ""
    last_reason: str = ""
    next_ocr_frame: int = 0
    pending: bool = False              # crop box ini sedang dibaca OCR di thread latar belakang
    gtin: str = ""                     # dari barcode / teks GTIN terakhir (untuk FEFO dan log)
    lot: str = ""
    expiry: dt.date | None = None
    logged: str = ""                   # keadaan stabil terakhir yang sudah dicatat ke log
    stale: bool = False                # nomor kotak ini sempat diragukan (masuk layar lagi, pasangan lemah / longgar):
                                       # bacaan lama tidak dipakai sampai ada satu bacaan baru (lihat SimpleTracker)
    epoch: int = 0                     # naik setiap kali stale; bacaan dari crop yang diambil SEBELUMNYA tidak
                                       # menghapus stale (OCR di thread latar bisa selesai terlambat)
    tid: int = -1                      # nomor kotak (untuk log)
    hint: float = 0.0                  # kemiripan tampilan dengan kotak yang dicari di inventaris (0..1, urutan baca)
    hint_id: str = ""                  # kotak inventaris yang paling mirip
    hint_frame: int = -1000            # frame terakhir kemiripan dihitung
    reads: int = 0                     # jumlah bacaan yang pernah masuk lewat add() (penanda "bacaan sesudah saat X")
    evs: deque = field(default_factory=lambda: deque(maxlen=5))       # Evidence tiap bacaan, sejajar history

    def mark_stale(self) -> None:
        """Nomor kotak diragukan: SEMUA bacaan lama dibuang (bisa jadi milik kotak lain) dan bacaan dari crop yang
        diambil sebelum saat ini tidak dipakai (epoch naik)."""
        if self.history or self.pending:
            self.history.clear()
            self.support.clear()
            self.evs.clear()
            self.stale = True
            self.epoch += 1

    def add(self, decision, key, sup=None, ev=None) -> None:
        """Satu bacaan baru: keputusan + key, dan (kalau ada) bukti per sumber dari src.box_verifier.support dan
        Evidence-nya (supaya bisa diputuskan ulang untuk permintaan lain, lihat redecide)."""
        if self.evs.maxlen != self.history.maxlen or len(self.evs) != len(self.history):
            self.evs = deque([None] * len(self.history), maxlen=self.history.maxlen)
        self.history.append((decision, key))
        self.support.append(sup if sup is not None else (frozenset(), False))
        self.evs.append(ev)
        self.reads += 1

    def redecide(self, decide) -> bool:
        """Permintaan berganti: setiap bacaan kotak ini diputuskan ulang dari Evidence-nya untuk permintaan baru
        (`decide(ev)` -> (keputusan, alasan, key, support)). Bacaan yang Evidence-nya tidak tersimpan -> semua bacaan
        dibuang (dibaca ulang). -> True kalau bacaan dipertahankan. Dulu setiap permintaan baru membuang SEMUA bacaan
        rak dan mulai dari nol (+-10-20 s lagi per permintaan)."""
        evs = list(self.evs)
        self.logged, self.hint, self.hint_id, self.hint_frame = "", 0.0, "", -1000
        if not evs or len(evs) != len(self.history) or any(e is None for e in evs):
            self.history.clear()
            self.support.clear()
            self.evs.clear()
            return False
        out = [decide(e) for e in evs]
        self.history = deque(((d, k) for d, _, k, _ in out), maxlen=self.history.maxlen)
        self.support = deque((s for _, _, _, s in out), maxlen=self.history.maxlen)
        self.last_reason = out[-1][1]
        return True

    def combined_match(self, need: int, window: int | None = None) -> bool:
        """Bukti dari beberapa bacaan kotak yang SAMA (nomor tidak diragukan) digabung: MATCH kalau di `window`
        bacaan berisi terakhir TIDAK ada satu pun yang bertentangan / bukan target / kedaluwarsa, minimal `need`
        bacaan ikut menyumbang bukti yang cocok, satu GTIN saja, dan skor gabungannya >= 2 (mis. teks ukuran di
        satu bacaan + REF di bacaan lain). Kotak jauh sering terbaca sebagian per bacaan; tanpa digabung, MATCH
        menunggu dua bacaan yang masing-masing lengkap (terukur 2,5 menit untuk satu kotak)."""
        if self.stale or len(self.support) != len(self.history):
            return False
        rows = [(d, k, s) for (d, k), s in zip(self.history, self.support) if k != EMPTY][-(window or COMBINE_WINDOW):]
        if len(rows) < need:
            return False
        if any(c or d in (IGNORED, EXPIRED) for d, _, (_, c) in rows):
            return False
        contrib = [a for _, _, (a, _) in rows if a]
        if len(contrib) < need:
            return False
        score, gtins = combined_score(frozenset().union(*contrib))
        return score >= 2 and len(gtins) <= 1

    def combined_sources(self, window: int | None = None) -> str:
        """Sumber bukti di jendela gabungan, untuk log / layar: mis. "teks + REF"."""
        rows = [s for (d, k), s in zip(self.history, self.support) if k != EMPTY][-(window or COMBINE_WINDOW):]
        atoms = frozenset().union(*[a for a, _ in rows]) if rows else frozenset()
        names = [n for n, ok in (("barcode", any(isinstance(a, tuple) and a[0] == "bc" for a in atoms)),
                                 ("teks GTIN", any(isinstance(a, tuple) and a[0] == "gt" for a in atoms)),
                                 ("teks", {"tp", "td", "tl"} <= atoms), ("REF", "ref" in atoms)) if ok]
        return " + ".join(names)

    def stable(self, need: int) -> str:
        """MATCH/IGNORED/CANDIDATE kalau `need` bacaan BERISI terakhir sepakat; EXPIRED begitu terbaca.
        Bacaan kosong (EMPTY: tidak terbaca / terlalu jauh, mis. tangan menutupi) dilewati, tapi 3 kali berturut-turut
        -> CONFIRM. Bacaan berisi yang berbeda (kandidat, bertentangan, bukan target) tetap memutus kesepakatan."""
        if self.stale:
            return READING
        h = list(self.history)
        if any(d == EXPIRED for d, _ in h):
            return EXPIRED
        trailing = 0
        for _, k in reversed(h):
            if k != EMPTY:
                break
            trailing += 1
        info = [(d, k) for d, k in h if k != EMPTY]
        if trailing >= 3 or (not info and trailing >= need):
            return CONFIRM
        if len(info) < need:
            return READING
        last = info[-need:]
        decisions = {d for d, _ in last}
        if decisions == {MATCH} and len({k for _, k in last}) == 1:
            return MATCH
        if decisions == {IGNORED}:
            return IGNORED
        if self.combined_match(need):
            return MATCH
        if decisions <= {CANDIDATE, MATCH}:
            return CANDIDATE
        return CONFIRM


def pick_fefo(states: dict, need: int):
    """Dari kotak yang stabil MATCH, pilih yang kedaluwarsa paling dulu (FEFO); tanggal tidak diketahui
    dipilih paling akhir. -> track id atau None."""
    cands = [(tid, st) for tid, st in states.items() if st.stable(need) == MATCH]
    if not cands:
        return None
    return min(cands, key=lambda x: (x[1].expiry is None, x[1].expiry or dt.date.max, x[0]))[0]


SAME_COLUMN = 0.3       # geser ke samping maks (x lebar kotak) untuk "kotak yang sama" (SimpleTracker._same_column)
SAME_ALONG = 0.15       # geser memanjang maks (x panjang kotak)
SAME_NESTED = 0.85      # ... atau deteksi yang lebih pendek >= 85 % berada di dalam rentang yang lain (terpotong)
CALM_FRAC = 0.1         # kamera "tenang": gerak terukur < 0.1 x lebar kotak ...
CALM_FRAMES = 5         # ... sekian frame berturut-turut (tripod: 0,1-1,5 px, lebar kotak +-45 px)


def _spine_geom(poly):
    """-> (pusat, arah sisi panjang (vektor satuan), panjang, lebar) sebuah poligon 4 titik."""
    q = np.asarray(poly, dtype=np.float32).reshape(-1, 2)[:4]
    e1, e2 = q[1] - q[0], q[2] - q[1]
    n1, n2 = float(np.linalg.norm(e1)), float(np.linalg.norm(e2))
    L, W, ax = (n1, n2, e1) if n1 >= n2 else (n2, n1, e2)
    return q.mean(axis=0), ax / max(L, 1e-6), max(L, 1.0), max(W, 1.0)


def _shift(old, new) -> tuple[float, float]:
    """Pergeseran pusat deteksi baru terhadap posisi lama: (ke samping / lebar, memanjang / panjang)."""
    c, ax, L, W = _spine_geom(old)
    v = _spine_geom(new)[0] - c
    return abs(float(v @ np.array([-ax[1], ax[0]]))) / W, abs(float(v @ ax)) / L


def _offsets(old, new) -> str:
    """Untuk log: beda deteksi baru terhadap posisi lama sebuah kotak (samping x lebar, memanjang x panjang,
    sudut, rasio panjang) - apakah IoU turun karena sudut / panjang deteksi, atau kotaknya memang bergeser."""
    c, ax, L, W = _spine_geom(old)
    c2, ax2, L2, _ = _spine_geom(new)
    v = c2 - c
    lat, along = abs(float(v @ np.array([-ax[1], ax[0]]))) / W, abs(float(v @ ax)) / L
    ang = float(np.degrees(np.arccos(min(1.0, abs(float(ax @ ax2))))))
    return f"samping {lat:.2f}w memanjang {along:.2f}L sudut {ang:.1f} panjang x{L2 / L:.2f}"


class SimpleTracker:
    """Beri ID yang sama untuk kotak yang sama antar frame (cocokkan lewat IoU).
    motion (2x3 affine dari frame sebelumnya ke frame ini, lihat src/box_motion.py): posisi track lama digeser
    mengikuti gerakan kamera dulu, jadi kamera yang digerakkan tangan tidak membuat semua kotak dianggap baru
    (bacaan OCR yang sudah terkumpul untuk kotak itu tetap tersimpan).
    frame_size (w, h): kotak yang keluar layar (< 60 % luasnya di dalam frame) tidak dihitung "hilang"; posisinya
    terus digeser mengikuti kamera sampai `max_out` frame, jadi rak bisa diperiksa per bagian dari dekat. Saat
    masuk layar lagi, kotak itu hanya dicocokkan kalau pas (IoU >= reenter_iou) dan tidak ambigu.
    Nomor kotak yang pasangannya diragukan (masuk layar lagi, IoU < sure_iou, atau lewat aturan longgar) -> stale:
    bacaan lamanya tidak dipakai sampai dibaca ulang. Di rekaman w300 nomor kotak memang pernah pindah ke kotak
    sebelah; dengan aturan ini bacaan kotak lama tidak pernah dipakai untuk kotak sebelah itu.
    Kotak yang SEKOLOM (same_column, lihat _same_column): pusatnya tetap di tempat walau sudut / panjang deteksinya
    berubah -> pasti kotak yang sama, tidak diragukan. Hanya saat kamera hampir diam (tripod) dan tidak ambigu."""

    def __init__(self, iou_thr: float = 0.3, max_missed: int = 15, history: int = 5, iom_thr: float = 0.8,
                 max_out: int = 100, reenter_iou: float = 0.5, sure_iou: float = 0.5,
                 same_column: float = SAME_COLUMN):
        self.iou_thr, self.max_missed, self.history, self.iom_thr = iou_thr, max_missed, history, iom_thr
        self.max_out, self.reenter_iou, self.sure_iou = max_out, reenter_iou, sure_iou
        self.same_column = same_column
        self.calm = 0                                  # frame berturut-turut dengan kamera tenang
        self.by_column: list[tuple[int, int, float]] = []   # (nomor, deteksi, IoU) diputuskan aturan sekolom (log)
        self.residual: dict[int, np.ndarray] = {}
        self.doubted: list[tuple[int, str]] = []
        self.tracks: dict[int, dict] = {}
        self._next = 1

    def reset(self) -> None:
        """Lupakan semua kotak (posisi kamera tidak diketahui lagi). Nomor baru tidak pernah mengulang nomor lama."""
        self.tracks.clear()

    def _same_column(self, polys) -> set:
        """Pasangan (nomor, deteksi) yang PASTI kotak yang sama menurut letak pusatnya: bergeser ke samping
        <= same_column x lebar kotak dan memanjang <= SAME_ALONG x panjangnya, lebar mirip (bukan dua punggung
        tergabung), dan tidak ada deteksi / nomor lain dalam 0,6 x lebar (tidak ambigu).
        Punggung kotak sempit dan tinggi: sudut deteksi yang berubah 4-5 derajat saja (kotak miring di rak) membuat
        IoU < 0,5 walau kotaknya diam (uji tripod 30-09: 25 nomor diragukan dalam 1 menit, kamera geser 0,1-1,5 px,
        AMBIL baru 39,5 s). Terukur di rekaman w300 saat kamera diam: kotak yang sama bergeser ke samping <= 0,12 x
        lebar (p99), kotak sebelah >= 0,86 x lebar (p5).
        HANYA saat kamera tenang (tripod): gerak terukur < CALM_FRAC x lebar kotak selama CALM_FRAMES frame
        berturut-turut. Replay rekaman kamera dipegang: dengan syarat lama (gerak < 0,5 x lebar kotak itu) 3 nomor
        kotak pindah ke kotak sebelah tanpa diragukan - saat kamera diayun cepat gerak terukur bisa jauh lebih kecil
        dari gerak sebenarnya. Saat kamera bergerak aturan IoU biasa yang dipakai.
        Kotak yang TERBELAH (lengan melintang di depan rak: uji_01 30-09, tiap kotak jadi potongan atas dan bawah di
        kolom yang sama) = satu kotak: nomornya ikut potongan terpanjang; saat menyatu lagi deteksi utuh ikut nomor
        yang paling banyak bacaannya. -> (pasangan yang PASTI kotak yang sama, termasuk semua potongan; pasangan
        untuk mencocokkan nomor, paling banyak satu per nomor dan per deteksi)."""
        if not self.same_column or not polys or not self.tracks:
            return set(), set()
        # sekaligus untuk semua deteksi (numpy): +-500 pasangan per frame di rak 20 kotak
        corners = np.asarray([np.asarray(p, dtype=np.float32).reshape(-1, 2)[:4] for p in polys])   # (N, 4, 2)
        centers = corners.mean(axis=1)
        widths = np.asarray([_spine_geom(p)[3] for p in polys], dtype=np.float32)
        info, near_t, near_d = {}, {}, {}
        for tid, t in self.tracks.items():
            if t.get("was_out"):
                continue
            c, ax, L, W = _spine_geom(t["poly"])
            nrm = np.array([-ax[1], ax[0]], dtype=np.float32)
            v = centers - c
            lat, along = np.abs(v @ nrm) / W, np.abs(v @ ax) / L
            a = (np.asarray(t["poly"], dtype=np.float32).reshape(-1, 2)[:4] - c) @ ax        # rentang kotak lama
            b = (corners - c) @ ax                                                              # (N, 4)
            over = np.minimum(a.max(), b.max(axis=1)) - np.maximum(a.min(), b.min(axis=1))
            nest = np.maximum(over, 0.0) / np.maximum(np.minimum(np.ptp(a), np.ptp(b, axis=1)), 1.0)
            for i in np.flatnonzero((lat <= 0.6) & (nest > 0)):   # sekolom dan bertumpang sepanjang sumbu: pesaing
                i = int(i)
                near_t.setdefault(tid, []).append(i)
                near_d.setdefault(i, []).append(tid)
                # potongan / bagian kotak yang sama: satu kolom dan seluruhnya di dalam rentang yang lain
                piece = bool(lat[i] <= self.same_column and nest[i] >= SAME_NESTED)
                # pusat tetap, atau hanya terpotong / memanjang di sepanjang sumbunya (lengan menutupi bagian atas:
                # uji 30-09 5-11 kotak sekaligus, panjang x0,4 lalu x2,0, ke samping 0,00-0,08 x lebar)
                same = bool(lat[i] <= self.same_column and (along[i] <= SAME_ALONG or nest[i] >= SAME_NESTED)
                            and 0.6 <= widths[i] / W <= 1.6)
                # satu-satunya nomor dan deteksi di kolom ini: tumpang >= 50 % sepanjang sumbu sudah cukup (uji_01:
                # lengan membuat deteksi bergeser 0,28 x panjang, ke samping hanya 0,03 x lebar)
                alone = bool(lat[i] <= self.same_column and nest[i] >= 0.5 and 0.6 <= widths[i] / W <= 1.6)
                info[(tid, i)] = (same, piece, float(over[i]), alone)

        def rank(t):                                  # nomor "tertua": bacaan terbanyak, lalu nomor terkecil
            st = self.tracks[t]["state"]
            return (len(st.history), st.reads, -t)

        ok, assign = set(), set()
        for tid, dets in near_t.items():
            if len(dets) == 1 and len(near_d[dets[0]]) == 1 and info[(tid, dets[0])][3]:
                ok.add((tid, dets[0]))                          # satu-satunya di kolom ini
                assign.add((tid, dets[0]))
                continue
            if len(dets) == 1:
                i = dets[0]
            elif all(info[(tid, j)][1] for j in dets):              # terbelah: semua deteksi potongan kotak ini
                ok.update((tid, j) for j in dets if info[(tid, j)][0])
                i = max(dets, key=lambda j: info[(tid, j)][2])        # nomor ikut potongan terpanjang
            else:
                continue                                              # ambigu (kotak sebelah ikut di kolom ini)
            if not info[(tid, i)][0]:
                continue
            trks = near_d[i]
            if len(trks) == 1 or (all(info[(t2, i)][1] for t2 in trks) and tid == max(trks, key=rank)):
                ok.add((tid, i))
                assign.add((tid, i))
        return ok, assign

    def update(self, polys: list, motion=None, frame_size=None) -> list[int]:
        self.residual = {}                            # posisi deteksi - posisi perkiraan (px), per nomor kotak
        self.doubted = []                             # (nomor, alasan) nomor kotak yang diragukan di frame ini (log)
        moved = {}                                    # berapa px gerak kamera menggeser tiap kotak lama
        if motion is not None:
            A = np.asarray(motion, dtype=np.float32)
            for tid, t in self.tracks.items():
                p = np.asarray(t["poly"], dtype=np.float32).reshape(-1, 2)
                t["poly"] = p @ A[:, :2].T + A[:, 2]
                moved[tid] = float(np.linalg.norm(t["poly"].mean(axis=0) - p.mean(axis=0)))
        ids = [-1] * len(polys)
        weak = set()
        iou = {(tid, i): poly_iou(t["poly"], p) for tid, t in self.tracks.items() for i, p in enumerate(polys)}
        iom = {k: poly_iom(self.tracks[k[0]]["poly"], polys[k[1]]) for k, v in iou.items() if v > 0}
        # kotak yang baru masuk layar lagi: hanya pasangan yang pas dan tidak ambigu yang boleh dipakai
        for tid, t in self.tracks.items():
            if not t.get("was_out"):
                continue
            near = [(v, i) for (t2, i), v in iou.items() if t2 == tid and v >= 0.2]
            for v, i in near:
                ok = len(near) == 1 and v >= self.reenter_iou
                if not ok:
                    iou[(tid, i)] = 0.0
                    iom.pop((tid, i), None)
        # satu deteksi menutupi >= 2 kotak lama yang BERBEDA (tidak saling tumpang tindih: dua punggung terdeteksi
        # jadi satu) -> kotak baru, tidak mewarisi bacaan. Dua track di tempat yang sama (track aktif + track lama
        # yang belum dihapus) bukan penggabungan.
        merged = set()
        for i in range(len(polys)):
            inside = [tid for (tid, j), v in iom.items() if j == i and v >= self.iom_thr]
            # dua kotak BERSEBELAHAN; potongan atas dan bawah satu kotak (terbelah lengan) bukan penggabungan
            if any(poly_iou(self.tracks[a]["poly"], self.tracks[b]["poly"]) < self.iou_thr
                   and _shift(self.tracks[a]["poly"], self.tracks[b]["poly"])[0] > self.same_column
                   for k, a in enumerate(inside) for b in inside[k + 1:]):
                merged.add(i)
        widths = [_spine_geom(t["poly"])[3] for t in self.tracks.values()]
        calm = bool(widths) and (np.median(list(moved.values())) if moved else 0.0) < CALM_FRAC * np.median(widths)
        self.calm = self.calm + 1 if calm else 0
        ok, assign = self._same_column(polys) if self.calm >= CALM_FRAMES else (set(), set())
        # perkiraan posisi yang meleset bersama (gerak kamera salah ukur): pasangan yang pasti pun bergeser ke samping
        # -> aturan sekolom tidak dipakai di frame ini
        sure = [_shift(self.tracks[t]["poly"], polys[i])[0] for (t, i), v in iou.items() if v >= self.sure_iou]
        if ok and len(sure) >= 3 and float(np.median(sure)) > CALM_FRAC:
            ok, assign = set(), set()
        used_t, used_i = set(), set(merged)
        why = {}
        self.by_column = []
        # kotak yang sama menurut kolomnya dicocokkan lebih dulu: sudut / panjang deteksi berubah, terpotong, terbelah
        for tid, i in sorted(assign):
            if tid in used_t or i in used_i:
                continue
            ids[i] = tid
            used_t.add(tid)
            used_i.add(i)
            if iou.get((tid, i), 0.0) < self.sure_iou:
                self.by_column.append((tid, i, iou.get((tid, i), 0.0)))
        pairs = [(v, tid, i) for (tid, i), v in iou.items() if v >= self.iou_thr and i not in merged]
        for v, tid, i in sorted(pairs, reverse=True):
            if tid in used_t or i in used_i:
                continue
            ids[i] = tid
            used_t.add(tid)
            used_i.add(i)
            if v < self.sure_iou:
                if (tid, i) in ok:
                    self.by_column.append((tid, i, v))
                else:
                    weak.add(tid)                    # pasangan lemah: bisa jadi kotak sebelah
                    why[tid] = f"pasangan lemah IoU {v:.2f} ({_offsets(self.tracks[tid]['poly'], polys[i])})"
        # cadangan: kotak yang sama terdeteksi terpotong / lebih panjang (IoU turun, tapi satu hampir di dalam yang
        # lain). Hanya kalau pasangannya TIDAK ambigu, supaya bacaan tidak pernah pindah ke kotak sebelah.
        loose = [(tid, i) for (tid, i), v in iom.items()
                 if v >= self.iom_thr and tid not in used_t and i not in used_i]
        for tid, i in loose:
            if sum(1 for t2, _ in loose if t2 == tid) == 1 and sum(1 for _, i2 in loose if i2 == i) == 1:
                ids[i] = tid
                used_t.add(tid)
                used_i.add(i)
                weak.add(tid)
                why[tid] = f"cocok longgar IoM {iom[(tid, i)]:.2f} ({_offsets(self.tracks[tid]['poly'], polys[i])})"
        for i, p in enumerate(polys):
            if ids[i] == -1:
                ids[i] = self._next
                self.tracks[self._next] = {"poly": p, "missed": 0, "out": 0, "still": 0, "jit": 1.0,
                                           "state": TrackState(history=deque(maxlen=self.history), tid=self._next)}
                self._next += 1
                continue
            t = self.tracks[ids[i]]
            q = np.asarray(p, dtype=np.float32).reshape(-1, 2)
            self.residual[ids[i]] = q.mean(axis=0) - np.asarray(t["poly"], dtype=np.float32).reshape(-1, 2).mean(axis=0)
            t["poly"], t["missed"] = p, 0
            if t.pop("was_out", False) or ids[i] in weak:
                t["out"] = 0
                if t["state"].history or t["state"].pending:
                    self.doubted.append((ids[i], why.get(ids[i], "masuk layar lagi")))
                t["state"].mark_stale()              # nomor kotak diragukan: bacaan lama tidak dipakai lagi
        self._update_still()
        for tid in list(self.tracks):
            if tid in ids:
                continue
            t = self.tracks[tid]
            if frame_size is not None and inside_fraction(t["poly"], frame_size) < 0.6:
                t["was_out"] = True                  # di luar layar: bukan "hilang", posisinya terus diikuti
                t["out"] = t.get("out", 0) + 1
                if t["out"] > self.max_out:
                    del self.tracks[tid]
            else:
                t["missed"] += 1
                if t["missed"] > self.max_missed:
                    del self.tracks[tid]
        return ids

    def _update_still(self) -> None:
        """Kotak di RAK diam terhadap kotak-kotak lain; kotak yang dipegang tangan / ditarik tidak. Selisih posisi
        (deteksi - perkiraan) dikurangi median semua kotak (= galat gerak kamera bersama), dirata-rata (EMA) dan
        dibanding lebar kotak. `still` = berapa frame berturut-turut kotak itu diam terhadap rak."""
        if len(self.residual) < 3:
            return
        shared = np.median(np.asarray(list(self.residual.values())), axis=0)
        for tid, r in self.residual.items():
            t = self.tracks[tid]
            q = np.asarray(t["poly"], dtype=np.float32).reshape(-1, 2)
            short = max(min(float(np.linalg.norm(q[1] - q[0])), float(np.linalg.norm(q[2] - q[1]))), 1.0)
            t["jit"] = 0.6 * t.get("jit", 1.0) + 0.4 * float(np.hypot(*(r - shared))) / short
            t["still"] = t.get("still", 0) + 1 if t["jit"] < STILL_JIT else 0

    def visible(self, frame_size) -> list[int]:
        """Track yang posisinya (setelah digeser) ada di dalam layar."""
        return [tid for tid, t in self.tracks.items() if inside_fraction(t["poly"], frame_size) >= 0.6]

    def state(self, tid: int) -> TrackState:
        return self.tracks[tid]["state"]

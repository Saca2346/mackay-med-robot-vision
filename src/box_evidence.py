"""
Kumpulkan bukti identitas SATU kotak (dipakai webcam, evaluasi foto, dan pendaftaran):
  1. barcode GS1 di crop punggung kotak (src/box_barcode.py)
  2. OCR crop size_label -> nama produk + diameter + panjang, kode REF, teks GTIN
  3. kalau bukti belum cukup dan kotak cukup besar di gambar: OCR seluruh punggung (dipotong-potong
     memanjang) untuk kode REF dan teks GTIN yang ada di luar size_label (mis. REF Terumo di dekat barcode)
Pemotongan crop dilakukan di thread utama (frame berikutnya menimpa gambar), pembacaannya bisa di thread
latar (OcrWorker di scripts/test_webcam_box.py).
"""
from __future__ import annotations

import re
import time

import cv2
import numpy as np

from src.box_barcode import decode_gs1, find_gtins_in_text
from src.box_geometry import to_poly, warp_upright
from src.box_ref import parse_refs
from src.box_verifier import (CANDIDATE, CONFIRM, Evidence, Identity, _find_product, decide_multi, parse_identity,
                              support)

OCR_MAX_SIDE = 160     # sisi pendek crop untuk OCR paling besar segini (foto 12 MP -> jauh lebih cepat)


def box_width_px(poly) -> float:
    """Sisi pendek kotak di gambar (piksel) = lebar punggung kemasan."""
    (_, _), (w, h), _ = cv2.minAreaRect(to_poly(poly))
    return float(min(w, h))


def _cap(img: np.ndarray, short: int = OCR_MAX_SIDE) -> np.ndarray:
    s = short / min(img.shape[:2])
    return img if s >= 1 else cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def prepare_crops(frame: np.ndarray, group, cfg: dict):
    """-> (crop size_label, crop punggung, terlalu_jauh). Crop punggung dibuat memanjang mendatar.
    Barcode didekode dari crop punggung resolusi penuh (garis barcode tipis), OCR dari versi yang dikecilkan."""
    v = cfg.get("verify", {})
    if box_width_px(group.box.poly) < v.get("min_box_width_px", 24):
        return [], None, True
    sl = [_cap(warp_upright(frame, lab.poly, min_side=cfg["ocr"]["min_crop_height"])) for lab in group.labels]
    box = warp_upright(frame, group.box.poly, min_side=16, pad=0.02)
    if box.shape[0] > box.shape[1]:
        box = cv2.rotate(box, cv2.ROTATE_90_CLOCKWISE)
    return sl, box, False


def _tiles(img: np.ndarray, ratio: float = 5.0, overlap: float = 0.25):
    """Potong crop punggung yang sangat panjang menjadi beberapa bagian (OCR mengecilkan gambar panjang)."""
    h, w = img.shape[:2]
    L = int(ratio * h)
    if w <= L:
        return [img]
    step = max(1, int(L * (1 - overlap)))
    return [img[:, x:x + L] for x in list(range(0, w - L, step)) + [w - L]]


def _merge_codes(ev: Evidence, found) -> None:
    have = {c.gtin for c in ev.codes}
    ev.codes += [c for c in found if c.gtin not in have]


_ALONG, _ACROSS = "along", "across"


def _rot(crop, which):
    """Arah baca: 'along' = teks yang berjalan sepanjang punggung dibuat mendatar, 'across' = sebaliknya."""
    tall = crop.shape[0] > crop.shape[1]
    if which == _ALONG:
        return cv2.ROTATE_90_COUNTERCLOCKWISE if tall else None
    return None if tall else cv2.ROTATE_90_COUNTERCLOCKWISE


# arah yang sama ditambah 180 derajat (kotak ditaruh terbalik di rak)
_FLIP = {None: cv2.ROTATE_180, cv2.ROTATE_180: None,
         cv2.ROTATE_90_COUNTERCLOCKWISE: cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_CLOCKWISE: cv2.ROTATE_90_COUNTERCLOCKWISE}


def _num_score(tokens) -> float:
    """Seberapa utuh angka terbaca di satu arah: '2.75', '33', '20' (>= 2 karakter) + token 'mm'.
    Arah yang salah memecah angka jadi '2', '75', '3' sehingga skornya rendah."""
    return sum(t.conf * len(t.text) for t in tokens if re.fullmatch(r"\d+(?:[.,]\d+)?", t.text) and len(t.text) >= 2) \
        + sum(0.5 for t in tokens if t.text.lower().startswith("mm"))


def _complete(i) -> bool:
    return i.product_ok and i.diameter_ok and i.length_ok


def _combine(a, b, ta, tb, min_conf: float = 0.0):
    """Produk dari arah yang mengenali produk, angka dari arah yang angkanya paling utuh (tidak dicampur).
    Petunjuk diameter LINTAS ARAH: diameter bulat iVascular ("4") dan panjangnya ("19") kadang terbaca RapidOCR
    di arah yang BERBEDA dari crop yang sama (replay 01-10, kardus: "4" selalu di arah punggung, "19" hanya di
    arah melintang) -> petunjuk satu-arah biasa (parse_identity) tidak pernah menyala karena keduanya tak pernah
    bersebelahan dalam satu bacaan, skor bukti macet di REF saja. Di sini digit [2-6] yang berdiri sendiri dan
    yakin (conf >= min_conf) di SALAH SATU arah dipasangkan dengan panjang yang SUDAH terbaca yakin (length_ok)
    di arah mana pun, HANYA kalau digit semacam itu satu-satunya kandidat di kedua arah (dua digit berbeda,
    mis. "4" dan "6" sekaligus, = ambigu, tidak dipakai, sama seperti petunjuk satu-arah). Hanya MENYEPAKATI,
    tidak pernah membantah (sama seperti Identity.diameter_hint biasa)."""
    out = Identity(text=(a.text + " | " + b.text).strip(" |"), notes=a.notes + b.notes)
    if a.product_ok and b.product_ok and a.product != b.product:
        out.notes.append(f"dua produk terbaca: {a.product}, {b.product}")
    elif a.product_ok or b.product_ok:
        src = a if a.product_ok else b
        out.product, out.product_ok = src.product, True
    else:
        out.product = a.product or b.product
    num = a if _num_score(ta) >= _num_score(tb) else b
    out.diameter, out.diameter_ok, out.length, out.length_ok = num.diameter, num.diameter_ok, num.length, num.length_ok
    out.diameter_hint = num.diameter_hint
    if out.diameter is None and out.diameter_hint is None and out.length_ok:
        lone = {t.text.strip() for t in ta + tb if re.fullmatch(r"[2-6]", t.text.strip()) and t.conf >= min_conf}
        if len(lone) == 1:
            out.diameter_hint = float(lone.pop())
            out.notes.append(f"diameter {out.diameter_hint:g} lintas arah: hanya pendukung")
    return out


UPSIDE_RATIO = 1.2     # arah terbalik membaca teks sekian kali lebih jelas -> kotak dianggap terbalik


def _quality(tokens) -> float:
    """Seberapa jelas teks terbaca dalam satu arah (jumlah karakter x confidence). Arah yang salah membaca kotak
    terbalik sebagai potongan pendek beryakin rendah ("nglie", "62", "55")."""
    return sum(t.conf * len(t.text) for t in tokens)


def _worth_flip(ident, tokens, req) -> bool:
    """Arah terbalik dicoba? Tanpa permintaan (pemindaian rak, tidak dikejar waktu): selalu. Dengan permintaan: hanya
    kotak yang nama produk atau kode REF-nya sudah cocok dengan produk yang diminta - replay tripod: mencoba semua
    kotak yang terbaca sebagian membuat OCR 362 -> 834 ms per kotak."""
    if req is None:
        return True
    if ident.product_ok and ident.product == req.product:
        return True
    refs = parse_refs(" ".join(t.text for t in tokens))
    if any(r.product == req.product for r in refs):
        return True
    # hanya angka, tanpa produk / REF (+-5 % bacaan): bisa angka kotak yang diminta tapi terbalik ("2,5" -> "25"
    # dianggap panjang 25 mm = bantahan; replay tripod: ditambah satu salah baca REF -> BATAL AMBIL)
    return not ident.product_ok and not refs and (ident.diameter_ok or ident.length_ok)


def gather(sl_crops, box_crop, reader, cfg: dict, req=None) -> tuple[Evidence, int]:
    """Barcode di crop punggung + OCR crop size_label -> (Evidence, jumlah pembacaan OCR).
    Teks dibaca dulu dalam arah punggung (nama produk & ukuran kebanyakan merek); kalau belum lengkap, juga
    melintang (angka Ultimaster Nagomi dicetak melintang). Dengan `req`: nama produk LAIN sudah terbaca yakin di
    arah punggung -> arah melintang dilewati (kotak itu sudah pasti bukan target; hemat ~0,1 s CPU per kotak)."""
    codes = decode_gs1(box_crop, hard=True) if box_crop is not None else []
    min_conf = cfg["ocr"]["min_confidence"]
    parse = lambda toks: parse_identity(toks, cfg["catalog"], min_conf,
                                        tuple(cfg["matching"]["diameter_range_mm"]),
                                        tuple(cfg["matching"]["length_range_mm"]))
    ta = [t for c in sl_crops for t in reader.read_orient(c, _rot(c, _ALONG)).tokens]
    ident, tokens, n = parse(ta), list(ta), len(sl_crops)
    other = req is not None and ident.product_ok and ident.product and ident.product != req.product
    tb = []
    if sl_crops and not _complete(ident) and not other:
        tb = [t for c in sl_crops for t in reader.read_orient(c, _rot(c, _ACROSS)).tokens]
        n += len(sl_crops)
        if tb:
            ident = _combine(ident, parse(tb), ta, tb, min_conf) if ta else parse(tb)
            tokens += tb
    if sl_crops and tokens and not _complete(ident) and not other and _worth_flip(ident, tokens, req):
        # kotak yang ditaruh TERBALIK (180 derajat): classifier arah RapidOCR membalik baris panjang (REF, tanggal)
        # tapi tidak angka pendek. Uji tripod 30-09: "2,5" dan "29" terbaca "25 29" / "62 55"; diputar 180 derajat
        # terbaca "angiolite 2.5 29" (0,99-1,00). Hanya kalau arah biasa sudah membaca sesuatu tapi belum lengkap:
        # crop yang tidak terbaca sama sekali tidak dicoba lagi (hemat waktu OCR).
        for which, normal in ((_ALONG, ta), (_ACROSS, tb)):
            tf = [t for c in sl_crops for t in reader.read_orient(c, _FLIP[_rot(c, which)]).tokens]
            n += len(sl_crops)
            if not tf:
                continue
            pf = parse(tf)
            # arah terbalik membaca teks jauh lebih jelas = kotaknya memang terbalik: angka arah biasa adalah angka
            # terbalik ("2,5" -> "25" dianggap panjang 25 mm, membantah 29) dan dibuang, angka dari arah terbalik
            upside = which == _ALONG and _quality(tf) > UPSIDE_RATIO * _quality(normal)
            if _complete(pf) or upside:
                ident = _combine(pf, ident, tf, [], min_conf)  # angka dari arah terbalik; produk tetap dicek kedua arah
            elif pf.product_ok and not ident.product_ok:
                ident = _combine(ident, pf, tokens, tf, min_conf)
            else:
                continue
            tokens += tf
            if _complete(ident):
                break
    text = " ".join(t.text for t in tokens)
    ev = Evidence(text=ident if tokens else None, refs=parse_refs(text), codes=list(codes))
    _merge_codes(ev, find_gtins_in_text(text))
    return ev, n


def gather_full_box(ev: Evidence, box_crop, reader, cfg: dict | None = None) -> tuple[int, str]:
    """Tambahkan kode REF / teks GTIN dari OCR seluruh punggung, dan nama produk kalau crop size_label tidak
    memuatnya (nama Accuforce / angiolite sering di luar size_label). Nama di punggung yang BERBEDA dari nama di
    size_label membuat produk teks ambigu (tidak dipakai). -> (jumlah crop OCR, teks)."""
    more, n = [], 0
    for tile in _tiles(_cap(box_crop)):
        more += reader.read_once(tile).tokens
        n += 1
    extra = " ".join(t.text for t in more)
    ev.spine = extra
    known = {(h.product, h.diameter, h.length) for h in ev.refs}
    ev.refs += [h for h in parse_refs(extra) if (h.product, h.diameter, h.length) not in known]
    _merge_codes(ev, find_gtins_in_text(extra))
    if cfg is not None and more:
        prod, ok, _ = _find_product(more, cfg["catalog"], cfg["ocr"]["min_confidence"])
        if prod and ok:
            if ev.text is None:
                ev.text = Identity(product=prod, product_ok=True, text=extra, notes=["produk dari punggung"])
            elif not ev.text.product_ok:
                ev.text.product, ev.text.product_ok = prod, True
                ev.text.notes.append("produk dari punggung")
            elif ev.text.product != prod:
                ev.text.notes.append(f"dua produk terbaca: {ev.text.product}, {prod}")
                ev.text.product, ev.text.product_ok = None, False
    return n, extra


def read_any(sl_crops, box_crop, too_far, reader, cfg: dict) -> Evidence:
    """Baca SEMUA bukti satu kotak tanpa permintaan (pindai inventaris rak, src/box_inventory.py): barcode, teks
    size_label dua arah, dan seluruh punggung kalau REF / teks ukuran belum lengkap. Lebih lambat daripada
    read_evidence, tapi pindai rak dilakukan sebelum operasi (tanpa tekanan waktu)."""
    if too_far:
        return Evidence(too_far=True)
    ev, _ = gather(sl_crops, box_crop, reader, cfg)
    t = ev.text
    complete = t is not None and t.product_ok and t.diameter_ok and t.length_ok
    near = box_crop is not None and min(box_crop.shape[:2]) >= cfg.get("verify", {}).get("full_box_min_width_px", 40)
    if near and (not complete or not ev.refs) and not ev.codes:
        gather_full_box(ev, box_crop, reader, cfg)
    return ev


def redecide(ev, req, cfg: dict, catalog=None):
    """Bukti yang SUDAH dibaca diputuskan untuk permintaan (lain): -> (keputusan, alasan, key, support). Bacaan untuk
    permintaan lama bisa kurang lengkap untuk permintaan baru (arah melintang dilewati untuk produk lain), jadi paling
    buruk hasilnya belum MATCH dan kotak itu dibaca lagi - tidak pernah MATCH yang tidak didukung buktinya."""
    v = cfg.get("verify", {})
    trust = v.get("trust_unverified_catalog", False)
    d, r, k = decide_multi(ev, req, catalog=catalog, min_score=v.get("min_score", 2), trust_unverified=trust)
    return d, r, k, support(ev, req, catalog, trust_unverified=trust)


def read_evidence(sl_crops, box_crop, too_far, reader, cfg: dict, req, catalog=None):
    """-> (Evidence, keputusan, alasan, key, ms, jumlah_crop_OCR)."""
    v = cfg.get("verify", {})
    t0 = time.perf_counter()
    kw = dict(catalog=catalog, min_score=v.get("min_score", 2), trust_unverified=v.get("trust_unverified_catalog", False))
    if too_far:
        ev = Evidence(too_far=True)
        d, r, k = decide_multi(ev, req, **kw)
        ev.support = support(ev, req, catalog, trust_unverified=kw["trust_unverified"])
        return ev, d, r, k, 0.0, 0
    ev, n = gather(sl_crops, box_crop, reader, cfg, req)
    d, r, k = decide_multi(ev, req, **kw)
    near = box_crop is not None and min(box_crop.shape[:2]) >= v.get("full_box_min_width_px", 40)
    if d in (CANDIDATE, CONFIRM) and v.get("ocr_full_box", True) and near:
        n += gather_full_box(ev, box_crop, reader, cfg)[0]
        d, r, k = decide_multi(ev, req, **kw)
    ev.support = support(ev, req, catalog, trust_unverified=kw["trust_unverified"])   # untuk bukti gabungan
    return ev, d, r, k, (time.perf_counter() - t0) * 1000.0, n

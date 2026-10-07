"""
Pindai SATU kotak dari dekat - depan, belakang, atau punggung (tombol b di scripts/test_webcam_box.py).

Membaca SEMUA kode di gambar (barcode 1D GS1-128 Terumo, DataMatrix / QR Boston Scientific, QR non-GS1 seperti
alamat web) dan SELURUH teks (OCR petak-petak resolusi penuh, tegak dan diputar 90 derajat), lalu menyusun kartu data:
  kode      : jenis, GTIN (+ isi katalog), kedaluwarsa, LOT, nomor seri, atau teks mentah
  REF       : kode katalog pabrik -> produk + ukuran (iVascular, Terumo)
  teks      : nama produk, diameter, panjang yang terbaca
  keputusan : dibandingkan dengan permintaan (decide_multi: MATCH hanya dengan >= 2 bukti independen, tanpa
              bukti yang bertentangan) + daftar sumber yang cocok / bertentangan
Sumber-sumber yang saling bertentangan (mis. REF 2.5 x 24 tapi teks 2.5 x 29) selalu ditampilkan, tidak disembunyikan.
"""
from __future__ import annotations

import datetime as dt

import cv2
import numpy as np

from src.box_barcode import _variants, parse_gs1
from src.box_ref import parse_refs
from src.box_verifier import Evidence, decide_multi, describe_request, parse_identity


def read_all_codes(img: np.ndarray) -> list[dict]:
    """Semua kode yang bisa didekode (juga yang bukan GS1). -> [{format, text, gs1 (Gs1Code|None)}]."""
    try:
        import zxingcpp
    except ImportError:
        return []
    seen: dict = {}

    def add(results):
        for r in results:
            key = (str(r.format), r.text)
            if key not in seen:
                seen[key] = {"format": str(r.format).split(".")[-1], "text": r.text,
                             "gs1": parse_gs1(r.text, str(r.format).split(".")[-1])}

    add(zxingcpp.read_barcodes(img))
    for v, _ in _variants(img, 4000):                  # dipertajam, diperbesar 2x
        add(zxingcpp.read_barcodes(v))
        add(zxingcpp.read_barcodes(v, binarizer=zxingcpp.Binarizer.GlobalHistogram))
    # foto ponsel 12 MP: barcode besar sering baru terbaca setelah gambar diperkecil (terukur: Sapphire 3,
    # barcode Code 128 terbaca di skala 0,5 - 0,25 dan tidak di resolusi penuh)
    for sc in (0.5, 0.35, 0.25):
        if max(img.shape[:2]) * sc < 800:
            break
        small = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        add(zxingcpp.read_barcodes(small))
        for v, _ in _variants(small, 4000):           # terukur: Sapphire 3 hanya terbaca di skala 0,35 x2 tajam
            add(zxingcpp.read_barcodes(v))
            add(zxingcpp.read_barcodes(v, binarizer=zxingcpp.Binarizer.GlobalHistogram))
    return list(seen.values())


def _starts(n: int, tile: int, step: int) -> list[int]:
    if n <= tile:
        return [0]
    out = list(range(0, n - tile, step))
    return out + [n - tile]                              # petak terakhir menempel tepi: seluruh gambar tercakup


def _grid(img: np.ndarray, tile: int = 1280, overlap: float = 0.2):
    h, w = img.shape[:2]
    step = int(tile * (1 - overlap))
    for y in _starts(h, tile, step):
        for x in _starts(w, tile, step):
            yield img[y:y + tile, x:x + tile]


def read_all_text(img: np.ndarray, reader, cfg: dict):
    """OCR tegak dan diputar 90 derajat. -> (semua token, identitas).
    Nama produk dan ANGKA UKURAN hanya dari gambar UTUH (ukuran asli dan setengah): petak bisa memotong angka besar
    ('2.5' jadi '5' - terukur pada foto close-up Accuforce). Kalau lintasan utuh memberi angka yang BERBEDA, angka
    itu ditandai ragu (tidak dipakai). Petak resolusi penuh hanya untuk teks kecil: REF, LOT, teks GTIN."""
    rng = (tuple(cfg["matching"]["diameter_range_mm"]), tuple(cfg["matching"]["length_range_mm"]))
    tokens, idents = [], []
    for rot in (None, cv2.ROTATE_90_CLOCKWISE):
        im = img if rot is None else cv2.rotate(img, rot)
        wholes = [im] + ([cv2.resize(im, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)]
                         if min(im.shape[:2]) > 600 else [])
        for w in wholes:
            toks = reader.read_once(w).tokens
            tokens += toks
            if toks:
                idents.append(parse_identity(toks, cfg["catalog"], cfg["ocr"]["min_confidence"], *rng))
        for t in _grid(im):
            if t.shape[:2] != im.shape[:2]:
                tokens += reader.read_once(t).tokens
    ident = parse_identity([], cfg["catalog"], cfg["ocr"]["min_confidence"], *rng)
    for field in ("product", "diameter", "length"):
        vals = {getattr(i, field) for i in idents if getattr(i, f"{field}_ok")}
        if len(vals) == 1:
            setattr(ident, field, vals.pop())
            setattr(ident, f"{field}_ok", True)
        elif len(vals) > 1:
            ident.notes.append(f"{field} berbeda antar bacaan: {sorted(map(str, vals))}")
    ident.text = " ".join(t.text for t in tokens)[:200]
    return tokens, ident


def scan_box(img: np.ndarray, reader, cfg: dict, req=None, catalog=None, today: dt.date | None = None) -> dict:
    today = today or dt.date.today()
    codes = read_all_codes(img)
    tokens, ident = read_all_text(img, reader, cfg)
    text = " ".join(t.text for t in tokens)
    refs = parse_refs(text)
    gs1 = []
    for c in codes:
        if c["gs1"] is not None and all(g.gtin != c["gs1"].gtin for g in gs1):
            gs1.append(c["gs1"])
    card = {"waktu": dt.datetime.now().isoformat(timespec="seconds"), "kode": [], "ref": [], "teks": {},
            "permintaan": describe_request(req) if req is not None else "", "keputusan": "", "alasan": ""}
    for c in codes:
        g = c["gs1"]
        row = {"jenis": c["format"], "isi": c["text"][:120]}
        if g is not None:
            e = catalog.get(g.gtin) if catalog is not None else None
            row.update(gtin=g.gtin, kedaluwarsa=g.expiry.isoformat() if g.expiry else "", lot=g.lot or "",
                       seri=g.serial or "", sudah_kedaluwarsa=bool(g.expiry and g.expiry < today),
                       katalog=(str(e) + (" (terverifikasi)" if e.verified else " (DRAF, belum diverifikasi)"))
                       if e else "belum ada di katalog - daftarkan dengan scripts/enroll_gtin.py")
        card["kode"].append(row)
    card["ref"] = [f"{r.text} = {r.product} {r.diameter:g} x {r.length:g}" for r in refs]
    card["catatan"] = ident.notes
    card["teks"] = {"produk": ident.product if ident.product_ok else "",
                    "diameter": ident.diameter if ident.diameter_ok else None,
                    "panjang": ident.length if ident.length_ok else None, "cuplikan": text[:200]}
    if req is not None:
        v = cfg.get("verify", {})
        ev = Evidence(text=ident if tokens else None, refs=refs, codes=gs1)
        d, reason, _ = decide_multi(ev, req, catalog=catalog, today=today, min_score=v.get("min_score", 2),
                                    trust_unverified=v.get("trust_unverified_catalog", False))
        card["keputusan"], card["alasan"] = d, reason
    return card


def card_lines(card: dict) -> list[str]:
    out = [f"PINDAI {card['waktu']}" + (f" | permintaan: {card['permintaan']}" if card["permintaan"] else "")]
    if not card["kode"]:
        out.append("kode: tidak ada barcode / QR yang terbaca (dekatkan, luruskan, cahaya cukup)")
    for k in card["kode"]:
        if "gtin" in k:
            out.append(f"{k['jenis']}: GTIN {k['gtin']} | exp {k['kedaluwarsa'] or '-'}"
                       + (" (KEDALUWARSA)" if k["sudah_kedaluwarsa"] else "")
                       + f" | LOT {k['lot'] or '-'}" + (f" | SN {k['seri']}" if k["seri"] else ""))
            out.append(f"   katalog: {k['katalog']}")
        else:
            out.append(f"{k['jenis']}: {k['isi']}")
    out += [f"REF {r}" for r in card["ref"]] or ["REF: tidak terbaca"]
    t = card["teks"]
    out.append(f"teks: produk {t['produk'] or '?'} | diameter {t['diameter'] if t['diameter'] is not None else '?'}"
               f" | panjang {t['panjang'] if t['panjang'] is not None else '?'}")
    out += [f"   catatan: {n}" for n in card.get("catatan", []) if "berbeda" in n]
    if card["keputusan"]:
        out.append(f"KEPUTUSAN: {card['keputusan']} - {card['alasan']}")
    return out

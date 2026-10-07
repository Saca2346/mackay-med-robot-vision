#!/usr/bin/env python3
"""
Kelola katalog GTIN (data/gtin_catalog.csv): hanya entri yang DIVERIFIKASI orang yang dipakai untuk MATCH.

1) Periksa draf (entri verified = 0, mis. yang dibaca dari foto dataset):
     python scripts/enroll_gtin.py --review
   Tiap entri: gambar bukti dibuka (data/gtin_evidence/GTIN.jpg), lalu jawab di terminal:
     y = benar (tandai terverifikasi)   e = ubah dulu lalu tandai   n = hapus entri   s = lewati

2) Daftarkan kotak baru dari kamera atau foto (barcode GS1 harus terbaca):
     python scripts/enroll_gtin.py --camera auto     (tekan e saat barcode terlihat, q keluar)
     python scripts/enroll_gtin.py --source foto_atau_folder
   Untuk tiap GTIN baru, saran produk/ukuran dari OCR ditampilkan; ketik ulang / koreksi, lalu konfirmasi.

3) Daftar produk/ukuran yang belum punya GTIN:  python scripts/enroll_gtin.py --belum
   Ganti nama pemeriksa yang tercatat:           python scripts/enroll_gtin.py --ganti-pemeriksa "NAMA LAMA"
   Tinjau ulang juga entri terverifikasi:        python scripts/enroll_gtin.py --review --semua
   (pilihan e = ubah produk, ukuran, merek, REF / tanda kemasan seperti "MR"; lalu tertanda terverifikasi)
Nama pemeriksa (untuk audit) ditanyakan di terminal; --by "Nama" / nama kedua di --ganti-pemeriksa juga bisa.
Contoh dari petunjuk yang tersalin apa adanya ("nama anda", "NamaAsliAnda", "<nama Anda>") ditolak.

Periksa SELALU dengan tulisan di kemasan asli: produk, diameter, panjang. Entri salah = kotak salah diambil.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import re
import subprocess
import sys
from datetime import datetime

import cv2

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.box_barcode import decode_gs1  # noqa: E402
from src.box_catalog import Catalog, CatalogEntry, norm_product  # noqa: E402
from src.box_verifier import brand_of  # noqa: E402
from src.box_ref import parse_refs  # noqa: E402
from src.config import load_config, resolve  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
PLACEHOLDERS = {"namaanda", "nama", "name", "namaasli", "namaaslianda", "namapemeriksa", "yourname", "lama", "baru"}


def is_placeholder(name: str | None) -> bool:
    """Contoh dari petunjuk yang tersalin apa adanya ("nama anda", "NamaAsliAnda", "<nama Anda>") bukan nama pemeriksa."""
    s = (name or "").strip()
    return not s or "<" in s or ">" in s or re.sub(r"[^a-z]", "", s.lower()) in PLACEHOLDERS


def open_image(path: pathlib.Path) -> None:
    """Buka gambar bukti dengan penampil bawaan (tidak memblokir terminal)."""
    try:
        if sys.platform == "win32":
            os.startfile(str(path))  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        print(f"   (buka manual: {path})")


def ask_name(given: str | None, what: str) -> str:
    """Nama pemeriksa dari argumen, atau ditanyakan di terminal (perintah di petunjuk bisa disalin apa adanya)."""
    name = (given if given is not None else input(f"{what} (ketik nama ASLI Anda, dicatat untuk audit): ")).strip()
    if is_placeholder(name):
        sys.exit(f"'{name}' bukan nama pemeriksa (contoh dari petunjuk / kosong). Jalankan lagi dan ketik nama ASLI Anda.")
    return name


def ask_identity(default: CatalogEntry | None):
    """Minta produk, diameter, panjang, merek, REF di terminal. Enter = pakai saran dalam [kurung];
    "-" = kosongkan. -> (produk, diameter, panjang, merek, REF) atau None."""
    def q(label, cur):
        v = input(f"   {label} [{'' if cur is None else cur}]: ").strip()
        return "" if v == "-" else (v if v else cur)
    product = q("produk", default.product if default else "")
    if not product:
        return None
    product = norm_product(product)
    d = q("diameter mm (kosong = tanpa ukuran)", None if default is None or default.diameter_mm is None
          else f"{default.diameter_mm:g}")
    length = q("panjang mm", None if default is None or default.length_mm is None else f"{default.length_mm:g}") if d else None
    brand = q("merek", (default.brand if default and default.brand else brand_of(product)) or "")
    ref = q("REF / tanda di kemasan (mis. DC-RM2720HHW, SCCDSR14150250029, MR; '-' = tidak ada)",
            default.ref if default else "")
    return product, float(str(d).replace(",", ".")) if d else None, \
        float(str(length).replace(",", ".")) if length else None, (brand or "").strip(), (ref or "").strip()


def review(catalog: Catalog, evidence_dir: pathlib.Path, by: str, everything: bool = False) -> None:
    todo = [e for e in catalog.entries.values() if everything or not e.verified]
    print(f"{len(todo)} entri " + ("(semua)" if everything else "belum diverifikasi"))
    for e in todo:
        print(f"\nGTIN {e.gtin}: {e} | merek {e.brand or '-'} | REF {e.ref or '-'}"
              + (f" | terverifikasi oleh {e.verified_by} {e.verified_at}" if e.verified else "")
              + f"\n   catatan: {e.note}")
        ev = evidence_dir / f"{e.gtin}.jpg"
        if ev.exists():
            open_image(ev)
        ans = input("   cocok dengan kemasan? y = benar, e = ubah, n = hapus, s = lewati: ").strip().lower()
        if ans == "y":
            catalog.verify(e.gtin, by)
        elif ans == "e":
            ident = ask_identity(e)
            if ident:
                e.product, e.diameter_mm, e.length_mm, e.brand, e.ref = ident
                catalog.verify(e.gtin, by)
        elif ans == "n":
            del catalog.entries[e.gtin]
        catalog.save()                                   # simpan tiap langkah
    n_ok = sum(1 for e in catalog.entries.values() if e.verified)
    print(f"\nselesai: {n_ok}/{len(catalog.entries)} entri terverifikasi -> {catalog.path}")


def rename_verifier(catalog: Catalog, old: str, new: str) -> int:
    n = 0
    for e in catalog.entries.values():
        if e.verified and e.verified_by == old:
            e.verified_by, n = new, n + 1
    return n


def missing_list(catalog: Catalog) -> None:
    """Produk/ukuran yang pernah terlihat di foto dataset (data/verify_eval_gt.csv) tapi belum punya GTIN di katalog."""
    import csv
    seen = {}
    with open(resolve("data/verify_eval_gt.csv"), newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if not (r.get("photo") or "").strip() or r["photo"].startswith("#"):
                continue
            d = float(r["diameter"]) if (r.get("diameter") or "").strip() else None
            length = float(r["length"]) if (r.get("length") or "").strip() else None
            k = (norm_product(r["product"]), d, length)
            seen[k] = seen.get(k, 0) + 1
    have = {e.key(): e for e in catalog.entries.values()}
    print(f"{'produk':<22}{'ukuran':<14}{'merek':<19}status katalog")
    for (p, d, length) in sorted(seen, key=lambda k: (brand_of(k[0]), k[0], str(k[1:]))):
        e = have.get((p, d, length))
        size = "-" if d is None else f"{d:g} x {length:g}"
        if e is None:
            status = "BELUM ADA GTIN - daftarkan"
        else:
            status = ("TERVERIFIKASI" if e.verified else "draf (belum diverifikasi)") + f" GTIN {e.gtin}"
        print(f"{p:<22}{size:<14}{brand_of(p) or '?':<19}{status}")
    for k, e in have.items():
        if k not in seen:
            size = "-" if e.diameter_mm is None else f"{e.diameter_mm:g} x {e.length_mm:g}"
            print(f"{e.product:<22}{size:<14}{e.brand or '?':<19}"
                  f"{'TERVERIFIKASI' if e.verified else 'draf'} GTIN {e.gtin} (tidak ada di foto dataset)")


def suggest(frame, reader, cfg) -> CatalogEntry | None:
    """Saran identitas dari OCR seluruh frame: kode REF dulu (paling spesifik), lalu teks ukuran."""
    if reader is None:
        return None
    from src.box_verifier import parse_identity
    toks = []
    for rot in (None, cv2.ROTATE_90_CLOCKWISE):
        toks += reader.read_once(frame if rot is None else cv2.rotate(frame, rot)).tokens
    text = " ".join(t.text for t in toks)
    refs = parse_refs(text)
    if len(refs) == 1:
        r = refs[0]
        return CatalogEntry(gtin="", product=r.product, diameter_mm=r.diameter, length_mm=r.length, ref=r.text)
    ident = parse_identity(toks, cfg["catalog"], 0.0, tuple(cfg["matching"]["diameter_range_mm"]),
                           tuple(cfg["matching"]["length_range_mm"]))
    if ident.product:
        return CatalogEntry(gtin="", product=ident.product, diameter_mm=ident.diameter, length_mm=ident.length)
    return None


def enroll_frame(frame, catalog, evidence_dir, by, reader, cfg) -> int:
    codes = decode_gs1(frame, hard=True)
    if not codes:
        print("   tidak ada barcode GS1 yang terbaca - dekatkan / luruskan barcode ke kamera")
        return 0
    added = 0
    for c in codes:
        cur = catalog.get(c.gtin)
        if cur is not None and cur.verified:
            print(f"   GTIN {c.gtin} sudah terverifikasi: {cur}")
            continue
        print(f"\n   GTIN {c.gtin} (exp {c.expiry}, LOT {c.lot})" + (f" - draf: {cur}" if cur else ""))
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ev = evidence_dir / f"{c.gtin}.jpg"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(ev), frame)
        open_image(ev)
        ident = ask_identity(cur or suggest(frame, reader, cfg))
        if not ident:
            print("   dilewati")
            continue
        size = "" if ident[1] is None else f" {ident[1]:g} x {ident[2]:g}"
        if input(f"   simpan {ident[0]}{size} | merek {ident[3] or '-'} | REF {ident[4] or '-'} sebagai terverifikasi? "
                 "(y/n): ").strip().lower() != "y":
            continue
        e = CatalogEntry(gtin=c.gtin, product=ident[0], diameter_mm=ident[1], length_mm=ident[2], brand=ident[3],
                         ref=ident[4], note=f"didaftarkan {stamp} dari kamera/foto (bukti {ev.name}); exp "
                                        f"{c.expiry or '-'}, LOT {c.lot or '-'}")
        catalog.upsert(e)
        catalog.verify(c.gtin, by)
        catalog.save()
        added += 1
    return added


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/box_pipeline_config.yaml")
    ap.add_argument("--review", action="store_true", help="periksa entri draf (verified = 0)")
    ap.add_argument("--camera", default=None, help='"auto" (webcam eksternal) atau nomor kamera 0 / 1')
    ap.add_argument("--source", default=None, help="foto atau folder foto")
    ap.add_argument("--by", default=None,
                    help="nama pemeriksa ASLI (dicatat di katalog sebagai bukti audit); kalau tidak diisi, ditanyakan")
    ap.add_argument("--semua", action="store_true", help="--review: tampilkan juga entri yang sudah terverifikasi")
    ap.add_argument("--belum", action="store_true",
                    help="daftar produk/ukuran yang pernah terlihat di dataset tapi GTIN-nya belum ada di katalog")
    ap.add_argument("--ganti-pemeriksa", nargs="+", metavar="NAMA",
                    help='ganti nama pemeriksa yang tercatat: --ganti-pemeriksa "NAMA LAMA" (nama baru ditanyakan)')
    args = ap.parse_args()

    cfg = load_config(resolve(args.config))
    catalog = Catalog(resolve(cfg.get("verify", {}).get("catalog_csv", "data/gtin_catalog.csv")))
    evidence_dir = resolve("data/gtin_evidence")
    if args.belum:
        missing_list(catalog)
        return
    if args.ganti_pemeriksa:
        if len(args.ganti_pemeriksa) > 2:
            sys.exit('--ganti-pemeriksa "NAMA LAMA" ["NAMA BARU"] (pakai tanda kutip kalau nama berisi spasi)')
        old = args.ganti_pemeriksa[0]
        names = sorted({e.verified_by for e in catalog.entries.values() if e.verified})
        if old not in names:
            sys.exit(f"tidak ada entri dengan pemeriksa '{old}'. Nama yang tercatat: {', '.join(names) or '-'}")
        new = ask_name(args.ganti_pemeriksa[1] if len(args.ganti_pemeriksa) == 2 else None,
                       f"nama baru untuk pemeriksa '{old}'")
        n = rename_verifier(catalog, old, new)
        catalog.save()
        print(f"{n} entri: pemeriksa '{old}' -> '{new}' ({catalog.path})")
        return
    for name in sorted({e.verified_by for e in catalog.entries.values() if e.verified and is_placeholder(e.verified_by)}):
        print(f"PERINGATAN: pemeriksa tercatat '{name}' bukan nama asli -> "
              f'python scripts/enroll_gtin.py --ganti-pemeriksa "{name}"')
    args.by = ask_name(args.by, "nama pemeriksa")
    if args.review or (args.camera is None and args.source is None):
        review(catalog, evidence_dir, args.by, args.semua)
        return
    from src.box_ocr import SizeTextReader
    reader = SizeTextReader(cfg["ocr"]["engine"], cfg["ocr"]["retry_flip_below"])
    if args.source:
        p = pathlib.Path(args.source)
        files = sorted(x for x in p.iterdir() if x.suffix.lower() in IMG_EXT) if p.is_dir() else [p]
        for f in files:
            print(f"\n{f.name}")
            enroll_frame(cv2.imread(str(f)), catalog, evidence_dir, args.by, reader, cfg)
        return
    from src.box_camera import open_camera, sharpness
    cap, idx, size = open_camera(cfg["camera"], None if args.camera == "auto" else args.camera)
    if cap is None:
        sys.exit("Kamera tidak bisa dibuka (cek USB webcam / tutup aplikasi lain yang memakai kamera).")
    print(f"kamera #{idx} {size}; tekan e saat barcode terlihat jelas di jendela, q untuk keluar")
    n = 0
    codes_txt = ""
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        if n % 10 == 0:                                   # cek barcode 3x per detik (dekode 1080p cukup berat)
            found = decode_gs1(frame, hard=True)
            codes_txt = ", ".join(c.gtin for c in found) if found else "belum ada barcode GS1 terbaca"
        s = 960 / frame.shape[1]
        view = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else frame.copy()
        cv2.putText(view, f"tajam {sharpness(frame):.0f} | {codes_txt}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Daftar GTIN (e daftar, q keluar)", view)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord("e"):
            enroll_frame(frame.copy(), catalog, evidence_dir, args.by, reader, cfg)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

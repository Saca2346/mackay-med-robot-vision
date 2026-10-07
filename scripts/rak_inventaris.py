#!/usr/bin/env python3
"""
Inventaris rak stent / balon untuk operasi (lihat src/box_inventory.py).

SEBELUM HARI OPERASI
  (opsional) daftar isi rak sudah ada (CSV: produk,diameter,panjang,jumlah[,gtin,lot,kedaluwarsa]):
      python scripts/rak_inventaris.py impor daftar_isi_rak.csv
  1. pindai rak (kamera DIAM di depan rak; rak besar: per bagian, lihat --lanjut):
      python scripts/rak_inventaris.py pindai
     Tiap kotak dibaca berkali-kali; identitas hanya ditetapkan kalau >= 2 bukti independen sepakat (barcode /
     teks ukuran / REF) tanpa bacaan yang bertentangan. Hijau = identitas ditetapkan, kuning = masih membaca,
     merah = bacaan bertentangan. Tekan q kalau semua kotak hijau (atau tidak berubah lagi) -> disimpan.
  2. perawat memeriksa SETIAP kotak (gambar punggung + data), mengonfirmasi atau mengoreksi:
      python scripts/rak_inventaris.py cek
     Lembar cetak semua kotak: data/rak/lembar_cek.jpg
  3. ringkasan:  python scripts/rak_inventaris.py lihat

SAAT OPERASI
      python scripts/test_webcam_box.py --inventaris data/rak/inventaris.csv --request "angiolite,2.5,29"
  Kotak yang dicari langsung diketahui dari inventaris (kedaluwarsa paling dulu), dicari lewat ciri tampilan
  punggungnya (tetap ketemu walau sudah dipindah), lalu diverifikasi ulang sebelum AMBIL. Kotak yang TERAMBIL
  ditandai "diambil" di inventaris. Dikembalikan ke rak (uji coba):  python scripts/rak_inventaris.py kembali K15
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import os
import pathlib
import queue
import shutil
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.box_catalog import Catalog  # noqa: E402
from src.box_detector import BoxDetector, group_labels  # noqa: E402
from src.box_evidence import box_width_px, prepare_crops, read_any  # noqa: E402
from src.box_geometry import draw_poly, put_label, to_poly  # noqa: E402
from src.box_inventory import (STATUS_CEK, STATUS_DAFTAR, STATUS_DIAMBIL, STATUS_DRAF, STATUS_OK, InvBox,  # noqa: E402
                               Inventory, codes_for, identify, import_list, place_scanned, sig_str, signature,
                               similarity, spine_crop)
from src.box_motion import MotionEstimator  # noqa: E402
from src.box_ocr import SizeTextReader  # noqa: E402
from src.box_verifier import CONFIRM, SimpleTracker, brand_of  # noqa: E402
from src.config import load_config, resolve  # noqa: E402

DEFAULT_INV = "data/rak/inventaris.csv"
GREEN, YELLOW, RED, GRAY = (0, 200, 0), (0, 215, 255), (0, 0, 230), (150, 150, 150)
MIN_FRAMES = 15               # track yang terlihat kurang dari ini = deteksi sesaat, bukan kotak
MIN_READS_UNKNOWN = 3         # kotak tanpa identitas dengan bacaan kurang dari ini tidak dicatat (sisa nomor lama)
SAME_KEY_SIM = 0.85           # segmen lain, identitas sama, tampilan semirip ini -> kotak yang sama (digabung)
SAME_UNKNOWN_SIM = 0.93       # segmen lain, salah satu tanpa identitas -> hanya digabung kalau semirip ini
REREAD_OPEN, REREAD_DONE = 6, 45   # jeda baca ulang (frame): kotak belum / sudah ditetapkan
DUP_SIM = 0.93                # dua kotak identitas + LOT sama dan tampilan semirip ini: mungkin kotak yang sama


# ------------------------------------------------------------------ pembaca latar
class IdentityWorker:
    def __init__(self, readers, cfg):
        self.cfg = cfg
        self.jobs: queue.Queue = queue.Queue(maxsize=len(readers) + 1)
        self.results: queue.Queue = queue.Queue()
        for r in readers:
            threading.Thread(target=self._loop, args=(r,), daemon=True).start()

    def submit(self, tid, epoch, crops) -> bool:
        try:
            self.jobs.put_nowait((tid, epoch, crops))
            return True
        except queue.Full:
            return False

    def _loop(self, reader):
        while True:
            tid, epoch, crops = self.jobs.get()
            try:
                self.results.put((tid, epoch, read_any(*crops, reader, self.cfg)))
            except Exception as e:  # noqa: BLE001
                print(f"[baca] gagal: {type(e).__name__}: {e}")
                self.results.put((tid, epoch, None))

    def drain(self):
        while True:
            try:
                yield self.results.get_nowait()
            except queue.Empty:
                return


def _new_rec(seq, seg=0):
    """seg = segmen pindai: naik setiap kali posisi kamera hilang (semua nomor kotak diulang)."""
    return {"evs": collections.deque(maxlen=8), "epoch": 0, "frames": 0, "crop": None, "sharp": -1.0,
            "seq": seq, "seg": seg, "reads": 0, "far": 0, "pending": False, "next": 0,
            "id": (None, "", "belum terbaca")}


# ------------------------------------------------------------------ 1. pindai
def pindai(args):
    import test_webcam_box as twb                 # sumber gambar (kamera / video / foto) yang sama dengan uji live
    cfg = load_config(resolve(args.config))
    inv_path = resolve(args.inventaris)
    inv = Inventory(inv_path)
    if not args.lanjut:
        _start_fresh(inv)
    catalog = Catalog(resolve(cfg.get("verify", {}).get("catalog_csv", "data/gtin_catalog.csv")))
    weights = resolve(args.weights or cfg["model"]["weights"])
    print(f"memuat model {weights.name} ...", flush=True)
    det = BoxDetector(str(weights), cfg["model"]["task"], cfg["model"]["imgsz"], cfg["model"]["device"],
                      cfg["model"]["conf"], cfg["model"]["iou"], cfg["classes"])
    threads = int(cfg["ocr"].get("threads", 0)) or max(1, (os.cpu_count() or 2) // 2)
    readers = [SizeTextReader(cfg["ocr"]["engine"], cfg["ocr"]["retry_flip_below"], threads=threads)
               for _ in range(max(int(cfg["ocr"].get("workers", 1)), 1))]
    worker = IdentityWorker(readers, cfg)
    tracker = SimpleTracker(history=5, max_out=10 ** 9, max_missed=60)
    motion_est = MotionEstimator()
    max_motion = cfg["ocr"].get("max_motion_px", 6)
    recs: dict[int, dict] = {}
    src_args = argparse.Namespace(source=args.source, camera=args.camera, video_fps=0.0, all_frames=False)
    info: dict = {}
    frame_idx, prev_t, resets, t_start = 0, None, 0, time.monotonic()
    print("PINDAI RAK: kamera diam di depan rak. Hijau = identitas ditetapkan, kuning = membaca, merah = "
          "bacaan bertentangan. q = selesai dan simpan.")
    last_vis = None
    for _, frame, _ in twb.iter_source(src_args, cfg, info):
        if frame is None:
            cv2.waitKey(1)
            continue
        now = info.get("t", time.monotonic())
        dt_s, prev_t = (None if prev_t is None else now - prev_t), now
        for tid, epoch, ev in worker.drain():
            r = recs.get(tid)
            if r is None:
                continue
            r["pending"] = False
            if tid in tracker.tracks:
                tracker.state(tid).pending = False
            if ev is None or epoch != r["epoch"]:
                continue                                  # crop diambil sebelum nomor kotak diragukan: dibuang
            r["evs"].append(ev)
            r["reads"] += 1
            r["far"] += ev.too_far
            if tid in tracker.tracks:
                tracker.state(tid).add(CONFIRM, ("pindai",))      # supaya "nomor diragukan" terdeteksi
            r["id"] = identify(list(r["evs"]), catalog)
        dets, _ = det.detect(frame)
        groups, _ = group_labels(dets, cfg["grouping"].get("min_label_overlap", 0.5))
        polys = [g.box.poly for g in groups]
        motion, shift = motion_est.update(frame)
        if motion is None and shift == float("inf"):
            if motion_est.fails > cfg["ocr"].get("max_lost_frames", 10):
                tracker.reset()
                motion_est.reset()
                resets += 1
                print("[pindai] posisi kamera hilang: nomor kotak diulang dari awal (kotak yang sama bisa tercatat "
                      "dua kali - dicek saat 'cek'). Pegang kamera diam.")
            continue
        h, w = frame.shape[:2]
        ids = tracker.update(polys, motion, (w, h))
        blur = shift / max((dt_s or 0.0) * 30.0, 1.0)
        still = blur <= max_motion
        for i, tid in enumerate(ids):
            r = recs.setdefault(tid, _new_rec(len(recs), resets))
            st = tracker.state(tid)
            if st.epoch != r["epoch"]:
                r["epoch"], r["evs"], r["id"] = st.epoch, collections.deque(maxlen=8), (None, "", "dibaca ulang")
            r["frames"] += 1
            if still and frame_idx % 3 == 0 and box_width_px(polys[i]) >= 24:
                crop = spine_crop(frame, polys[i])
                sharp = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
                if sharp > r["sharp"]:
                    r["crop"], r["sharp"] = crop, sharp
        if still:
            due = sorted((i for i, tid in enumerate(ids) if not recs[tid]["pending"] and recs[tid]["next"] <= frame_idx),
                         key=lambda i: (recs[ids[i]]["id"][0] is not None, recs[ids[i]]["reads"]))
            for i in due[:2]:
                tid, r = ids[i], recs[ids[i]]
                crops = prepare_crops(frame, groups[i], cfg)
                if worker.submit(tid, r["epoch"], crops):
                    r["pending"] = tracker.state(tid).pending = True
                    r["next"] = frame_idx + (REREAD_DONE if r["id"][0] is not None else REREAD_OPEN)
        vis = frame.copy()
        n_ok = n_bad = 0
        for i, tid in enumerate(ids):
            key, _, note = recs[tid]["id"]
            if key is not None:
                color, text = GREEN, f"#{tid} {_fmt_key(key)}"
                n_ok += 1
            elif "bertentangan" in note or "dua identitas" in note:
                color, text = RED, f"#{tid} bertentangan"
                n_bad += 1
            else:
                color, text = YELLOW, f"#{tid} membaca ({recs[tid]['reads']})"
            p = to_poly(polys[i])
            draw_poly(vis, p, color, 3 if key is not None else 2)
            top = p[np.argmin(p[:, 1])]
            put_label(vis, text, (int(top[0]), int(top[1]) - 4), color, 0.5)
        hud = (f"PINDAI RAK | terlihat {len(ids)} kotak | ditetapkan {n_ok} | bertentangan {n_bad} | "
               f"{'KAMERA BERGERAK' if not still else 'diam'} | {time.monotonic() - t_start:.0f} s | q = simpan")
        put_label(vis, hud, (10, 30), (40, 40, 40), 0.7)
        last_vis = vis
        frame_idx += 1
        if not args.no_window:
            show = vis
            if vis.shape[1] > 1600:
                s = 1600 / vis.shape[1]
                show = cv2.resize(vis, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            cv2.imshow("Pindai rak (q = selesai dan simpan)", show)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
        if args.max_frames and frame_idx >= args.max_frames:
            break
    cv2.destroyAllWindows()
    # bacaan yang masih diproses ditunggu sebentar supaya tidak hilang
    t_wait = time.monotonic()
    while any(r["pending"] for r in recs.values()) and time.monotonic() - t_wait < 5:
        for tid, epoch, ev in worker.drain():
            r = recs.get(tid)
            if r is not None:
                r["pending"] = False
                if ev is not None and epoch == r["epoch"]:
                    r["evs"].append(ev)
                    r["reads"] += 1
                    r["id"] = identify(list(r["evs"]), catalog)
        time.sleep(0.05)
    _save_scan(inv, recs, catalog, resets, last_vis)


def _merge_segments(real):
    """Setelah posisi kamera hilang, kotak yang sama mendapat nomor baru (segmen baru). Catatan dari segmen yang
    LEBIH AKHIR digabung ke catatan segmen lebih awal kalau identitasnya sama dan tampilannya mirip (atau, bila salah
    satu tanpa identitas, tampilannya sangat mirip). Satu catatan lama hanya menyerap satu catatan per segmen,
    jadi dua kotak kembar yang sama-sama terlihat di satu segmen tetap dihitung dua. -> (catatan, jumlah digabung)."""
    kept, merged = [], 0
    for tid, r in real:
        r["sig"] = signature(r["crop"]) if r["crop"] is not None else None
        key = r["id"][0]
        best, best_sim = None, 0.0
        for k in kept:
            if k["seg"] >= r["seg"] or r["seg"] in k.setdefault("absorbed", set()):
                continue
            sim = similarity(k["sig"], r["sig"])
            k_key = k["id"][0]
            same = (key is not None and k_key == key and sim >= SAME_KEY_SIM) or                 ((key is None or k_key is None) and sim >= SAME_UNKNOWN_SIM)
            if same and sim > best_sim:
                best, best_sim = k, sim
        if best is None:
            kept.append(r)
            continue
        merged += 1
        best["absorbed"].add(r["seg"])
        # simpan yang lebih kuat: identitas pasti > usulan > tanpa identitas, lalu bacaan terbanyak
        rank = lambda x: (x["id"][0] is not None, not x["id"][1].startswith("USULAN"), x["reads"])
        if rank(r) > rank(best):
            for f in ("id", "evs", "reads", "far", "crop", "sig", "sharp"):
                best[f] = r[f]
        best["frames"] += r["frames"]
    return [(None, r) for r in kept], merged


def _start_fresh(inv: Inventory) -> None:
    """Pindai = petakan ulang seluruh rak. Inventaris lama dicadangkan; entri dari daftar isi rak dikembalikan ke
    'belum_dipetakan', hasil pindai sebelumnya dihapus dari inventaris aktif."""
    if not inv.boxes:
        return
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = inv.path.with_name(f"{inv.path.stem}_{stamp}.csv")
    shutil.copy2(inv.path, backup)
    print(f"inventaris lama dicadangkan -> {backup.name}")
    for bid, b in list(inv.boxes.items()):
        if b.note.startswith("dari daftar isi rak") and b.status != STATUS_DIAMBIL:
            b.status, b.sources, b.thumb, b.sig, b.order = STATUS_DAFTAR, "", "", "", 0
            b.checked_by = b.checked_at = ""
        else:
            del inv.boxes[bid]


def _save_scan(inv: Inventory, recs, catalog, resets, last_vis) -> None:
    thumbs = inv.dir / "punggung"
    thumbs.mkdir(parents=True, exist_ok=True)
    real = [(tid, r) for tid, r in recs.items() if r["frames"] >= MIN_FRAMES
            and (r["id"][0] is not None or r["reads"] >= MIN_READS_UNKNOWN)]
    real.sort(key=lambda x: (x[1]["seg"], x[1]["seq"]))
    real, n_merged = _merge_segments(real)
    made = []
    for order, (tid, r) in enumerate(real, 1):
        key, src, note = r["id"]
        sig = signature(r["crop"]) if r["crop"] is not None else None
        if key is not None:
            gtin, lot, exp = codes_for(list(r["evs"]), key, catalog)
            b = place_scanned(inv, key, src, gtin, lot, exp, order, "", sig_str(sig) if sig is not None else "")
        else:
            b = InvBox(box_id=inv.new_id(), status=STATUS_CEK, order=order,
                       sig=sig_str(sig) if sig is not None else "",
                       note=f"pindai: {note or 'bukti kurang'} ({r['reads']} bacaan"
                            + (f", {r['far']} terlalu jauh" if r["far"] else "") + ")")
            inv.boxes[b.box_id] = b
        if r["crop"] is not None:
            b.thumb = f"punggung/{b.box_id}.jpg"
            cv2.imwrite(str(inv.dir / b.thumb), r["crop"], [cv2.IMWRITE_JPEG_QUALITY, 92])
        made.append(b)
    # kotak yang mungkin tercatat dua kali (nomor kotak sempat diulang)
    for i, a in enumerate(made):
        for b in made[i + 1:]:
            if a.product and a.key() == b.key() and a.lot == b.lot and \
                    similarity(a.signature(), b.signature()) >= DUP_SIM:
                b.note = (b.note + f"; mungkin kotak yang sama dengan {a.box_id} - cek").strip("; ")
    inv.save()
    if last_vis is not None:
        cv2.imwrite(str(inv.dir / "pindai_terakhir.jpg"), last_vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
    sheet = contact_sheet(inv)
    c = inv.counts()
    print(f"\nPindai selesai: {len(made)} kotak tercatat -> {inv.path}")
    print(f"  identitas ditetapkan (draf, perlu cek perawat): {sum(1 for b in made if b.status == STATUS_DRAF)}")
    print(f"  identitas BELUM bisa ditetapkan (perlu_dicek)  : {sum(1 for b in made if b.status == STATUS_CEK)}")
    if c.get(STATUS_DAFTAR):
        print(f"  di daftar isi rak tapi TIDAK ditemukan         : {c[STATUS_DAFTAR]}  (cek apakah ada di rak)")
    extra = [b for b in made if "tidak ada di daftar" in b.note]
    if extra:
        print(f"  ditemukan tapi TIDAK ada di daftar isi rak      : {len(extra)} ({', '.join(b.box_id for b in extra)})")
    if resets:
        print(f"  PERINGATAN: posisi kamera hilang {resets}x; {n_merged} catatan ganda digabung otomatis "
              f"(identitas + tampilan sama). Hitung ulang jumlah fisik saat cek.")
    print(f"  lembar cek (gambar semua kotak): {sheet}")
    print("Langkah berikut: python scripts/rak_inventaris.py cek")


# ------------------------------------------------------------------ lembar cek (cetak / layar)
def contact_sheet(inv: Inventory) -> pathlib.Path:
    rows = []
    W, H = 900, 70
    order = {STATUS_CEK: 0, STATUS_DRAF: 1, STATUS_DAFTAR: 2, STATUS_OK: 3, STATUS_DIAMBIL: 4}
    for b in sorted(inv.boxes.values(), key=lambda b: (order.get(b.status, 9), b.order, b.box_id)):
        row = np.full((H, W, 3), 255, np.uint8)
        p = inv.thumb_path(b) if b.thumb else None
        if p is not None and p.exists():
            img = cv2.imread(str(p))
            s = min(420 / img.shape[1], (H - 8) / img.shape[0])
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            row[4:4 + img.shape[0], 4:4 + img.shape[1]] = img
        color = {STATUS_OK: (0, 140, 0), STATUS_DRAF: (0, 120, 200), STATUS_CEK: (0, 0, 200),
                 STATUS_DAFTAR: (150, 0, 150), STATUS_DIAMBIL: (120, 120, 120)}.get(b.status, (0, 0, 0))
        cv2.putText(row, f"{b.box_id}  {b.label()}  [{b.status}]", (435, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        extra = " ".join(x for x in (f"GTIN {b.gtin}" if b.gtin else "", f"LOT {b.lot}" if b.lot else "",
                                     f"exp {b.expiry}" if b.expiry else "", b.sources) if x)
        cv2.putText(row, extra[:60], (435, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)
        if b.note:
            cv2.putText(row, b.note[:70], (435, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 180), 1)
        cv2.line(row, (0, H - 1), (W, H - 1), (200, 200, 200), 1)
        rows.append(row)
    out = inv.dir / "lembar_cek.jpg"
    if rows:
        cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out


# ------------------------------------------------------------------ 2. cek perawat
def open_image(path: pathlib.Path) -> None:
    try:
        if sys.platform == "win32":
            os.startfile(str(path))  # noqa: S606
        else:
            subprocess.Popen(["xdg-open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        print(f"   (buka manual: {path})")


def ask_name(given: str | None) -> str:
    from enroll_gtin import is_placeholder
    name = (given if given is not None else input("Nama pemeriksa (ketik nama ASLI Anda, dicatat untuk audit): ")).strip()
    if is_placeholder(name):
        sys.exit(f"'{name}' bukan nama pemeriksa. Jalankan lagi dan ketik nama ASLI Anda.")
    return name


def _num(s: str) -> float | None:
    s = s.strip().replace(",", ".")
    return float(s) if s else None


def _ask(label, cur, check, tries: int = 3):
    """Tanya satu isian sampai lolos `check` (-> (nilai, pesan salah | None)). Enter = tetap, '-' = kosongkan."""
    for _ in range(tries):
        v = input(f"   {label} [{'' if cur in (None, '') else cur}]: ").strip()
        v = "" if v == "-" else (v if v else ("" if cur is None else str(cur)))
        val, err = check(v)
        if err is None:
            return val
        print(f"   ! {err}")
    raise ValueError(f"{label}: tidak diisi dengan benar")


def _check_product(v):
    """Produk harus salah satu dari daftar (nomor atau nama). Nama merek saja ditolak dengan pilihan produknya."""
    from src.box_verifier import ALIASES, BRANDS, PRODUCTS
    s = v.strip().lower()
    if s.isdigit() and 1 <= int(s) <= len(PRODUCTS):
        return PRODUCTS[int(s) - 1], None
    s = ALIASES.get(s, s)
    if s in PRODUCTS:
        return s, None
    brand = next((b for b in BRANDS if s.replace(" ", "") == b.replace(" ", "")), None)
    if brand:
        opts = ", ".join(f"{PRODUCTS.index(p) + 1} {p}" for p in sorted(BRANDS[brand]))
        return None, f"'{v}' itu MEREK, bukan produk. Pilih produknya: {opts}"
    return None, "pilih nomor / nama produk dari daftar di atas"


def _check_size(lo, hi):
    def check(v):
        if not v:
            return None, "wajib diisi (angka)"
        try:
            x = float(v.replace(",", "."))
        except ValueError:
            return None, "harus angka, mis. 2.75"
        return (x, None) if lo <= x <= hi else (None, f"di luar {lo:g}-{hi:g} mm")
    return check


def _check_gtin(v):
    from src.box_barcode import gtin_valid
    if not v:
        return "", None
    g = v.replace(" ", "")
    if g.isdigit() and len(g) == 13:
        g = "0" + g
    if not (g.isdigit() and len(g) == 14 and gtin_valid(g)):
        return None, "GTIN = 14 angka dari barcode (01)... dengan digit cek benar; kode REF (mis. DE-RS3024ASM) bukan GTIN"
    return g, None


def _check_lot(v):
    from src.box_verifier import BRANDS, PRODUCTS
    low = v.lower().replace("tm", "").strip()
    if any(p in low for p in PRODUCTS) or any(b.replace(" ", "") in low.replace(" ", "") for b in BRANDS):
        return None, "LOT berisi nama produk / merek - LOT adalah kode lot di kemasan (mis. 2511631), kosongkan kalau tidak ada"
    return v.strip(), None


def _check_expiry(v):
    if not v:
        return "", None
    try:
        return dt.date.fromisoformat(v).isoformat(), None
    except ValueError:
        return None, "format YYYY-MM-DD, mis. 2028-07-31"


def edit_box(b: InvBox) -> None:
    """Isi / koreksi satu kotak. Produk dipilih dari daftar; ukuran, GTIN, LOT dan tanggal diperiksa sebelum diterima
    (dulu: merek masuk ke produk dan nama produk masuk ke LOT tanpa peringatan)."""
    from src.box_verifier import BRANDS, NO_SIZE, PRODUCTS
    print("   produk: " + " | ".join(f"{i + 1} {p} ({next(k for k, v in BRANDS.items() if p in v)})"
                                     for i, p in enumerate(PRODUCTS)))
    prod = _ask("produk (nomor atau nama)", b.product or None, _check_product)
    b.product, b.brand = prod, brand_of(prod)
    if prod in NO_SIZE:
        b.diameter = b.length = None
    else:
        b.diameter = _ask("diameter mm", None if b.diameter is None else f"{b.diameter:g}", _check_size(1.0, 10.0))
        b.length = _ask("panjang mm", None if b.length is None else f"{b.length:g}", _check_size(6.0, 60.0))
    b.gtin = _ask("GTIN (14 angka dari barcode, '-' = kosong)", b.gtin or None, _check_gtin)
    b.lot = _ask("LOT (kode lot di kemasan, '-' = kosong)", b.lot or None, _check_lot)
    b.expiry = _ask("kedaluwarsa YYYY-MM-DD ('-' = kosong)", b.expiry or None, _check_expiry)


def cek(args):
    inv = Inventory(resolve(args.inventaris))
    if not inv.boxes:
        sys.exit(f"Inventaris kosong: {inv.path}. Jalankan 'pindai' dulu.")
    by = ask_name(args.by)
    todo_status = (STATUS_CEK, STATUS_DRAF) + ((STATUS_OK,) if args.semua else ())
    only = {k.strip().upper() for k in (getattr(args, "kotak", None) or "").split(",") if k.strip()}
    todo = sorted((b for b in inv.boxes.values() if (b.box_id in only if only else b.status in todo_status)),
                  key=lambda b: (b.status != STATUS_CEK, b.order, b.box_id))
    print(f"{len(todo)} kotak untuk diperiksa. Bandingkan dengan KOTAK ASLI di rak (nama produk, diameter, panjang, "
          f"kedaluwarsa), bukan hanya gambarnya.\nLembar semua kotak: {contact_sheet(inv)}")
    for b in todo:
        print(f"\n{b.box_id} (urutan {b.order}): {b.label()} | status {b.status}"
              + (f" | GTIN {b.gtin}" if b.gtin else "") + (f" | LOT {b.lot}" if b.lot else "")
              + (f" | exp {b.expiry}" if b.expiry else "")
              + (f"\n   bukti: {b.sources}" if b.sources else "") + (f"\n   catatan: {b.note}" if b.note else ""))
        if b.thumb and not args.tanpa_gambar and inv.thumb_path(b).exists():
            open_image(inv.thumb_path(b))
        ans = input("   y = benar, e = ubah/isi, n = hapus (bukan kotak / tercatat dua kali), s = lewati: ").strip().lower()
        if ans == "y" and _check_product(b.product or "")[1] is not None:
            print(f"   produk '{b.product}' belum benar / belum diisi: isi dulu")
            ans = "e"
        if ans == "e":
            try:
                edit_box(b)
            except ValueError as e:
                print(f"   {e} - kotak ini dilewati, ulangi nanti")
                continue
        if ans in ("y", "e"):
            if not b.product:
                print("   produk kosong: tidak bisa ditandai benar, dilewati")
                continue
            b.status, b.checked_by, b.checked_at = STATUS_OK, by, dt.datetime.now().isoformat(timespec="seconds")
        elif ans == "n":
            del inv.boxes[b.box_id]
        inv.save()                                        # simpan tiap langkah
    missing = [b for b in inv.boxes.values() if b.status == STATUS_DAFTAR]
    if missing:
        print(f"\n{len(missing)} kotak di daftar isi rak TIDAK ditemukan saat pindai:")
        for b in missing:
            ans = input(f"   {b.box_id} {b.label()}: n = memang tidak ada (hapus), s = biarkan (pindai ulang nanti): ")
            if ans.strip().lower() == "n":
                del inv.boxes[b.box_id]
                inv.save()
    contact_sheet(inv)
    lihat(args)


# ------------------------------------------------------------------ impor / lihat
def impor(args):
    inv = Inventory(resolve(args.inventaris))
    with open(args.file, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    n = import_list(inv, rows)
    inv.save()
    print(f"{n} kotak dari daftar -> {inv.path} (status belum_dipetakan). Berikutnya: pindai rak.")


def lihat(args):
    inv = Inventory(resolve(args.inventaris))
    c = inv.counts()
    print(f"\nInventaris {inv.path}: {len(inv.boxes)} kotak | " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())))
    groups: dict = collections.defaultdict(list)
    for b in inv.boxes.values():
        if b.status == STATUS_OK:
            groups[b.label()].append(b)
    for label in sorted(groups):
        bs = sorted(groups[label], key=lambda b: (b.expiry_date() is None, b.expiry_date() or dt.date.max))
        print(f"  {label:<34} {len(bs)} kotak: " + ", ".join(f"{b.box_id}" + (f" (exp {b.expiry})" if b.expiry else "")
                                                         for b in bs))
    if c.get(STATUS_DRAF) or c.get(STATUS_CEK):
        print(f"  BELUM siap operasi: {c.get(STATUS_DRAF, 0) + c.get(STATUS_CEK, 0)} kotak belum diperiksa perawat "
              f"(python scripts/rak_inventaris.py cek)")


def kembali(args):
    """Kotak yang tercatat "diambil" ternyata dikembalikan ke rak (uji coba / batal dipakai): status kembali seperti
    sebelum diambil. Hanya kotak yang sebelumnya sudah diperiksa perawat yang langsung terverifikasi lagi."""
    inv = Inventory(resolve(args.inventaris))
    now = dt.datetime.now().isoformat(timespec="seconds")
    for bid in [s.strip().upper() for s in args.kotak.split(",") if s.strip()]:
        b = inv.boxes.get(bid)
        if b is None:
            print(f"  {bid}: tidak ada di inventaris")
        elif b.status != STATUS_DIAMBIL:
            print(f"  {bid}: status {b.status}, bukan diambil - tidak diubah")
        else:
            b.status = STATUS_OK if b.checked_by else STATUS_CEK
            b.taken_at = ""
            b.note = (b.note + "; " if b.note else "") + f"dikembalikan ke rak {now}"
            print(f"  {bid}: {b.label()} -> {b.status}")
    inv.save()


def _fmt_key(key) -> str:
    p, d, l = key
    return p if d is None else f"{p} {d:g}x{l:g}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pindai", help="pindai rak dan petakan semua kotak")
    p.add_argument("--source", default=None, help="video / foto pengganti webcam")
    p.add_argument("--camera", default=None)
    p.add_argument("--lanjut", action="store_true", help="tambahkan ke inventaris yang ada (bagian rak berikutnya)")
    p.add_argument("--no-window", action="store_true")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--weights", default=None)
    c = sub.add_parser("cek", help="perawat memeriksa / mengoreksi tiap kotak")
    c.add_argument("--by", default=None, help="nama pemeriksa (kalau kosong, ditanyakan)")
    c.add_argument("--semua", action="store_true", help="periksa ulang juga kotak yang sudah terverifikasi")
    c.add_argument("--tanpa-gambar", action="store_true", help="jangan buka gambar punggung")
    c.add_argument("--kotak", default=None, help="hanya kotak ini, mis. K01,K08 (juga yang sudah terverifikasi)")
    i = sub.add_parser("impor", help="impor daftar isi rak (CSV)")
    i.add_argument("file")
    sub.add_parser("lihat", help="ringkasan inventaris")
    k = sub.add_parser("kembali", help="kotak yang tercatat diambil dikembalikan ke rak")
    k.add_argument("kotak", help="nomor kotak, mis. K15 atau K15,K16")
    for sp in sub.choices.values():
        sp.add_argument("--inventaris", default=DEFAULT_INV)
        sp.add_argument("--config", default="config/box_pipeline_config.yaml")
    args = ap.parse_args()
    {"pindai": pindai, "cek": cek, "impor": impor, "lihat": lihat, "kembali": kembali}[args.cmd](args)


if __name__ == "__main__":
    main()

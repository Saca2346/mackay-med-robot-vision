#!/usr/bin/env python3
"""
Uji pipeline BOX + SIZE_LABEL + verifikasi live dari webcam, dari stream webcam laptop, atau dari foto.

Tahap 1 (tanpa --request): deteksi generik. Semua kotak digambar biru, size_label
oranye. Pakai ini untuk menghitung "x dari 11 item terdeteksi" dan mencatat FPS.

Tahap 2 (dengan --request): cari SATU kotak yang diminta dan verifikasi (src/box_verifier.decide_multi),
dengan alur penyeleksi src/box_session.py: CEK RAK (kosong -> berhenti) -> hitung kotak -> MENCARI ->
DITEMUKAN (AMBIL) / TIDAK ADA / PERLU KONFIRMASI -> TERAMBIL (slot kosong terverifikasi) -> permintaan berikutnya.
  hijau tebal + AMBIL = kotak yang dipilih (MATCH; kalau ada beberapa, yang kedaluwarsa paling dulu)
  hijau      = MATCH      (>= 2 bukti independen: barcode GS1 terverifikasi, atau teks ukuran + kode REF)
  biru muda  = KANDIDAT   (baru satu bukti cocok -> dekatkan kamera / tunjukkan barcode)
  kuning     = CONFIRM    (ragu / bertentangan / terlalu jauh -> minta konfirmasi, tidak menebak)
  abu-abu    = IGNORED    (pasti bukan target: produk ATAU ukuran terbaca berbeda)
  putih      = masih membaca ("cocok? 1/2" = baru 1 dari 2 bacaan; "cek ulang" = sempat keluar layar)
  ungu       = KEDALUWARSA (dari barcode) -> tidak pernah diambil
  banner merah = TIDAK ADA / RAK KOSONG; banner kuning = PERLU KONFIRMASI; banner hijau = DITEMUKAN / SELESAI
Pada sumber live, OCR dan barcode dibaca di thread latar belakang: video tetap jalan.
Peristiwa penting (JUMLAH KOTAK, AMBIL, TERAMBIL, TIDAK ADA, MATCH, KEDALUWARSA, ...) dicatat ke results/verify_log.csv.

Contoh:
  python scripts/test_webcam_box.py                                   # Tahap 1, webcam 0
  python scripts/test_webcam_box.py --request "angiolite,2.5,29"      # Tahap 2
  python scripts/test_webcam_box.py --source data/uji/ --request "accuforce,2.75,20"   # dari foto
  python scripts/test_webcam_box.py --weights models/box_obb.onnx --camera 1
  # di server GPU (lewat Remmina), webcam laptop dikirim scripts/stream_webcam.py + tunnel SSH:
  python scripts/test_webcam_box.py --config config/box_pipeline_config_server.yaml --source http://127.0.0.1:18090/video.mjpg
  # rekam frame kamera asli sambil menguji, lalu putar ulang rekaman yang sama setelah mengubah setelan:
  python scripts/test_webcam_box.py --request "angiolite,2.5,29" --record results/uji_tahap2.mp4
  python scripts/test_webcam_box.py --request "angiolite,2.5,29" --source results/uji_tahap2.mp4

Kamera digerakkan tangan: posisi kotak diikuti lewat optical flow (src/box_motion.py) supaya nomor kotak dan
bacaannya tidak hilang; frame yang buram karena gerakan tidak di-OCR ("KAMERA BERGERAK - tahan diam").
Label "#3 cocok? 1/2" = baru 1 dari 2 bacaan yang dibutuhkan; belum boleh dipakai.

Tombol: q/ESC keluar | s simpan frame (asli + beranotasi) | l catat keadaan saat ini ke CSV
        o paksa baca ulang semua kotak sekarang | r permintaan yang sama sekali lagi (setelah kotak diambil)
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import pathlib
import queue
import re
import sys
import threading
import statistics
import time
from datetime import datetime

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.box_cascade import refine_candidates
from src.box_catalog import Catalog
from src.box_detector import BoxDetector, group_labels
from src.box_evidence import box_width_px, prepare_crops, read_evidence, redecide
from src.box_motion import MotionEstimator, consistent
from src.box_geometry import draw_poly, put_label, to_poly
from src.box_inventory import STATUS_DIAMBIL, Inventory, signature, similarity, spine_crop
from src.box_ocr import SizeTextReader
from src.box_camera import open_camera, sharpness
from src.box_order import OrderQueue, order_command, parse_order
from src.box_scan import card_lines, scan_box
from src.box_session import DITEMUKAN, RAK_KOSONG, SELESAI, TIDAK_ADA, SelectionSession
from src.box_stress import StressRun, load_plan
from src.box_stream import LatestFrame, MjpegReader, VideoRecorder, load_times
from src.box_verifier import (CANDIDATE, CONFIRM, EMPTY, EXPIRED, IGNORED, MATCH, READING, SimpleTracker,
                              describe_request, pick_fefo)
from src.config import load_config, resolve

COLORS = {  # BGR
    "stage1": (255, 170, 0), "label": (0, 140, 255),
    MATCH: (0, 200, 0), IGNORED: (150, 150, 150), CONFIRM: (0, 215, 255), READING: (255, 255, 255),
    CANDIDATE: (255, 230, 0), EXPIRED: (255, 0, 200),      # kedaluwarsa = ungu (merah khusus TIDAK ADA / RAK KOSONG)
}
PHASE_BGR = {"white": (255, 255, 255), "red": (0, 0, 230), "green": (0, 200, 0), "yellow": (0, 215, 255)}
TAGS = {MATCH: "MATCH", IGNORED: "", CONFIRM: "CONFIRM", READING: "membaca...", CANDIDATE: "KANDIDAT",
        EXPIRED: "KEDALUWARSA"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
NEXT_AFTER_TIDAK_ADA_S = 3.0   # pesanan beberapa barang: TIDAK ADA bertahan sekian detik -> barang berikutnya
SLOW_S = 1.0          # iterasi lebih lama dari ini dicetak sebagai [lambat]
LOG_FIELDS = ["time", "request", "track", "state", "reason", "gtin", "lot", "expiry", "text"]


READ_LOG = None       # --debug-readings: satu baris per bacaan (untuk memeriksa apakah nomor kotak pernah tertukar),
                      # plus baris decision=stale: kenapa nomor kotak diragukan (pasangan lemah / longgar / masuk layar)


def _identity_str(ev) -> str:
    t = ev.text
    parts = ["", "", ""]
    if t is not None:
        parts = [t.product if t.product_ok else "", f"{t.diameter:g}" if t.diameter_ok else "",
                 f"{t.length:g}" if t.length_ok else ""]
    refs = ";".join(f"{r.product}:{r.diameter:g}x{r.length:g}" for r in ev.refs)
    codes = ";".join(c.gtin for c in ev.codes)
    return "|".join(parts) + f"|{refs}|{codes}"


def _raw_str(ev) -> str:
    """Teks OCR mentah bacaan ini (label || REF yang dikenali || punggung): untuk mencari asal salah baca."""
    label = ev.text.text if ev.text is not None else ""
    refs = " ".join(r.text for r in ev.refs)
    return f"{label} || {refs} || {ev.spine}".replace('"', "'").replace("\n", " ")


def apply_reading(st, ev, decision, reason, key, ms, n, stats=None, epoch=None):
    if READ_LOG is not None:
        READ_LOG.write(f"{time.monotonic():.2f},{st.tid},{epoch if epoch is not None else ''},{st.epoch},"
                       f"{decision},\"{_identity_str(ev)}\",\"{_raw_str(ev)}\"\n")
        READ_LOG.flush()
    st.pending = False
    if stats is not None:
        stats["ocr_calls"] += n
        stats["ocr_ms"] += ms
        stats["reads"] += 1
    if epoch is not None and epoch != st.epoch:
        return                            # crop diambil SEBELUM nomor kotak diragukan: mungkin kotak lain, dibuang
    st.add(decision, key, getattr(ev, "support", None), ev)
    st.last_text = (ev.text.text if ev.text is not None else "")[:60]
    st.stale = False
    if decision != MATCH and st.combined_match(2):
        reason = f"gabungan {len([k for _, k in st.history if k != EMPTY][-4:])} bacaan: {st.combined_sources()}"
    st.last_reason = reason
    code = next((c for c in ev.codes if c.source == "barcode"), ev.codes[0] if ev.codes else None)
    if code is not None:
        st.gtin, st.lot, st.expiry = code.gtin, code.lot or "", code.expiry


class OcrWorker:
    """OCR + barcode di thread latar belakang (0,5-3 detik per crop di CPU laptop) supaya jendela video tidak
    membeku. Beberapa pembaca (`ocr.workers`, tiap thread punya mesin OCR sendiri) untuk CPU server yang kuat."""

    def __init__(self, readers, cfg, req, catalog, max_pending: int | None = None):
        readers = readers if isinstance(readers, list) else [readers]
        self.cfg, self.req, self.catalog = cfg, req, catalog
        # antrean pendek: crop yang menunggu lama sudah basi (kamera sudah pindah), lebih baik ambil yang baru
        self.jobs: queue.Queue = queue.Queue(maxsize=max_pending or len(readers) + 1)
        self.results: queue.Queue = queue.Queue()
        for reader in readers:
            threading.Thread(target=self._loop, args=(reader,), daemon=True).start()

    def submit(self, st, crops) -> bool:
        try:
            self.jobs.put_nowait((st, crops, st.epoch))
            return True
        except queue.Full:
            return False

    def _loop(self, reader) -> None:
        while True:
            st, crops, epoch = self.jobs.get()
            req = self.req                           # permintaan saat crop ini DIBACA (bisa berganti sesudahnya)
            try:
                self.results.put((st, epoch, req,
                                  *read_evidence(*crops, reader, self.cfg, req, self.catalog)))
            except Exception as e:  # noqa: BLE001 - satu crop gagal tidak boleh mematikan thread baca
                print(f"[ocr] gagal: {type(e).__name__}: {e}")
                st.pending = False

    def drain(self):
        while True:
            try:
                yield self.results.get_nowait()
            except queue.Empty:
                return


def rates(shown, info) -> dict:
    """fps NYATA 3 detik terakhir: frame yang ditampilkan, dan frame yang masuk dari kamera / stream. Angka "proses
    maks" hanya kecepatan satu iterasi (tanpa menunggu frame), jadi bisa tinggi walau gambar patah-patah."""
    now = time.perf_counter()
    t = [x for x in shown if x >= now - 3.0]
    out = {"real": (len(t) - 1) / (t[-1] - t[0]) if len(t) > 2 and t[-1] > t[0] else None}
    live = info.get("live")
    if live is not None:
        hist = info.setdefault("in_hist", collections.deque(maxlen=200))
        hist.append((now, live.captured))
        old = next(((a, b) for a, b in hist if a >= now - 3.0), None)
        if old and now > old[0]:
            out["inrate"] = (live.captured - old[1]) / (now - old[0])
    return out


REF_FPS = 30.0        # gerak kamera diukur dalam "px per 1/30 detik" (~ panjang buram gerak), apa pun fps kameranya


HINT_MIN = 0.80        # kemiripan tampilan dengan kotak inventaris yang dicari: di atas ini dibaca lebih dulu
HINT_EVERY = 10        # kemiripan tiap kotak dihitung ulang tiap 10 frame (atau setelah nomor kotak diragukan)


def update_hints(frame, groups, ids, tracker, targets, frame_idx):
    """Kemiripan punggung tiap kotak dengan kotak inventaris yang dicari (src/box_inventory.signature). Hanya
    menentukan URUTAN baca: kotak yang dipindah tetap ditemukan lebih dulu; keputusan AMBIL tetap dari verifikasi."""
    for g, tid in zip(groups, ids):
        if tid < 0:
            continue
        st = tracker.state(tid)
        if frame_idx - st.hint_frame < HINT_EVERY or box_width_px(g.box.poly) < 16:
            continue
        sig = signature(spine_crop(frame, g.box.poly))
        best = max(((similarity(sig, t_sig), bid) for bid, t_sig in targets), default=(0.0, ""))
        st.hint, st.hint_id, st.hint_frame = best[0], best[1], frame_idx


ROI_FILE = "results/area_rak.json"   # area rak yang dipilih dengan tombol a (kamera tripod), dipakai lagi otomatis


def in_roi(poly, roi, w, h) -> bool:
    """Pusat poligon di dalam area rak (x1, y1, x2, y2 dalam pecahan lebar / tinggi frame)?"""
    cx, cy = to_poly(poly).mean(axis=0)
    return roi[0] * w <= cx <= roi[2] * w and roi[1] * h <= cy <= roi[3] * h


def load_roi(arg, live):
    """--roi "x1,y1,x2,y2" (pecahan 0-1) / "none"; tanpa --roi: area tersimpan (ROI_FILE), hanya untuk kamera live
    (area itu milik posisi kamera sekarang, bukan rekaman lama)."""
    if arg and arg.lower() == "none":
        return None
    if arg:
        v = [float(x) for x in arg.split(",")]
        if len(v) != 4 or not (0 <= v[0] < v[2] <= 1 and 0 <= v[1] < v[3] <= 1):
            sys.exit('--roi harus "x1,y1,x2,y2" dalam pecahan 0-1, mis. 0.05,0,0.62,0.75')
        return tuple(v)
    p = resolve(ROI_FILE)
    if live and p.exists():
        v = json.loads(p.read_text(encoding="utf-8"))["roi"]
        print(f"[area rak] dipakai dari {ROI_FILE} (tombol a = pilih ulang, --roi none = matikan)")
        return tuple(v)
    return None


def process(frame, det, reader, tracker, req, cfg, frame_idx, ocr_all=False, stats=None, worker=None,
            catalog=None, motion_est=None, need=2, dt=None, session=None, now=0.0, targets=None, roi=None,
            det_big=None, cascade_cfg=None):
    """Deteksi + (Tahap 2) alur penyeleksi + baca bukti. Dengan worker, pembacaan dikirim ke thread latar.
    ids berisi -1 pada frame yang gerakan kameranya tidak terukur (kotak tidak dicocokkan ke nomor lama).
    det_big: model kedua (lebih besar/presisi), dipanggil cuma pada kotak kandidat lewat refine_candidates
    (lihat src/box_cascade.py) -- bukan tiap frame, supaya biayanya murah rata-rata."""
    ocr_ms = 0.0
    if worker is not None:
        for st, epoch, jreq, ev, decision, reason, key, ms, n in worker.drain():
            if req is not None and jreq is not req:  # dibaca untuk permintaan sebelumnya: putuskan untuk yang sekarang
                decision, reason, key, ev.support = redecide(ev, req, cfg, catalog)
            apply_reading(st, ev, decision, reason, key, ms, n, stats=stats, epoch=epoch)
            ocr_ms = ms
    dets, det_ms = det.detect(frame)
    groups, orphans = group_labels(dets, cfg["grouping"].get("min_label_overlap", 0.5))
    h, w = frame.shape[:2]
    if roi is not None:                       # area rak (kamera diam): benda di luar rak tidak dihitung sebagai kotak
        groups = [g for g in groups if in_roi(g.box.poly, roi, w, h)]
        orphans = [o for o in orphans if in_roi(o.poly, roi, w, h)]
    polys = [g.box.poly for g in groups]
    max_motion = cfg["ocr"].get("max_motion_px", 6)
    motion, shift = (None, 0.0) if motion_est is None else motion_est.update(frame)
    if motion is not None and not consistent(
            [t["poly"] for t in tracker.tracks.values() if t.get("missed", 0) == 0 and not t.get("was_out")],
            polys, motion):
        motion_est.reject()                   # optical flow terkecoh (tangan di depan kamera diam?): frame dilewati
        motion, shift = None, float("inf")
        if stats is not None:
            stats["rejected"] = stats.get("rejected", 0) + 1
    if motion is None and shift == float("inf"):
        # gerakan kamera tidak terukur (buram berat / tertutup): kotak di frame ini TIDAK dicocokkan ke nomor lama
        # (bisa tertukar dengan kotak sebelah) dan tidak di-OCR; frame berikutnya diukur dari frame terakhir yang
        # berhasil. Terlalu lama hilang -> semua nomor dilupakan (bacaan lama tidak dipakai lagi).
        if stats is not None:
            stats["lost"] = stats.get("lost", 0) + 1
            stats["moving"] = stats.get("moving", 0) + 1
            stats["blur"] = float("inf")
        if motion_est.fails > cfg["ocr"].get("max_lost_frames", 15):
            tracker.reset()
            motion_est.reset()
            if session is not None:
                session.lost(now)
            if stats is not None:
                stats["resets"] = stats.get("resets", 0) + 1
        return dets, groups, orphans, [-1] * len(groups), det_ms, ocr_ms
    ids = tracker.update(polys, motion, (w, h))
    # kecepatan gerak kamera dalam px per 1/30 s (waktu sebenarnya antar frame yang diproses) ~ panjang buram gerak
    blur = shift / max((dt or 0.0) * REF_FPS, 1.0)
    if READ_LOG is not None and getattr(tracker, "doubted", None):
        for tid, why in tracker.doubted:      # kenapa nomor kotak diragukan (untuk analisis kunci AMBIL yang batal)
            READ_LOG.write(f"{time.monotonic():.2f},{tid},,{tracker.state(tid).epoch},stale,"
                           f"\"{why}; geser kamera {shift:.1f} px; {len(tracker.doubted)} kotak\"\n")
        READ_LOG.flush()
    if stats is not None:
        stats["blur"] = blur
        stats["moving"] = stats.get("moving", 0) + (blur > max_motion)
    if session is not None:
        session.update(now, ids, polys, tracker, (w, h), motion, blur <= max_motion, frame)
    if targets and blur <= max_motion:
        update_hints(frame, groups, ids, tracker, targets, frame_idx)
    if req is not None and (session is None or session.ocr_allowed):
        n_read = 0
        # kamera bergerak -> frame buram karena gerakan: jangan buang waktu OCR (2-4 s per kotak di CPU laptop)
        still = ocr_all or blur <= max_motion
        if det_big is not None and still:
            # model besar HANYA pada kotak kandidat (pick / CANDIDATE / hint tinggi) -> OBB & size_label lebih
            # presisi tepat di kotak yang sedang diputuskan, tanpa menaikkan biaya tiap frame untuk kotak lain
            groups = refine_candidates(frame, groups, ids, tracker, det_big, session, need,
                                       HINT_MIN, (cascade_cfg or {}).get("pad_frac", 0.25))

        def priority(i):
            st = tracker.state(ids[i])
            c = to_poly(groups[i].box.poly).mean(axis=0)
            off = abs(c[0] / w - 0.5) + abs(c[1] / h - 0.5)        # kotak di tengah layar dulu
            # urutan: kotak AMBIL -> produk/merek sama dengan permintaan tapi ukuran belum pasti (kandidat) -> belum
            # dibaca -> belum terbaca jelas -> sudah pasti (termasuk merek / produk lain: paling akhir)
            rank = {CANDIDATE: 0, READING: 1, CONFIRM: 2}.get(st.stable(need), 3)
            if rank in (1, 2) and st.hint >= HINT_MIN:
                rank = 0.5                        # mirip kotak yang dicari di inventaris: dibaca sebelum yang lain
            pick = session is not None and ids[i] == session.pick
            return (not pick, rank, st.next_ocr_frame > frame_idx, -round(st.hint, 2), round(off, 1),
                    -box_width_px(groups[i].box.poly))

        due = sorted(range(len(groups)), key=priority)
        budget = len(groups) if ocr_all else cfg["ocr"]["max_crops_per_frame"]
        for i in due:
            st = tracker.state(ids[i])
            if not still or n_read >= budget or st.pending or (not ocr_all and st.next_ocr_frame > frame_idx):
                continue
            # kotak AMBIL / kandidat (merek & produk sama, ukuran belum pasti) dibaca ulang lebih sering: bukti keduanya
            # yang ditunggu; kotak lain cukup sesekali
            hot = (session is not None and ids[i] == session.pick) or st.stable(need) == CANDIDATE \
                or (st.hint >= HINT_MIN and st.stable(need) in (READING, CONFIRM))
            st.next_ocr_frame = frame_idx + (cfg["ocr"].get("every_n_frames_candidate", 5) if hot
                                             else cfg["ocr"]["every_n_frames"])
            n_read += 1
            crops = prepare_crops(frame, groups[i], cfg)
            if worker is not None:
                st.pending = worker.submit(st, crops)
                if not st.pending:
                    st.next_ocr_frame = frame_idx + 1      # antrean penuh: coba lagi di frame berikutnya
                continue
            ev, decision, reason, key, ms, n = read_evidence(*crops, reader, cfg, req, catalog)
            apply_reading(st, ev, decision, reason, key, ms, n, stats=stats)
            ocr_ms += ms
    return dets, groups, orphans, ids, det_ms, ocr_ms


def draw(frame, groups, orphans, ids, tracker, req, need, hud, session=None):
    vis = frame.copy()
    ui = hud.get("ui", 1.0)                          # >1 kalau tampilan diperkecil (--view-width): huruf diperbesar
    counts = {s: 0 for s in TAGS}
    if session is not None:
        pick = session.pick
    else:
        pick = pick_fefo({t: tracker.state(t) for t in ids if t >= 0}, need) if req is not None else None
    for g, tid in zip(groups, ids):
        for lab in g.labels:
            draw_poly(vis, lab.poly, COLORS["label"], 1)
        p = to_poly(g.box.poly)
        top = p[np.argmin(p[:, 1])]
        if req is None:
            draw_poly(vis, p, COLORS["stage1"], 2)
            put_label(vis, f"#{tid} box {g.box.conf:.2f}", (top[0], top[1] - 4), COLORS["stage1"], 0.45 * ui)
            continue
        if tid < 0:                                  # gerakan kamera tidak terukur di frame ini: belum dicocokkan
            draw_poly(vis, p, COLORS[READING], 1)
            continue
        st = tracker.state(tid)
        s = st.stable(need)
        counts[s] += 1
        held = tid == pick and session is not None and not session.solid
        if held:                                     # dikunci, tapi bacaan terakhir kurang lengkap: jangan diambil
            draw_poly(vis, p, COLORS[CONFIRM], 4)
            text = f"#{tid} AMBIL? cek ulang"
        elif tid == pick:
            draw_poly(vis, p, COLORS[MATCH], 5)
            text = f"#{tid} AMBIL" + (f" exp {st.expiry.isoformat()}" if st.expiry else "")
        elif st.stale:
            draw_poly(vis, p, COLORS[READING], 1)
            text = f"#{tid} cek ulang" + (" ..." if st.pending else "")
        elif s == READING and st.history:
            # baru sebagian dari `need` bacaan: tampilkan dugaan sementara, belum boleh dipakai
            d = st.history[-1][0]
            draw_poly(vis, p, COLORS[READING], 1)
            guess = {MATCH: "cocok", CANDIDATE: "kandidat", IGNORED: "bukan", CONFIRM: "ragu"}.get(d, "?")
            text = f"#{tid} {guess}? {len(st.history)}/{need}" + (" ..." if st.pending else "")
        else:
            draw_poly(vis, p, COLORS[s], 3 if s in (MATCH, EXPIRED) else 2)
            detail = st.last_reason if s in (CONFIRM, CANDIDATE, EXPIRED) else st.last_text
            text = f"#{tid} {TAGS[s]} {detail[:30]}".strip() + (" ..." if s == READING and st.pending else "")
        label_color = COLORS[CONFIRM] if held else COLORS[MATCH] if tid == pick else COLORS[s]
        if st.hint >= HINT_MIN and tid != pick and s in (READING, CONFIRM, CANDIDATE):
            text += f" | mirip {st.hint_id}"            # petunjuk dari inventaris (belum diverifikasi)
        put_label(vis, text, (top[0], top[1] - 4), label_color, 0.45 * ui)
    for lab in orphans:
        draw_poly(vis, lab.poly, COLORS["label"], 1)
    rate = (f"tampil {hud['real']:.1f} fps" + (f" | masuk {hud['inrate']:.1f} fps" if hud.get("inrate") is not None
                                                  else "") + f" | proses maks {hud['fps']:.0f} fps"
            if hud.get("real") is not None else f"FPS {hud['fps']:.1f}")
    lines = [(f"{rate} | detect {hud['det_ms']:.0f} ms | boxes {len(groups)} | "
              f"size_labels {sum(len(g.labels) for g in groups) + len(orphans)}"
              + (f" | tajam {hud['sharp']:.0f}" if hud.get("sharp") is not None else ""), (40, 40, 40))]
    if hud.get("inv"):
        lines.append((hud["inv"], (120, 60, 0)))
    if session is not None:
        if session.slot is not None and pick not in ids:        # kotak AMBIL tidak terdeteksi: tunjukkan slotnya
            draw_poly(vis, session.slot, COLORS[MATCH], 2)
            q = to_poly(session.slot)
            put_label(vis, f"slot #{pick}", q[np.argmin(q[:, 1])] - (0, 4), COLORS[MATCH], 0.45 * ui)
        if session.watch is not None:
            draw_poly(vis, session.watch[0], COLORS[MATCH], 1)
            q = to_poly(session.watch[0])
            put_label(vis, f"#{session.watch[2]} diambil - slot kosong", q[np.argmin(q[:, 1])] - (0, 4),
                      COLORS[MATCH], 0.45 * ui)
        lines.append((session.message, PHASE_BGR[session.color]))
        moving = hud.get("moving") and session.phase not in (DITEMUKAN, SELESAI, TIDAK_ADA, RAK_KOSONG)
        detail = "KAMERA BERGERAK - tahan diam 2-3 detik supaya kotak bisa dibaca" if moving else session.detail
        if detail:
            lines.append((detail, (40, 40, 40)))
        if session.warning:
            lines.append((session.warning, COLORS[CONFIRM]))
        lines.append((f"hijau AMBIL {int(pick is not None)} | abu {counts[IGNORED]} bukan target | kuning "
                      f"{counts[CONFIRM]} ragu | biru {counts[CANDIDATE]} kandidat | putih {counts[READING]} dibaca"
                      + (f" | ungu {counts[EXPIRED]} kedaluwarsa" if counts[EXPIRED] else "")
                      + f" | OCR {hud['ocr_ms']:.0f} ms", (40, 40, 40)))
    elif req is not None:
        lines.append((f"request: {req} | MATCH {counts[MATCH]} | KANDIDAT {counts[CANDIDATE]} | CONFIRM "
                      f"{counts[CONFIRM]} | EXP {counts[EXPIRED]} | membaca {counts[READING]} | OCR {hud['ocr_ms']:.0f} ms",
                      (40, 40, 40)))
        if pick is not None:
            st = tracker.state(pick)
            lines.append((f"AMBIL #{pick}: {st.last_reason[:40]}" + (f" | GTIN {st.gtin}" if st.gtin else "")
                          + (f" | LOT {st.lot}" if st.lot else "") + (f" | exp {st.expiry}" if st.expiry else ""),
                          COLORS[MATCH]))
        elif counts[CANDIDATE]:
            cand = [t for t in ids if t >= 0 and tracker.state(t).stable(need) == CANDIDATE]
            lines.append(("dekatkan kamera ke " + ", ".join(f"#{t}" for t in cand[:4]) + " (kandidat) untuk verifikasi",
                          (40, 40, 40)))
        else:
            lines.append(("target belum terverifikasi", (40, 40, 40)))
    for k, (line, color) in enumerate(lines):
        put_label(vis, line, (10, int((28 + 26 * k) * ui)), color, 0.6 * ui)
    return vis, counts, pick


def log_transitions(path, req, ids, tracker, need, pick, session_events=()):
    """Catat sekali setiap kali sebuah kotak menjadi MATCH / KEDALUWARSA / AMBIL, plus peristiwa alur penyeleksi
    (JUMLAH KOTAK, RAK KOSONG, AMBIL, TERAMBIL, TIDAK ADA, BATAL AMBIL, PERINGATAN, PERMINTAAN LAGI)."""
    rows = []
    for tid in ids:
        if tid < 0:
            continue
        st = tracker.state(tid)
        s = "AMBIL" if tid == pick else st.stable(need)
        if s in ("AMBIL", MATCH, EXPIRED) and st.logged != s:
            st.logged = s
            rows.append([datetime.now().isoformat(timespec="seconds"), str(req), tid, s, st.last_reason, st.gtin,
                         st.lot, st.expiry.isoformat() if st.expiry else "", st.last_text])
    for _, what, tid, reason, st in session_events:
        rows.append([datetime.now().isoformat(timespec="seconds"), str(req), "" if tid is None else tid, what, reason,
                     st.gtin if st else "", st.lot if st else "",
                     st.expiry.isoformat() if st and st.expiry else "", st.last_text if st else ""])
    if rows:
        new = not path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(LOG_FIELDS)
            w.writerows(rows)
        for r in rows:
            print(f"[log] #{r[2]} {r[3]}: {r[4]} {('GTIN ' + r[5]) if r[5] else ''} {('exp ' + r[7]) if r[7] else ''}")


def _mark_taken(inv, req, st) -> str:
    """Kotak TERAMBIL -> entri inventaris yang cocok ditandai 'diambil'. Pilihan: LOT sama (dari barcode), lalu
    entri yang paling mirip tampilannya (st.hint_id), lalu FEFO. -> box_id atau ''."""
    inv.reload()                          # perubahan dari jendela lain (mis. `rak_inventaris kembali`) tidak tertimpa
    cands = inv.candidates(req)
    if not cands:
        return ""
    lot = st.lot if st is not None else ""
    b = next((c for c in cands if lot and c.lot == lot), None) \
        or next((c for c in cands if st is not None and c.box_id == st.hint_id), None) or cands[0]
    b.status, b.taken_at = STATUS_DIAMBIL, datetime.now().isoformat(timespec="seconds")
    inv.save()
    return b.box_id


def is_photo_source(source) -> bool:
    if not source or re.match(r"^[a-z]+://", source, re.I):
        return False
    p = pathlib.Path(source)
    return p.is_dir() or p.suffix.lower() in IMG_EXT


def iter_source(args, cfg, info):
    """(nama, frame, is_photo). Sumber live memberi frame None selama belum ada frame baru."""
    src = args.source
    if is_photo_source(src):
        p = pathlib.Path(src)
        for f in (sorted(x for x in p.iterdir() if x.suffix.lower() in IMG_EXT) if p.is_dir() else [p]):
            yield f.name, cv2.imread(str(f)), True
        return
    if src and not re.match(r"^[a-z]+://", src, re.I):
        # file video (mis. rekaman --record). Default: seperti kamera live - frame yang lewat selama deteksi/OCR
        # dilompati, jadi hasil sama dengan saat live. --all-frames: semua frame diproses (lebih lambat dari aslinya).
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            sys.exit(f"Video tidak bisa dibuka: {src}")
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        times = load_times(src)                     # waktu asli tiap frame dari --record (kalau ada)
        if times is None:
            fps = args.video_fps or cap.get(cv2.CAP_PROP_FPS) or 30.0
            times = [i / fps for i in range(max(n_frames, 1))]
            print(f"video   : {n_frames} frame, dianggap {fps:.1f} fps"
                  + ("" if args.video_fps else " (tulisan di file; rekaman lama bisa lebih lambat - pakai --video-fps)"))
        else:
            print(f"video   : {len(times)} frame, waktu asli dari {pathlib.Path(src).name}.times.csv "
                  f"({(len(times) - 1) / max(times[-1], 1e-6):.1f} fps nyata)")
        info.update(video_frames=n_frames, video_skipped=0)
        idx, t0 = 0, None
        try:
            while True:
                skip = 0
                if t0 is not None and not args.all_frames:
                    target = times[0] + (time.perf_counter() - t0)       # jam video seiring jam dinding
                    while idx + skip + 1 < len(times) and times[idx + skip + 1] <= target and cap.grab():
                        skip += 1
                ok, frame = cap.read()
                if not ok:
                    break
                if t0 is None:
                    t0 = time.perf_counter()
                idx += skip
                info.update(t=times[min(idx, len(times) - 1)], gap=skip + 1)
                info["video_skipped"] += skip
                idx += 1
                yield None, frame, False
        finally:
            cap.release()
        return
    if src and src.lower().startswith(("http://", "https://")):
        try:
            cap = MjpegReader(src)
        except OSError as e:
            sys.exit(f"Stream tidak bisa dibuka: {src} ({e})\n"
                     "Cek di laptop: scripts/stream_webcam.py jalan dan tunnel `ssh -N -R ...` terbuka.")
    elif src:
        cap = cv2.VideoCapture(src)                              # rtsp:// dan sejenisnya
    else:
        cap, idx, size = open_camera(cfg["camera"], args.camera)
        if cap is None:
            sys.exit("Kamera tidak bisa dibuka. Cek kabel USB webcam, tutup aplikasi lain yang memakai kamera "
                     "(Zoom, Teams, Camera), atau pilih manual: --camera 0 / --camera 1.")
        want = (int(cfg["camera"]["width"]), int(cfg["camera"]["height"]))
        if size and size != want:
            print(f"PERINGATAN: kamera #{idx} memberi {size[0]}x{size[1]}, bukan {want[0]}x{want[1]} "
                  f"(kamera laptop maksimum 640x480 - webcam eksternal terpasang?)")
    if not cap.isOpened():
        sys.exit("Kamera / stream tidak bisa dibuka. Coba --camera 1, tutup aplikasi lain yang memakai "
                 "kamera, atau (Linux) cek `ls /dev/video*`.")
    rec = info.get("recorder")
    live = info["live"] = LatestFrame(cap, on_frame=rec.write if rec is not None else None)
    try:
        while True:
            frame = live.get(0.5)
            if frame is None:
                if live.ended:
                    break
                yield None, None, False
                continue
            if "size" not in info:
                info["size"] = f"{frame.shape[1]}x{frame.shape[0]}"
                print(f"kamera  : {info['size']}")
            info.update(t=live.t, gap=live.gap)
            yield None, frame, False
    finally:
        live.stop()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/box_pipeline_config.yaml")
    ap.add_argument("--weights", default=None, help="override model.weights (.onnx atau .pt)")
    ap.add_argument("--imgsz", default=None,
                    help='override model.imgsz: "960" atau "544,960" (tinggi,lebar; model ekspor: harus sama dengan ekspor)')
    ap.add_argument("--device", default=None, help='override model.device: "intel:gpu.0", "intel:cpu", "cpu", "0"')
    ap.add_argument("--request", default=None,
                    help='mis. "angiolite,2.5,29" (produk,diameter,panjang) atau "permanent sled bag"')
    ap.add_argument("--camera", default=None,
                    help='nomor kamera (0, 1, ...) atau "auto" = resolusi terbesar; default dari config camera.index')
    ap.add_argument("--source", default=None,
                    help="foto / folder foto / video / URL stream (http://127.0.0.1:8090/video.mjpg), pengganti webcam")
    ap.add_argument("--no-window", action="store_true", help="jangan buka jendela (foto: hanya simpan hasil)")
    ap.add_argument("--max-frames", type=int, default=0, help="berhenti setelah N frame (uji / perbandingan)")
    ap.add_argument("--record", default=None,
                    help="simpan frame kamera ASLI (tanpa anotasi) ke video .mp4, untuk diputar ulang dengan --source "
                         "dan untuk data latih jarak dekat")
    ap.add_argument("--all-frames", action="store_true",
                    help="sumber file video: proses SEMUA frame (default: lompati frame seperti kamera live)")
    ap.add_argument("--video-fps", type=float, default=0.0,
                    help="fps sebenarnya dari rekaman tanpa file .times.csv (mis. 10.7), kalau beda dari tulisan di file")
    ap.add_argument("--qty", type=int, default=1, help="berapa kotak yang diminta (default 1)")
    ap.add_argument("--expected", type=int, default=0,
                    help="jumlah kotak di rak kalau diketahui (mis. 20). Kalau terdeteksi lebih sedikit, sistem tidak "
                         "akan menyimpulkan TIDAK ADA")
    ap.add_argument("--inventaris", default=None,
                    help="inventaris rak hasil scripts/rak_inventaris.py (mis. data/rak/inventaris.csv): kotak yang "
                         "dicari diketahui lebih dulu dan dicari lewat ciri tampilannya (tetap ketemu walau dipindah)")
    ap.add_argument("--debug-readings", default=None, help="CSV: satu baris per bacaan OCR (diagnosa)")
    ap.add_argument("--roi", default=None,
                    help='area rak "x1,y1,x2,y2" (pecahan 0-1) atau "none". Tanpa ini, kamera live memakai area yang '
                         f"dipilih dengan tombol a ({ROI_FILE}). Deteksi di luar area diabaikan (kamera tripod)")
    ap.add_argument("--view-width", type=int, default=None,
                    help="lebar jendela tampilan (mis. 1280 lewat Remmina supaya RDP lancar); pemrosesan tetap resolusi "
                         "penuh. Default dari config output.view_width, 0 = ukuran asli")
    ap.add_argument("--rencana", default=None,
                    help="stress test: file rencana dari scripts/stress_plan.py (permintaan bergiliran, jeda ganti "
                         "formasi). Mengganti --request")
    ap.add_argument("--waktu-per-barang", type=float, default=90.0,
                    help="stress test: batas waktu per barang (detik); belum AMBIL -> dilewati dan dicatat")
    ap.add_argument("--log-stress", default=None, help="stress test: CSV hasil per barang (mis. results/stress_01.csv)")
    ap.add_argument("--pakai-ulang-bacaan", action="store_true",
                    help="stress test: bacaan rak dari barang sebelumnya dipakai lagi (seperti operasi biasa). "
                         "Default: tiap barang dibaca dari nol supaya waktu per kotak bisa dibandingkan")
    args = ap.parse_args()
    stress = None
    if args.rencana:
        try:
            stress = StressRun(load_plan(resolve(args.rencana)), args.waktu_per_barang,
                               resolve(args.log_stress) if args.log_stress else None)
        except (OSError, ValueError) as e:
            sys.exit(f"Rencana stress test tidak bisa dibaca: {e}")
        args.request = stress.first_request()
        stress.advance()
        while stress.current.kind != "req":         # jeda sebelum permintaan pertama tidak berarti apa-apa
            stress.advance()
        print(f"stress  : {stress.n_req} permintaan dari {args.rencana}, {args.waktu_per_barang:g} s per barang"
              + (f", hasil -> {args.log_stress}" if args.log_stress else ""))
    if args.record and args.source and not re.match(r"^[a-z]+://", args.source, re.I):
        sys.exit("--record hanya untuk kamera / stream live (sumber file sudah berupa rekaman / foto)")
    if args.debug_readings:
        global READ_LOG
        READ_LOG = open(args.debug_readings, "w", encoding="utf-8")
        READ_LOG.write("t,track,job_epoch,epoch,decision,identity,teks\n")

    cfg = load_config(resolve(args.config))
    req, orders = None, None
    if args.request:
        try:
            items = parse_order(args.request, cfg.get("catalog"))
        except ValueError as e:
            sys.exit(f"Permintaan tidak jelas: {e}")
        if args.qty > 1 and len(items) == 1 and items[0].qty == 1:
            items[0].qty = args.qty
        orders = OrderQueue(items)
        req = orders.start_next().req
        print("permintaan dibaca sebagai: " + "; ".join(describe_request(i.req) + (f" x{i.qty}" if i.qty > 1 else "")
                                                        for i in items), flush=True)
    if args.imgsz:
        size = [int(v) for v in args.imgsz.split(",")]
        cfg["model"]["imgsz"] = size[0] if len(size) == 1 else size
    if args.device:
        cfg["model"]["device"] = args.device
    weights = resolve(args.weights or cfg["model"]["weights"])
    if not weights.exists():
        sys.exit(f"Model tidak ditemukan: {weights}\nSalin best.pt dari server ke {weights} dulu.")
    print(f"memuat model {weights.name} di {cfg['model']['device']} "
          f"(pertama kali di GPU bisa 10-60 detik, tunggu sampai jendela muncul) ...", flush=True)
    t_load = time.perf_counter()
    det = BoxDetector(str(weights), cfg["model"]["task"], cfg["model"]["imgsz"], cfg["model"]["device"],
                      cfg["model"]["conf"], cfg["model"]["iou"], cfg["classes"])
    cfg["model"]["imgsz"], cfg["model"]["device"] = det.imgsz, det.device
    cascade_cfg = cfg["model"].get("cascade") or {}
    det_big = None
    if cascade_cfg.get("enabled"):
        big_weights = resolve(cascade_cfg["weights"])
        if not big_weights.exists():
            sys.exit(f"Model kaskade (besar) tidak ditemukan: {big_weights}")
        det_big = BoxDetector(str(big_weights), cfg["model"]["task"], cascade_cfg.get("imgsz", 1280),
                              cascade_cfg.get("device", cfg["model"]["device"]), cfg["model"]["conf"],
                              cfg["model"]["iou"], cfg["classes"])
        print(f"[kaskade] model besar {big_weights.name} aktif untuk kotak kandidat "
              f"(imgsz {det_big.imgsz}, {det_big.device})")
    photo_mode = is_photo_source(args.source)
    roi = None if photo_mode else load_roi(
        args.roi, live=not args.source or bool(re.match(r"^[a-z]+://", args.source, re.I)))
    n_workers = 1 if photo_mode else max(int(cfg["ocr"].get("workers", 1)), 1)
    # tiap mesin OCR dapat bagian core sendiri (mesin paralel yang semuanya memakai semua core saling berebut)
    ocr_threads = int(cfg["ocr"].get("threads", 0)) or max(1, (os.cpu_count() or 2) // n_workers)
    ocr_cuda = str(cfg["ocr"].get("device", "cpu")).lower() == "cuda"

    def new_reader():
        return SizeTextReader(cfg["ocr"]["engine"], cfg["ocr"]["retry_flip_below"], threads=ocr_threads,
                              use_cuda=ocr_cuda)

    reader = new_reader() if req else None
    catalog = None
    if req is not None:
        cat_path = resolve(cfg.get("verify", {}).get("catalog_csv", "data/gtin_catalog.csv"))
        catalog = Catalog(cat_path)
        n_ok = sum(1 for e in catalog.entries.values() if e.verified)
        print(f"katalog : {cat_path.name}: {len(catalog.entries)} GTIN, {n_ok} terverifikasi"
              + ("" if n_ok else "  (barcode belum bisa dipakai untuk MATCH - verifikasi dengan scripts/enroll_gtin.py)"))
        if catalog.gtins_for(req.product, req.diameter, req.length):
            print(f"          GTIN untuk permintaan ini: {', '.join(catalog.gtins_for(req.product, req.diameter, req.length))}")
    inv = Inventory(resolve(args.inventaris)) if args.inventaris else None
    if inv is not None and not inv.boxes:
        sys.exit(f"Inventaris kosong / tidak ada: {inv.path}. Jalankan scripts/rak_inventaris.py pindai + cek dulu.")

    def inv_targets(r):
        """Kotak inventaris (terverifikasi perawat) untuk permintaan r -> (ciri tampilan, baris info layar)."""
        if inv is None or r is None:
            return [], ""
        inv.reload()
        cands = inv.candidates(r)
        if not cands:
            line = f"INVENTARIS: tidak ada {describe_request(r)} yang terverifikasi di rak - tetap dicari"
        else:
            line = "INVENTARIS: " + ", ".join(f"{b.box_id}" + (f" exp {b.expiry}" if b.expiry else "")
                                              for b in cands[:4]) + f" ({len(cands)} kotak, FEFO: {cands[0].box_id} dulu)"
        print(f"[inventaris] {line[12:]}")
        return [(b.box_id, b.signature()) for b in cands if b.signature() is not None], line

    worker = None
    if req and not photo_mode:
        worker = OcrWorker([reader] + [new_reader() for _ in range(n_workers - 1)], cfg, req, catalog)
    view_width = int(args.view_width if args.view_width is not None else cfg["output"].get("view_width", 0) or 0)
    scan = {"busy": False, "card": None, "until": 0.0, "reader": None}   # tombol b: pindai detail satu kotak

    def start_scan(img):
        """Pindai seluruh gambar (semua barcode / DataMatrix / QR + semua teks) di thread latar."""
        def run():
            try:
                if scan["reader"] is None:
                    scan["reader"] = new_reader()
                cat = catalog or Catalog(resolve(cfg.get("verify", {}).get("catalog_csv", "data/gtin_catalog.csv")))
                card = scan_box(img, scan["reader"], cfg, req, cat)
                out = screens / "pindai"
                out.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(str(out / f"{stamp}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                (out / f"{stamp}.json").write_text(json.dumps(card, ensure_ascii=False, indent=1, default=str),
                                                   encoding="utf-8")
                print("[pindai]")
                for line in card_lines(card):
                    print("  " + line)
                print(f"  -> {out / stamp}.jpg / .json")
                scan.update(card=card, until=time.monotonic() + 15.0)
            except Exception as e:  # noqa: BLE001 - pindai gagal tidak boleh menghentikan video
                print(f"[pindai] gagal: {type(e).__name__}: {e}")
            finally:
                scan["busy"] = False
        scan["busy"] = True
        threading.Thread(target=run, daemon=True).start()
    need = cfg["matching"]["require_consistent"]
    tracker = SimpleTracker(history=cfg["matching"]["history"])
    screens = resolve(cfg["output"]["screens_dir"])
    screens.mkdir(parents=True, exist_ok=True)
    log_path = resolve(cfg["output"]["log_csv"])
    verify_log = resolve(cfg["output"].get("verify_log_csv", "results/verify_log.csv"))

    print(f"model   : {weights.name} ({cfg['model']['task']}, imgsz {cfg['model']['imgsz']}, "
          f"device {cfg['model']['device']}), siap dalam {time.perf_counter() - t_load:.0f} s")
    print(f"mode    : {'Tahap 2 - request ' + str(req) if req else 'Tahap 1 - deteksi generik'}"
          + (f" (OCR + barcode di {n_workers} thread latar x {ocr_threads} core"
             + (", GPU" if ocr_cuda else "") + ")" if worker else ""))

    motion_est = None if photo_mode else MotionEstimator()
    recorder = None if photo_mode or not args.record else VideoRecorder(resolve(args.record),
                                                                        cfg["camera"].get("fps", 30))
    session = None
    if req is not None and not photo_mode:
        session = SelectionSession(req, need, orders.current.qty, cfg.get("selection"), args.expected)
        print(f"alur    : cek rak -> hitung kotak -> cari {orders.current.qty} x {req}"
              + (f" (rak berisi {args.expected} kotak)" if args.expected else "")
              + " | tombol r = permintaan yang sama lagi")
    all_events = []
    new_requests: queue.Queue = queue.Queue()
    if session is not None and sys.stdin is not None and sys.stdin.isatty():
        def read_stdin():
            for line in sys.stdin:
                if line.strip():
                    new_requests.put(line.strip())
        threading.Thread(target=read_stdin, daemon=True).start()
        print("permintaan baru: ketik di terminal ini lalu Enter kapan saja, mis. pilih terumo accuforce diameter "
              "2.75 panjang 20 | beberapa barang: angiolite 2.5 29; accuforce 2.75 20 jumlah 2 | awalan + = "
              "tambah ke antrean | lewati = barang berikutnya")
    targets, inv_line = inv_targets(req)
    t_req, found_times = None, []           # waktu permintaan dimulai; (permintaan, detik sampai AMBIL pertama)
    tidak_ada_since = None

    def switch(item, now, why):
        """Barang berikutnya / permintaan baru. Bacaan rak TIDAK dibuang: bukti tiap kotak diputuskan ulang untuk
        permintaan baru (TrackState.redecide); dulu semua nomor kotak dilupakan dan rak dibaca dari nol lagi."""
        nonlocal req, session, targets, inv_line, t_req, tidak_ada_since
        req = item.req
        if worker is not None:
            worker.req = req
        kept = sum(tracker.state(t).redecide(lambda ev: redecide(ev, req, cfg, catalog)) for t in list(tracker.tracks))
        session = SelectionSession(req, need, item.qty, cfg.get("selection"), args.expected)
        session.events.append((now, why, None, str(item) + f" | bacaan {kept} kotak dipakai ulang", None))
        print(f"[{why.lower()}] {orders.describe()} (bacaan {kept} kotak dipakai ulang)")
        targets, inv_line = inv_targets(req)
        t_req, tidak_ada_since = now, None
    shown = collections.deque(maxlen=60)   # waktu frame yang benar-benar ditampilkan (fps nyata di HUD)
    t_first = None                      # waktu frame pertama: peristiwa dicetak dalam detik sejak mulai
    stats = {"frames": 0, "det_ms": 0.0, "loop_s": 0.0, "ocr_calls": 0, "ocr_ms": 0.0, "reads": 0, "slow": 0,
             "sharp": [], "blur": 0.0, "moving": 0}
    fps, frame_idx, wall0, last_sharp, prev_now = 0.0, 0, None, None, None
    last_counts, pick = {}, None
    info = {"recorder": recorder}
    stress_pause_t, stress_done, stress_started = None, False, False
    stress_tty = sys.stdin is not None and sys.stdin.isatty()

    def stress_next(now):
        """Langkah rencana stress berikutnya: permintaan baru atau jeda. False = rencana selesai."""
        nonlocal orders, stress_pause_t
        while True:
            step = stress.advance()
            if step is None:
                return False
            if step.kind == "pause":
                stress_pause_t = now
                print(f"\n[stress] JEDA: {step.text}\n         "
                      + ("ketik lanjut lalu Enter di terminal ini untuk meneruskan" if stress_tty
                         else "(bukan terminal: diteruskan otomatis 5 s)"), flush=True)
                return True
            try:
                item = parse_order(step.text, cfg.get("catalog"))[0]
            except ValueError as e:
                print(f"[stress] permintaan tidak jelas, dilewati: {step.text} ({e})")
                stress.start_item(now, stats)
                stress.finish_item(now, "TIDAK JELAS", stats, None)
                continue
            orders = OrderQueue([item])
            if not args.pakai_ulang_bacaan:
                tracker.reset()                       # tiap barang dibaca dari nol: waktu per kotak jujur
            switch(orders.start_next(), now, "STRESS")
            stress.start_item(now, stats)
            return True

    def stress_finish(now, result):
        found = stress._found
        teks = tracker.state(found[1]).last_text if found and result == "AMBIL" and found[1] in tracker.tracks else ""
        row = stress.finish_item(now, result, stats, getattr(session, "shelf_n", None), teks)
        print(f"[stress] {row['no']}/{stress.n_req} {row['permintaan']}: {result} {row['detik']} s"
              + (f" #{row['kotak']} ({row['bukti']})" if row["kotak"] != "" else "")
              + (f", cek ulang {row['cek_ulang']}" if row["cek_ulang"] else "")
              + (f", {row['ms_per_bacaan']} ms/bacaan" if row["ms_per_bacaan"] else ""), flush=True)
    for name, frame, is_photo in iter_source(args, cfg, info):
        if frame is None:
            if not args.no_window:
                cv2.waitKey(1)                               # jendela tetap merespons sambil menunggu frame
            continue
        if wall0 is None:
            wall0 = time.perf_counter()
        t0 = time.perf_counter()
        if is_photo:
            tracker = SimpleTracker(history=cfg["matching"]["history"])   # tiap foto berdiri sendiri
        now = info.get("t", time.monotonic())
        while session is not None and not new_requests.empty():
            text = new_requests.get()
            if stress is not None:
                low = text.strip().lower()
                if stress.waiting and low in ("lanjut", "l", "ok", "y", "ya"):
                    tracker.reset()                   # formasi berubah: nomor kotak lama tidak dipakai lagi
                    if motion_est is not None:
                        motion_est.reset()
                    stress_done = not stress_next(now)
                elif not stress.waiting and low in ("lewati", "skip"):
                    stress_finish(now, "DILEWATI")
                    stress_done = not stress_next(now)
                else:
                    print("[stress] selama stress test: ketik lanjut (saat jeda) atau lewati (barang sekarang)")
                continue
            try:
                kind, items = order_command(text, cfg.get("catalog"))
            except ValueError as e:
                print(f"[permintaan] tidak jelas, diabaikan (pesanan tidak berubah): {e}")
                continue
            if kind == "add":
                orders.add(items)
                print(f"[pesanan] ditambahkan | {orders.describe()}")
                continue
            if kind == "skip" and not orders.pending:
                print("[pesanan] tidak ada barang berikutnya - tetap mencari " + str(orders.current))
                continue
            if session.phase == DITEMUKAN and session.pick is not None and t_first is not None:
                # kotak yang sedang dikunci ditinggalkan: TERAMBIL-nya tidak akan tercatat
                all_events.append((now - t_first, "PERMINTAAN DIGANTI", session.pick,
                                   f"AMBIL {req} ditinggalkan sebelum terambil"))
                print(f"[pesanan] AMBIL #{session.pick} {req} ditinggalkan sebelum terambil")
            orders.finish(len(session.taken), "DILEWATI" if kind == "skip" else "DIGANTI")
            if kind == "replace":
                orders.replace(items)
            switch(orders.start_next(), now, "PERMINTAAN BARU")
        t_first = now if t_first is None else t_first
        t_req = now if t_req is None else t_req
        if stress is not None and not stress_started:
            stress.start_item(now, stats)
            stress_started = True
        dt, prev_now = (None if prev_now is None else now - prev_now), now
        dets, groups, orphans, ids, det_ms, ocr_ms = process(
            frame, det, reader, tracker, req, cfg, frame_idx, ocr_all=is_photo, stats=stats, worker=worker,
            catalog=catalog, motion_est=None if is_photo else motion_est, need=1 if is_photo else need,
            dt=dt, session=session, now=now, targets=targets, roi=roi,
            det_big=det_big, cascade_cfg=cfg["model"].get("cascade"))
        sharp = None if is_photo or frame_idx % 5 else sharpness(frame)
        if sharp is not None:
            stats["sharp"].append(sharp)
            last_sharp = sharp
        vis, last_counts, pick = draw(frame, groups, orphans, ids, tracker, req, 1 if is_photo else need,
                                      {"fps": fps, "det_ms": det_ms, "ocr_ms": ocr_ms,
                                       "sharp": None if is_photo else last_sharp,
                                       "moving": stats["blur"] > cfg["ocr"].get("max_motion_px", 6),
                                       "ui": max(1.0, frame.shape[1] / view_width) if view_width else 1.0,
                                       "inv": " | ".join(x for x in (
                                           stress.status(now) if stress is not None else "",
                                           f"{inv_line} | {now - t_req:.1f} s sejak permintaan" if inv_line else "")
                                           if x),
                                       **rates(shown, info)}, session)
        if roi is not None:
            H, W = frame.shape[:2]
            cv2.rectangle(vis, (int(roi[0] * W), int(roi[1] * H)), (int(roi[2] * W), int(roi[3] * H)), (160, 160, 160), 1)
        if req is not None:
            events = session.drain_events() if session is not None else []
            all_events += [(now - t_first, e[1], e[2], e[3]) for e in events]
            log_transitions(verify_log, req, ids, tracker, 1 if is_photo else need,
                            None if session is not None else pick, events)
            for _, what, tid, _, est in events:
                if what == "AMBIL" and (not found_times or found_times[-1][0] != (str(req), t_req)):
                    found_times.append(((str(req), t_req), now - t_req))
                    print(f"[waktu] {req}: kotak #{tid} ditemukan dan terverifikasi {now - t_req:.1f} s sejak permintaan")
                if what == "TERAMBIL" and inv is not None:
                    taken = _mark_taken(inv, req, est)
                    if taken:
                        print(f"[inventaris] {taken} ditandai diambil -> {inv.path.name}")
            if stress is not None and session is not None and not stress_done:
                stress.on_events(events)
                if stress.waiting:
                    if not stress_tty and now - (stress_pause_t or now) >= 5.0:
                        tracker.reset()
                        stress_done = not stress_next(now)
                else:
                    result = stress.check(now, session.phase, session.pick)
                    if result is not None:
                        stress_finish(now, result)
                        stress_done = not stress_next(now)
            if session is not None and orders is not None and orders.pending:
                # barang berikutnya dalam pesanan: sesudah SELESAI, atau TIDAK ADA bertahan NEXT_AFTER_TIDAK_ADA_S
                tidak_ada_since = (tidak_ada_since or now) if session.phase == TIDAK_ADA else None
                if session.phase == SELESAI or (tidak_ada_since is not None
                                                and now - tidak_ada_since >= NEXT_AFTER_TIDAK_ADA_S):
                    orders.finish(len(session.taken), session.phase)
                    switch(orders.start_next(), now, "PERMINTAAN BERIKUTNYA")
            for _, what, tid, _, _ in events:           # bukti audit: gambar beranotasi tiap peristiwa penting
                if what in ("AMBIL", "TERAMBIL", "TIDAK ADA", "RAK KOSONG", "BATAL AMBIL", "PERINGATAN"):
                    ev_dir = screens / "peristiwa"
                    ev_dir.mkdir(parents=True, exist_ok=True)
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(str(ev_dir / f"{stamp}_t{now - t_first:07.1f}_{what.replace(' ', '_')}"
                                              f"{'' if tid is None else '_' + str(tid)}.jpg"), vis,
                                [cv2.IMWRITE_JPEG_QUALITY, 85])
        loop = time.perf_counter() - t0
        fps = 1.0 / loop if fps == 0 else 0.9 * fps + 0.1 * (1.0 / max(loop, 1e-6))
        shown.append(time.perf_counter())
        stats["frames"] += 1
        stats["det_ms"] += det_ms
        stats["loop_s"] += loop
        if loop > SLOW_S and not is_photo:
            stats["slow"] += 1
            print(f"[lambat] frame {frame_idx}: {loop * 1000:.0f} ms (deteksi {det_ms:.0f} ms, "
                  f"OCR/gambar/lainnya {loop * 1000 - det_ms:.0f} ms)")
        frame_idx += 1
        done = (bool(args.max_frames) and stats["frames"] >= args.max_frames) or stress_done

        if is_photo:
            out = screens / f"{pathlib.Path(name).stem}_result.png"
            cv2.imwrite(str(out), vis)
            print(f"{name}: boxes {len(groups)}, size_labels {sum(len(g.labels) for g in groups)}, "
                  f"detect {det_ms:.0f} ms"
                  + (f", MATCH {last_counts[MATCH]}, KANDIDAT {last_counts[CANDIDATE]}, CONFIRM {last_counts[CONFIRM]}"
                     f", EXP {last_counts[EXPIRED]}" + (f", AMBIL #{pick}" if pick is not None else "") if req else "")
                  + f" -> {out}")
        if args.no_window:
            if done:
                break
            continue
        show = vis
        if scan["busy"] or (scan["card"] is not None and time.monotonic() < scan["until"]):
            show = vis.copy()
            ui = max(1.0, show.shape[1] / view_width) if view_width else 1.0
            rows = ["MEMINDAI... tahan kotak diam di depan kamera"] if scan["busy"] else card_lines(scan["card"])
            y0 = show.shape[0] - int((len(rows) * 26 + 12) * ui)
            for k, row in enumerate(rows):
                put_label(show, row[:110], (10, y0 + int((k + 1) * 26 * ui)), (60, 60, 60), 0.6 * ui)
        if view_width and vis.shape[1] > view_width:         # tampilan kecil (Remmina/RDP), simpanan tetap penuh
            s = view_width / vis.shape[1]
            show = cv2.resize(vis, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        cv2.imshow("Box pipeline (q keluar, a area rak, s simpan, b pindai detail, r minta lagi, o baca ulang)", show)
        key = cv2.waitKey(0 if is_photo else 1) & 0xFF
        if key in (ord("q"), 27) or done:
            break
        if key == ord("b") and not scan["busy"] and not is_photo:
            print("[pindai] membaca semua kode + teks pada frame ini ...")
            start_scan(frame.copy())
        if key == ord("a") and not is_photo:
            # pilih area rak (kamera tripod): benda di luar area (kotak kertas, kotak lain) tidak dihitung sebagai kotak
            x, y, rw, rh = cv2.selectROI("Tarik kotak di sekitar RAK lalu Enter (c = batal)", frame, showCrosshair=False)
            cv2.destroyWindow("Tarik kotak di sekitar RAK lalu Enter (c = batal)")
            if rw > 20 and rh > 20:
                H, W = frame.shape[:2]
                roi = (x / W, y / H, (x + rw) / W, (y + rh) / H)
                p = resolve(ROI_FILE)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps({"roi": [round(v, 4) for v in roi]}), encoding="utf-8")
                print(f"[area rak] {', '.join(f'{v:.3f}' for v in roi)} disimpan ke {ROI_FILE}; mulai hitung ulang")
                if session is not None and orders is not None and orders.current is not None:
                    switch(orders.current, now, "AREA RAK BARU")      # jumlah kotak dihitung ulang di area baru
        if key == ord("s"):
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(str(screens / f"{stamp}_raw.png"), frame)
            cv2.imwrite(str(screens / f"{stamp}_vis.png"), vis)
            print(f"[simpan] {screens / (stamp + '_vis.png')}")
        if key == ord("o") and req is not None:
            for tid in ids:
                if tid >= 0:
                    tracker.state(tid).next_ocr_frame = 0
        if key == ord("r") and session is not None:
            session.request_again(now=now)
            print(f"[permintaan] {req} sekali lagi (sisa {session.remaining})")
        if key == ord("l") and req is not None:
            new = not log_path.exists()
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["time", "request", "boxes", "match", "candidate", "confirm", "expired", "ignored",
                                "reading", "pick", "pick_gtin", "pick_expiry"])
                st = tracker.state(pick) if pick is not None else None
                w.writerow([datetime.now().isoformat(timespec="seconds"), str(req), len(groups),
                            last_counts[MATCH], last_counts[CANDIDATE], last_counts[CONFIRM], last_counts[EXPIRED],
                            last_counts[IGNORED], last_counts[READING], pick if pick is not None else "",
                            st.gtin if st else "", st.expiry.isoformat() if st and st.expiry else ""])
            print(f"[catat] {log_path}")
    cv2.destroyAllWindows()
    if recorder is not None:
        recorder.close()

    if stats["frames"] == 0:
        print("\nTIDAK ADA FRAME yang diproses: kamera / stream tidak mengirim gambar.\n"
              "  Cabut-colok USB webcam, tutup program lain yang memakai kamera (Camera, Zoom, Teams, enroll_gtin),\n"
              "  tunggu 5 detik, lalu jalankan lagi. Masih gagal: tambah --camera 0 atau --camera 1.")
        return
    n = max(stats["frames"], 1)
    wall = time.perf_counter() - wall0 if wall0 else 0.0
    print("\nRingkasan")
    print(f"  model / device     : {weights.name} / {cfg['model']['device']} / imgsz {cfg['model']['imgsz']}")
    if "size" in info:
        print(f"  resolusi kamera    : {info['size']}")
    if stats["sharp"]:
        print(f"  ketajaman tengah   : median {statistics.median(stats['sharp']):.0f} "
              f"(maks {max(stats['sharp']):.0f}; < 100 = buram, geser jarak kamera ke kotak)")
    print(f"  frame diproses     : {stats['frames']}")
    print(f"  rata-rata deteksi  : {stats['det_ms'] / n:.1f} ms/frame  (= {1000.0 / max(stats['det_ms'] / n, 1e-6):.1f} FPS deteksi saja)")
    print(f"  rata-rata loop     : {1.0 / max(stats['loop_s'] / n, 1e-6):.1f} FPS (deteksi + OCR + gambar)")
    if not photo_mode and wall > 0:
        print(f"  FPS nyata          : {stats['frames'] / wall:.1f} (frame ditampilkan / {wall:.0f} s, termasuk menunggu kamera)")
        print(f"  iterasi > {SLOW_S:.0f} s      : {stats['slow']}")
    live = info.get("live")
    if live is not None:
        print(f"  frame kamera       : {live.captured} diterima, {live.skipped} dilewati (diganti frame lebih baru)")
        if getattr(live.source, "reconnects", 0):
            print(f"  sambung ulang      : {live.source.reconnects} kali (kamera sempat gagal membaca)")
    if "video_frames" in info:
        print(f"  frame video        : {info['video_frames']}, dilompati {info['video_skipped']} "
              + ("(seperti kamera live)" if not args.all_frames else "(--all-frames)"))
    if not photo_mode:
        print(f"  kamera bergerak    : {stats['moving']} dari {stats['frames']} frame "
              f"({100 * stats['moving'] / n:.0f} %; OCR dilewati pada frame ini)"
              + (f", gerak tak terukur {stats['lost']}" if stats.get("lost") else "")
              + (f" (di antaranya {stats['rejected']} gerak tidak cocok dengan kotak)" if stats.get("rejected") else "")
              + (f", nomor kotak direset {stats['resets']}x" if stats.get("resets") else ""))
    if stats["reads"]:
        print(f"  verifikasi         : {stats['reads']} kali baca kotak, {stats['ocr_calls']} crop OCR, "
              f"rata-rata {stats['ocr_ms'] / stats['reads']:.0f} ms per kotak")
    if recorder is not None and recorder.n:
        print(f"  rekaman            : {recorder.n} frame ({recorder.real_fps:.1f} fps nyata"
              + (f", {recorder.dropped} dibuang karena penulisan kalah cepat" if recorder.dropped else "")
              + f") -> {recorder.path}\n                       waktu tiap frame -> {VideoRecorder.times_path(recorder.path).name}")
    for (r, _), secs in found_times:
        print(f"  waktu sampai AMBIL : {secs:5.1f} s  ({r})")
    if session is not None and orders is not None:
        orders.finish(len(session.taken), session.phase)
        if len(orders.results) > 1 or orders.pending:
            print("  pesanan            : " + " | ".join(f"{r} x{q}: {t} diambil ({o})" for r, q, t, o in orders.results)
                  + (" | belum dicari: " + ", ".join(str(i) for i in orders.pending) if orders.pending else ""))
    if session is not None:
        print(f"  alur penyeleksi    : akhir = {session.phase} | {session.message}")
        print(f"                       {session.shelf_n} kotak di rak, {len(session.taken)} diambil, "
              f"sisa permintaan {max(session.remaining, 0)}")
        for t, what, tid, reason in all_events:
            print(f"    {t:8.1f} s  {what:<16} {'#' + str(tid) if tid is not None else '':<6} {reason}")
    if stress is not None:
        print("\nStress test")
        for line in stress.summary():
            print("  " + line)
        if stress.log_path is not None and stress.rows:
            print(f"  hasil per barang -> {stress.log_path}")


if __name__ == "__main__":
    main()

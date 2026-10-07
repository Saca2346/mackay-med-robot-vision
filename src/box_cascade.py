"""
Kaskade deteksi dua model: model KECIL (cepat) jalan tiap frame seperti biasa lewat
BoxDetector.detect() di scripts/test_webcam_box.py; model BESAR (lebih presisi, imgsz lebih
tinggi) dipanggil lagi lewat modul ini HANYA pada kotak yang sedang jadi kandidat (sedang
diambil, atau cocok merek/produk permintaan, atau sangat mirip kotak yang dicari di
inventaris) -- supaya biaya rata-rata tetap murah tapi presisi naik tepat saat dibutuhkan.

Dipakai kalau config model.cascade.enabled = true (lihat config/box_pipeline_config*.yaml).
"""
from __future__ import annotations

import numpy as np

from src.box_detector import BoxDetector, BoxGroup, group_labels
from src.box_geometry import aabb
from src.box_verifier import CANDIDATE


def candidate_indices(groups: list[BoxGroup], ids, tracker, session, need: int, hint_min: float) -> list[int]:
    """Indeks `groups` yang pantas dideteksi ulang dengan model besar: kotak yang sedang diambil (pick),
    berstatus CANDIDATE (merek/produk cocok dengan permintaan, ukuran belum pasti), atau tampilannya sangat
    mirip kotak yang dicari di inventaris (hint tinggi). Definisi ini sama dengan "hot" yang dipakai
    penjadwalan OCR di test_webcam_box.py, supaya model besar dan OCR fokus ke kotak yang sama."""
    out = []
    for i, tid in enumerate(ids):
        if tid < 0:
            continue
        st = tracker.state(tid)
        pick = session is not None and tid == session.pick
        if pick or st.stable(need) == CANDIDATE or st.hint >= hint_min:
            out.append(i)
    return out


def _crop_region(frame: np.ndarray, poly, pad_frac: float):
    """Area persegi (axis-aligned) di sekitar poligon, plus konteks pad_frac, dipotong ke batas frame.
    -> (crop, x_offset, y_offset) supaya hasil deteksi di crop bisa digeser balik ke koordinat frame penuh."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = aabb(poly)
    pw, ph = (x2 - x1) * pad_frac, (y2 - y1) * pad_frac
    x1, y1 = max(0, int(x1 - pw)), max(0, int(y1 - ph))
    x2, y2 = min(w, int(x2 + pw)), min(h, int(y2 + ph))
    return frame[y1:y2, x1:x2], x1, y1


def refine_group(frame: np.ndarray, group: BoxGroup, det_big: BoxDetector, pad_frac: float) -> BoxGroup:
    """Deteksi ulang SATU kotak kandidat dengan model besar, di crop seputar posisi model kecil, lalu
    kembalikan group yang sudah diperbarui (atau `group` apa adanya kalau tidak ada yang cocok)."""
    crop, x0, y0 = _crop_region(frame, group.box.poly, pad_frac)
    if crop.size == 0:
        return group
    dets, _ = det_big.detect(crop)
    for d in dets:
        d.poly = d.poly + np.array([x0, y0], dtype=np.float32)   # koordinat crop -> koordinat frame penuh

    # TODO(human): cocokkan deteksi "box" di `dets` yang merupakan KOTAK INI (bukan kotak tetangga yang ikut
    # masuk crop karena padding `pad_frac`), lalu pasangkan size_label-nya (group_labels(dets) sudah diimpor
    # dan bisa dipakai langsung di atas daftar `dets` versi crop ini).
    #
    # Yang perlu diputuskan, dengan konsekuensinya masing-masing:
    #   1. Ambang kecocokan: dua poligon OBB jarang identik piksel demi piksel (model besar melihat sudut
    #      sedikit berbeda dari model kecil). poly_iou / poly_iom di src/box_geometry.py bisa dipakai --
    #      IoU terlalu ketat bisa menolak deteksi yang sebenarnya benar; terlalu longgar bisa salah
    #      mengambil kotak TETANGGA yang kebetulan ikut ter-crop.
    #   2. Tidak ada deteksi "box" yang cocok sama sekali (model besar tidak melihatnya di crop ini, mis.
    #      karena crop terlalu sempit atau kotak baru saja bergerak): kembalikan `group` ASLI (hasil model
    #      kecil) -- jangan sampai kotak yang sudah terlacak malah hilang gara-gara percobaan penyempurnaan.
    #   3. Lebih dari satu kandidat cocok (dua "box" di crop dengan IoU mirip terhadap group.box.poly):
    #      kriteria pemilihannya apa -- confidence tertinggi, IoU tertinggi, atau yang lain?
    #
    # Kembalikan BoxGroup baru: BoxGroup(box=<Det box terpilih>, labels=<size_label pasangannya>).

    return group


def refine_candidates(frame, groups: list[BoxGroup], ids, tracker, det_big: BoxDetector, session,
                      need: int, hint_min: float, pad_frac: float = 0.25) -> list[BoxGroup]:
    """Jalankan refine_group HANYA pada kotak kandidat (candidate_indices); kotak lain dipakai apa adanya
    dari model kecil. Dipanggil dari process() di scripts/test_webcam_box.py setelah tracker.update()."""
    out = list(groups)
    for i in candidate_indices(groups, ids, tracker, session, need, hint_min):
        out[i] = refine_group(frame, groups[i], det_big, pad_frac)
    return out

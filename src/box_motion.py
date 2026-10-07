"""
Gerakan kamera antar frame (kamera dipegang tangan / di lengan robot): affine 2x3 dari frame acuan ke frame
ini, dari optical flow titik sudut pada gambar kecil (lebar 480 px, sekitar 5-10 ms di CPU).

Dipakai untuk dua hal:
  - tracker: posisi kotak lama digeser dulu sebelum dicocokkan (ID kotak tidak berganti saat kamera bergerak)
  - OCR: frame yang diambil saat kamera bergerak cepat buram karena gerakan -> tidak dikirim ke OCR

Pengaman (kotak identik berjejer = pola berulang, gerakan bisa "melompat" satu kotak):
  - tiap titik dilacak maju lalu mundur; titik yang tidak kembali ke tempat asalnya dibuang
  - kalau gerakan gagal diukur, frame acuan TIDAK diganti: frame berikutnya diukur dari frame terakhir yang
    berhasil, jadi rantai posisi kotak tidak terputus. `fails` = berapa frame berturut-turut gagal.
"""
from __future__ import annotations

import cv2
import numpy as np

FB_MAX_PX = 1.0          # galat maju-mundur maksimum (px gambar kecil)
MIN_POINTS = 12
MIN_INLIERS = 10


class MotionEstimator:
    def __init__(self, width: int = 480):
        self.width = width
        self.prev = None
        self.fails = 0

    def reset(self) -> None:
        self.prev, self.fails = None, 0

    def update(self, frame: np.ndarray):
        """-> (affine 2x3 dari frame acuan ke frame ini dalam piksel frame asli, atau None;
               pergeseran median dalam piksel frame asli, inf kalau gagal diukur)."""
        s = self.width / frame.shape[1]
        g = cv2.cvtColor(cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        prev = self.prev
        if prev is None or prev.shape != g.shape:
            self.prev, self.fails = g, 0
            return None, 0.0                                   # frame pertama: belum ada pembanding
        M, shift = self._estimate(prev, g)
        if M is None:
            self.fails += 1                                    # acuan tetap frame terakhir yang berhasil
            return None, float("inf")
        self._undo = (prev, self.fails)
        self.prev, self.fails = g, 0
        M[:, 2] /= s                                           # translasi ke piksel frame asli
        return M, shift / s

    def reject(self) -> None:
        """Gerakan yang baru diukur ternyata tidak masuk akal (lihat consistent): dianggap gagal diukur, acuan
        kembali ke frame sebelumnya."""
        prev, fails = getattr(self, "_undo", (self.prev, self.fails))
        self.prev, self.fails = prev, fails + 1

    @staticmethod
    def _estimate(prev, g):
        pts = cv2.goodFeaturesToTrack(prev, maxCorners=300, qualityLevel=0.01, minDistance=8)
        if pts is None or len(pts) < MIN_POINTS:
            return None, 0.0
        lk = dict(winSize=(21, 21), maxLevel=3)
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, pts, None, **lk)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(g, prev, nxt, None, **lk)
        fb = np.linalg.norm((pts - back).reshape(-1, 2), axis=1)
        ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < FB_MAX_PX)
        a, b = pts[ok].reshape(-1, 2), nxt[ok].reshape(-1, 2)
        if len(a) < MIN_POINTS:
            return None, 0.0
        M, inl = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=2.0)
        if M is None or inl is None or inl.sum() < MIN_INLIERS:
            return None, 0.0
        keep = inl.ravel() == 1
        return M.astype(np.float32).copy(), float(np.median(np.linalg.norm(b[keep] - a[keep], axis=1)))


def _centers_sides(polys):
    q = np.asarray([np.asarray(p, dtype=np.float32).reshape(-1, 2)[:4] for p in polys], dtype=np.float32)
    sides = np.linalg.norm(np.roll(q, -1, axis=1) - q, axis=2)            # (n, 4)
    return q.mean(axis=1), np.maximum(sides.min(axis=1), 1.0)


def consistent(prev_polys, polys, motion, tol: float = 0.35, min_boxes: int = 5) -> bool:
    """Gerakan kamera terukur cocok dengan kotak-kotak yang terdeteksi? Kotak yang terlihat di frame sebelumnya
    digeser dengan `motion` lalu dicari deteksinya (pusat dalam `tol` x lebar kotak). Tangan yang bergerak di depan
    kamera DIAM bisa membuat optical flow salah sekali (dugaan dari uji 30-09: banyak nomor kotak diragukan
    sekaligus saat tangan meraih kotak). Salah = dengan gerakan itu jauh lebih sedikit kotak yang ketemu daripada
    tanpa gerakan -> frame ini dianggap "gerakan tidak terukur" (kotak tidak dicocokkan ke nomor lama). "Tanpa
    gerakan" TIDAK pernah dipakai sebagai gantinya (kotak berjejer bisa tampak cocok bergeser satu kotak)."""
    if motion is None or len(prev_polys) < min_boxes or len(polys) < min_boxes:
        return True
    pc, ps = _centers_sides(prev_polys)
    dc, ds = _centers_sides(polys)
    A = np.asarray(motion, dtype=np.float32)
    moved = pc @ A[:, :2].T + A[:, 2]
    if float(np.median(np.linalg.norm(moved - pc, axis=1) / ps)) < tol:
        return True                                                        # gerakan kecil: keduanya sama saja

    def fit(c):
        # bagian DETEKSI yang punya kotak lama di dekatnya (kotak yang tertutup tangan tidak ikut dinilai)
        d = np.linalg.norm(dc[:, None, :] - c[None, :, :], axis=2).min(axis=1)
        return float(np.mean(d <= tol * ds))

    f_move, f_still = fit(moved), fit(pc)
    return not (f_still >= 0.6 and f_move < 0.5 * f_still)

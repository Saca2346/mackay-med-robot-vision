"""
Membuka kamera untuk pipeline box: pilih kamera, resolusi, backend dan format.

Laptop Lenovo + webcam eksternal HP w300 (diuji 2026-09-27):
  kamera 0 = HP w300      1920x1080 30 fps (MSMF + MJPG), DSHOW hanya 20 fps
  kamera 1 = kamera laptop 640x480 maksimum
Nomor kamera di Windows bisa berubah (colok / cabut USB, urutan boot), jadi index "auto" memilih kamera dengan
resolusi terbesar, yaitu webcam eksternal kalau terpasang.
"""
from __future__ import annotations

import os
import time

# Akselerasi hardware MSMF (DXVA) untuk dekode MJPG: tanpa itu FPS sama (7.3) tapi kamera siap 11 s lebih cepat,
# dan error "can't grab frame -2147024882" (0x8007000E) yang muncul sesekali saat OpenVINO memakai iGPU yang sama
# dihindari. Harus diset sebelum VideoCapture MSMF pertama dibuka.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

BACKENDS = {"msmf": cv2.CAP_MSMF, "dshow": cv2.CAP_DSHOW, "any": cv2.CAP_ANY}


def _open(index: int, width: int, height: int, fps: float, backend: str, fourcc: str | None):
    cap = cv2.VideoCapture(index, BACKENDS.get(str(backend).lower(), cv2.CAP_ANY))
    if not cap.isOpened():
        return None
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:
        cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def list_cameras(max_index: int = 4, backend: str = "msmf") -> list[tuple[int, int, int]]:
    """[(index, lebar maks, tinggi maks)] untuk kamera yang bisa dibuka (minta 3840x2160, lihat yang didapat).
    Berhenti di nomor pertama yang tidak ada (nomor kamera Windows berurutan), jadi tidak ada peringatan OpenCV
    untuk nomor kosong."""
    out = []
    log = getattr(getattr(cv2, "utils", None), "logging", None)
    old = log.getLogLevel() if log else None
    if log:
        log.setLogLevel(log.LOG_LEVEL_ERROR)          # "can't be used to capture by index" untuk nomor kosong
    try:
        for i in range(max_index):
            cap = _open(i, 3840, 2160, 0, backend, "MJPG")
            if cap is None:
                break
            ok, frame = cap.read()
            cap.release()
            if ok and frame is not None:
                out.append((i, frame.shape[1], frame.shape[0]))
    finally:
        if log:
            log.setLogLevel(old)
    return out


class ReconnectingCamera:
    """cv2.VideoCapture yang menyambung ulang sendiri. Driver webcam USB di Windows sesekali gagal (MSMF
    "can't grab frame", kabel longgar, kamera sempat dipakai program lain): read() yang gagal membuka ulang kamera
    yang sama (maks `retries` kali berturut-turut, jeda makin lama) dan setelah 2 kali gagal mencoba backend DSHOW
    (20 fps, tapi driver berbeda). Antarmuka sama dengan VideoCapture: read(), isOpened(), release()."""

    def __init__(self, index: int, cam_cfg: dict, retries: int = 6):
        self.index, self.cfg, self.retries = index, dict(cam_cfg), retries
        self.backend = str(cam_cfg.get("backend", "msmf")).lower()
        self.reconnects = 0
        self.cap = self._make(self.backend)

    def _make(self, backend):
        c = self.cfg
        return _open(self.index, int(c.get("width", 1920)), int(c.get("height", 1080)), float(c.get("fps", 30)),
                     backend, c.get("fourcc", "MJPG"))

    def isOpened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def read(self):
        for attempt in range(self.retries + 1):
            if self.cap is not None:
                ok, frame = self.cap.read()
                if ok and frame is not None:
                    return True, frame
            if attempt == self.retries:
                break
            if self.cap is not None:
                self.cap.release()
            time.sleep(min(0.5 * (attempt + 1), 2.0))
            backend = self.backend if attempt < 2 else ("dshow" if self.backend == "msmf" else "msmf")
            self.cap = self._make(backend)
            self.reconnects += 1
            print(f"[kamera] gagal membaca, sambung ulang #{self.index} lewat {backend} "
                  f"(percobaan {attempt + 1}/{self.retries})", flush=True)
        return False, None

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def open_camera(cam_cfg: dict, index_override=None):
    """-> (cam, index, (lebar, tinggi) yang benar-benar didapat); cam = ReconnectingCamera.
    cam_cfg = bagian 'camera' dari config."""
    index = cam_cfg.get("index", "auto") if index_override is None else index_override
    backend = cam_cfg.get("backend", "msmf")
    if str(index).lower() == "auto":
        cams = list_cameras(backend=backend)
        if not cams:
            return None, None, None
        index = max(cams, key=lambda c: (c[1] * c[2], -c[0]))[0]
        print("kamera  : " + ", ".join(f"#{i} {w}x{h}" for i, w, h in cams) + f" -> pakai #{index}")
    cam = ReconnectingCamera(int(index), cam_cfg)
    ok, frame = cam.read()
    if not ok:
        cam.release()
        return None, int(index), None
    return cam, int(index), (frame.shape[1], frame.shape[0])


def sharpness(frame: np.ndarray, frac: float = 0.5) -> float:
    """Ketajaman bagian tengah frame (varians Laplacian pada gambar abu-abu lebar 640 px).
    Webcam fokus tetap: geser jarak kotak sampai angka ini paling tinggi."""
    h, w = frame.shape[:2]
    y0, x0 = int(h * (1 - frac) / 2), int(w * (1 - frac) / 2)
    c = frame[y0:h - y0, x0:w - x0]
    s = 640 / max(1, c.shape[1])
    g = cv2.cvtColor(cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())

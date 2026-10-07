"""
Sumber video live untuk scripts/test_webcam_box.py.

  LatestFrame : baca kamera / stream di thread terpisah dan simpan HANYA frame terbaru. Kalau deteksi
                lebih lambat dari kamera, frame lama dilewati sehingga video tidak makin tertinggal.
  VideoRecorder: rekam frame kamera asli di thread sendiri + waktu tiap frame (untuk putar ulang yang setia).
  MjpegReader : baca stream MJPEG lewat HTTP (dari scripts/stream_webcam.py), misalnya webcam laptop
                yang dikirim ke server GPU lewat tunnel SSH. Antarmukanya sama dengan cv2.VideoCapture.
"""
from __future__ import annotations

import pathlib
import re
import threading
import time
import urllib.request

import cv2
import numpy as np


class MjpegReader:
    """Stream multipart MJPEG (tiap bagian punya Content-Length) -> read() -> (ok, frame BGR)."""

    def __init__(self, url: str, timeout: float = 10.0):
        self.url = url
        self._resp = urllib.request.urlopen(url, timeout=timeout)
        self._buf = b""

    def isOpened(self) -> bool:
        return self._resp is not None

    def _fill(self) -> bool:
        read1 = getattr(self._resp, "read1", None)
        chunk = read1(65536) if read1 else self._resp.read(4096)
        if not chunk:
            return False
        self._buf += chunk
        return True

    def read(self):
        try:
            while True:
                i = self._buf.find(b"\r\n\r\n")
                if i < 0:
                    if not self._fill():
                        return False, None
                    continue
                m = re.search(rb"content-length:\s*(\d+)", self._buf[:i], re.I)
                start = i + 4
                if not m:                       # pembatas tanpa header panjang: lewati
                    self._buf = self._buf[start:]
                    continue
                n = int(m.group(1))
                while len(self._buf) < start + n:
                    if not self._fill():
                        return False, None
                jpg, self._buf = self._buf[start:start + n], self._buf[start + n:]
                frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    return True, frame
        except OSError:
            return False, None

    def release(self) -> None:
        if self._resp is not None:
            self._resp.close()
            self._resp = None


class VideoRecorder:
    """Simpan frame kamera asli ke .mp4 di thread sendiri (membaca kamera tidak ikut melambat) + waktu tiap frame
    ke <file>.times.csv. Kamera sering memberi fps lebih rendah dari yang diminta (cahaya redup, CPU sibuk), jadi
    putar ulang memakai waktu sebenarnya dari file ini, bukan fps yang tertulis di .mp4.
    Antrean penuh (penulisan kalah cepat) -> frame dibuang dan dihitung di `dropped`."""

    def __init__(self, path, fps: float = 30.0, max_queue: int = 60):
        import queue
        self.path, self.fps = pathlib.Path(path), float(fps)
        self.n = self.dropped = 0
        self._q = queue.Queue(maxsize=max_queue)
        self._times: list[float] = []
        self._writer = None
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @staticmethod
    def times_path(path) -> pathlib.Path:
        return pathlib.Path(str(path) + ".times.csv")

    def write(self, frame, t: float | None = None) -> None:
        try:
            self._q.put_nowait((frame, time.monotonic() if t is None else t))
        except Exception:  # noqa: BLE001 - queue.Full
            self.dropped += 1

    def _loop(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            frame, t = item
            if self._writer is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps,
                                               (frame.shape[1], frame.shape[0]))
                if not self._writer.isOpened():
                    print(f"[rekam] tidak bisa menulis {self.path}")
                    return
            self._writer.write(frame)
            self._times.append(t)
            self.n += 1

    def close(self) -> None:
        self._q.put(None)
        self._thread.join(timeout=30.0)
        if self._writer is not None:
            self._writer.release()
        if self._times:
            t0 = self._times[0]
            with open(self.times_path(self.path), "w", encoding="utf-8") as f:
                f.write("frame,t\n")
                f.writelines(f"{i},{t - t0:.4f}\n" for i, t in enumerate(self._times))

    @property
    def real_fps(self) -> float:
        return (len(self._times) - 1) / (self._times[-1] - self._times[0]) if len(self._times) > 1 else 0.0


def load_times(video_path) -> list[float] | None:
    """Waktu tiap frame (detik) dari <video>.times.csv buatan VideoRecorder, atau None kalau tidak ada."""
    p = VideoRecorder.times_path(video_path)
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        next(f)
        return [float(line.split(",")[1]) for line in f if line.strip()]


class LatestFrame:
    """Thread pembaca: get() mengembalikan frame terbaru yang belum pernah diambil; `t` = waktu frame itu diterima
    (time.monotonic), dipakai untuk mengukur kecepatan gerak kamera dengan waktu sebenarnya."""

    def __init__(self, source, on_frame=None):
        self.source = source
        self.on_frame = on_frame   # on_frame(frame, t) untuk SETIAP frame kamera (mis. rekam); harus cepat
        self.captured = 0          # frame yang diterima dari kamera / stream
        self.skipped = 0           # frame yang tertimpa sebelum sempat diproses
        self.ended = False
        self.gap = 1               # jumlah frame kamera sejak frame yang diambil sebelumnya (1 = tidak ada yang dilewati)
        self.t = 0.0               # waktu frame terakhir yang diambil lewat get()
        self._frame, self._frame_t, self._seq, self._taken, self._stop = None, 0.0, 0, 0, False
        self._cond = threading.Condition()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        try:
            while not self._stop:
                ok, frame = self.source.read()
                t = time.monotonic()
                if ok and self.on_frame is not None:
                    self.on_frame(frame, t)
                with self._cond:
                    if not ok:
                        break
                    if self._seq != self._taken:
                        self.skipped += 1
                    self._frame, self._frame_t, self._seq = frame, t, self._seq + 1
                    self.captured += 1
                    self._cond.notify_all()
        finally:
            with self._cond:
                self.ended = True
                self._cond.notify_all()
            self.source.release()

    def get(self, timeout: float = 0.5):
        """Frame baru, atau None kalau belum ada frame baru dalam `timeout` detik / sumber sudah habis."""
        with self._cond:
            self._cond.wait_for(lambda: self.ended or self._seq != self._taken, timeout)
            if self._seq == self._taken:
                return None
            self.gap = max(self._seq - self._taken, 1)
            self._taken = self._seq
            self.t = self._frame_t
            return self._frame

    def stop(self) -> None:
        self._stop = True
        self._thread.join(timeout=2.0)

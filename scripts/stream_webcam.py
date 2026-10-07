#!/usr/bin/env python3
"""
Kirim webcam laptop sebagai stream MJPEG supaya deteksi bisa diuji di server GPU (dilihat lewat Remmina).

Stream hanya dibuka di 127.0.0.1 (tidak terlihat dari jaringan); server membacanya lewat tunnel SSH. Port 8090 di
server sudah dipakai layanan lain, jadi di server stream ini muncul di port 18090:

  Laptop, jendela cmd 1 :  venv\\Scripts\\python.exe scripts\\stream_webcam.py
  Laptop, jendela cmd 2 :  ssh -N -R 18090:127.0.0.1:8090 Zen@140.113.149.94     (isi password, biarkan terbuka)
  Server (Remmina)      :  python scripts/test_webcam_box.py --config config/box_pipeline_config_server.yaml \\
                               --source http://127.0.0.1:18090/video.mjpg --request "angiolite,2.5,29"

Kamera hanya bisa dipakai satu program: tutup stream ini sebelum menjalankan test_webcam_box.py di laptop.
Default: webcam eksternal (kamera resolusi terbesar) 1920x1080, maksimum 15 fps. Jaringan lambat:
--width 1280 --height 720 atau --quality 60.
Uji tanpa kamera: --source <folder foto> (diputar berulang pada --fps), atau --video <rekaman.mp4> (rekaman
dari --record diputar dengan waktu aslinya; rekaman lama tanpa .times.csv: tambah --video-fps 10.7).
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.box_camera import open_camera  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


class FrameSource:
    """Ambil frame di thread sendiri; simpan hanya JPEG terbaru (klien lambat melewatkan frame, tidak menumpuk)."""

    def __init__(self, args):
        self.args = args
        self.jpeg, self.seq, self.captured, self.bytes = None, 0, 0, 0
        self.cond = threading.Condition()
        self.error = None
        threading.Thread(target=self._loop, daemon=True).start()

    def _frames(self):
        a = self.args
        if a.video:                                    # rekaman diputar dengan waktu aslinya, seperti kamera live
            from src.box_stream import load_times
            cap = cv2.VideoCapture(a.video)
            if not cap.isOpened():
                raise RuntimeError(f"video tidak bisa dibuka: {a.video}")
            times = load_times(a.video)
            fps = a.video_fps or cap.get(cv2.CAP_PROP_FPS) or 30.0
            print(f"[stream] video {pathlib.Path(a.video).name}: "
                  + ("waktu asli dari .times.csv" if times else f"{fps:.1f} fps"), flush=True)
            i = 0
            if a.start:                                # mulai dari detik ke-N rekaman
                i = next((k for k, tk in enumerate(times) if tk >= a.start), 0) if times else int(a.start * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            t0 = time.perf_counter() - (times[i] if times and i < len(times) else i / fps)
            last = -1.0
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError("video selesai")
                    t = times[i] if times and i < len(times) else i / fps
                    i += 1
                    wait = t0 + t - time.perf_counter()
                    if wait > 0:
                        time.sleep(wait)
                    if a.fps and t - last < 1.0 / a.fps - 1e-3:   # batasi fps kiriman seperti kamera
                        continue
                    last = t
                    yield frame
            finally:
                cap.release()
        if a.source:
            files = sorted(p for p in pathlib.Path(a.source).iterdir() if p.suffix.lower() in IMG_EXT)
            imgs = [cv2.resize(cv2.imread(str(p)), (a.width, a.height)) for p in files]
            if not imgs:
                raise RuntimeError(f"tidak ada foto di {a.source}")
            k = 0
            while True:
                time.sleep(1.0 / a.fps)
                yield imgs[k % len(imgs)]
                k += 1
        cap, idx, size = open_camera({"index": a.camera, "width": a.width, "height": a.height, "fps": 30,
                                      "backend": "msmf", "fourcc": "MJPG"})
        if cap is None:
            raise RuntimeError("kamera tidak bisa dibuka (cek USB webcam, tutup aplikasi lain yang memakai kamera, "
                               "atau --camera 0 / 1)")
        print(f"[stream] kamera #{idx}: {size[0]}x{size[1]}" if size else f"[stream] kamera #{idx}", flush=True)
        gap = 1.0 / a.fps if a.fps else 0.0
        last = 0.0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("kamera berhenti mengirim frame")
                now = time.perf_counter()
                if now - last < gap:                   # batasi fps kiriman supaya jaringan tidak penuh
                    continue
                last = now
                yield frame
        finally:
            cap.release()

    def _loop(self):
        try:
            for frame in self._frames():
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.args.quality])
                if not ok:
                    continue
                with self.cond:
                    self.jpeg, self.seq = buf.tobytes(), self.seq + 1
                    self.captured += 1
                    self.bytes += len(self.jpeg)
                    self.cond.notify_all()
        except Exception as e:  # noqa: BLE001
            self.error = str(e)
            print(f"[stream] BERHENTI: {e}", flush=True)
            with self.cond:
                self.cond.notify_all()


def make_handler(src: FrameSource, clients: list):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):          # jangan cetak satu baris per request
            pass

        def do_GET(self):
            if not self.path.startswith("/video"):
                self.send_error(404, "pakai /video.mjpg")
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            clients.append(self.client_address)
            print(f"[stream] klien tersambung {self.client_address[0]}", flush=True)
            seen = 0
            try:
                while src.error is None:
                    with src.cond:
                        src.cond.wait_for(lambda: src.seq != seen or src.error is not None, timeout=5)
                        if src.seq == seen:
                            continue
                        jpg, seen = src.jpeg, src.seq
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
            finally:
                clients.remove(self.client_address)
                print(f"[stream] klien terputus {self.client_address[0]}", flush=True)

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", default="auto", help='"auto" = resolusi terbesar (webcam eksternal), atau 0 / 1')
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--quality", type=int, default=80, help="kualitas JPEG 1-100 (lebih kecil = lebih ringan di jaringan)")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--source", default=None, help="folder foto sebagai pengganti kamera (uji)")
    ap.add_argument("--video", default=None,
                    help="rekaman (mis. results/uji_tahap2.mp4) diputar dengan waktu aslinya sebagai pengganti kamera")
    ap.add_argument("--video-fps", type=float, default=0.0,
                    help="fps sebenarnya rekaman lama tanpa .times.csv (mis. 10.7)")
    ap.add_argument("--start", type=float, default=0.0, help="--video: mulai dari detik ke-N rekaman")
    ap.add_argument("--fps", type=float, default=15.0,
                    help="fps maksimum yang dikirim (1080p JPEG 80 x 15 fps = sekitar 35 Mbit/s)")
    args = ap.parse_args()

    src = FrameSource(args)
    clients: list = []
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(src, clients))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[stream] http://127.0.0.1:{args.port}/video.mjpg  ({args.width}x{args.height}, JPEG {args.quality})")
    print("[stream] Ctrl+C untuk berhenti")
    last_n, last_b, last_t = 0, 0, time.perf_counter()
    try:
        while src.error is None:
            time.sleep(5)
            t = time.perf_counter()
            n, b = src.captured - last_n, src.bytes - last_b
            print(f"[stream] {n / (t - last_t):.1f} FPS kamera, {b / max(n, 1) / 1024:.0f} KB/frame, "
                  f"{b * 8 / (t - last_t) / 1e6:.1f} Mbit/s, klien {len(clients)}", flush=True)
            last_n, last_b, last_t = src.captured, src.bytes, t
    except KeyboardInterrupt:
        pass
    server.shutdown()
    sys.exit(1 if src.error else 0)


if __name__ == "__main__":
    main()

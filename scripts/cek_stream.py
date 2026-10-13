#!/usr/bin/env python3
"""Ambil beberapa frame dari stream laptop dan laporkan: resolusi, fps, ketajaman, kecerahan.

  python scripts/cek_stream.py [URL] [--simpan results/frame_cek.jpg]

Keluar 0 bila gambar masuk dan layak; 1 bila tidak ada gambar; 2 bila ada gambar tetapi buram/gelap/kecil
(tetap bisa dilanjutkan, tetapi bacaan teks ukuran kemungkinan gagal)."""
import argparse
import sys
import time

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("url", nargs="?", default="http://127.0.0.1:18090/video.mjpg")
ap.add_argument("--simpan", default="results/frame_cek.jpg")
ap.add_argument("--frames", type=int, default=20)
args = ap.parse_args()

cap = cv2.VideoCapture(args.url)
frames, t0 = [], time.time()
while len(frames) < args.frames and time.time() - t0 < 15:
    ok, f = cap.read()
    if ok and f is not None:
        frames.append(f)
cap.release()
if not frames:
    print("GAGAL: tidak ada gambar dari stream (tunnel/stream laptop belum benar).")
    sys.exit(1)

fps = len(frames) / max(time.time() - t0, 1e-3)
last = frames[-1]
h, w = last.shape[:2]
gray = cv2.cvtColor(last, cv2.COLOR_BGR2GRAY)
sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
bright = float(gray.mean())
cv2.imwrite(args.simpan, last)
print(f"Gambar masuk: {w}x{h}, ~{fps:.1f} fps, ketajaman {sharp:.0f}, kecerahan {bright:.0f}/255")
print(f"Frame contoh disimpan: {args.simpan}  (buka untuk memastikan yang terlihat memang rak)")
warn = []
if min(w, h) < 720:
    warn.append(f"resolusi rendah ({w}x{h}); teks ukuran sulit terbaca. Jalankan stream tanpa --width/--height kecil")
if sharp < 100:
    warn.append(f"gambar buram (ketajaman {sharp:.0f} < 100): dekatkan kamera / perbaiki fokus; saat rak penuh angka bisa lebih rendah")
if bright < 60 or bright > 215:
    warn.append(f"pencahayaan kurang baik (kecerahan {bright:.0f}): tambah/kurangi cahaya, hindari pantulan")
for m in warn:
    print("PERINGATAN:", m)
sys.exit(2 if warn else 0)

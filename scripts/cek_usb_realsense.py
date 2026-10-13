#!/usr/bin/env python3
"""
Pantau jenis USB kamera Intel RealSense selama beberapa menit, untuk mencari port / kabel yang benar-benar USB 3.
Cabut dan colok kameranya ke port lain selagi skrip ini jalan; tiap perubahan dicetak. Yang dicari: "USB 3.x".
D455 di USB 2.1 hanya menawarkan profil warna kecil dan di laptop ini frame tidak sampai (diuji 2026-10-08).

  venv\\Scripts\\python.exe scripts\\cek_usb_realsense.py            # 120 detik
  venv\\Scripts\\python.exe scripts\\cek_usb_realsense.py --detik 300
"""
from __future__ import annotations

import argparse
import time


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detik", type=int, default=120)
    a = ap.parse_args()
    try:
        import pyrealsense2 as rs
    except ImportError:
        raise SystemExit("pyrealsense2 belum terpasang: venv\\Scripts\\python.exe -m pip install pyrealsense2")
    print(f"memantau {a.detik} detik. Pindahkan kamera antar port / ganti kabel; Ctrl+C untuk berhenti.", flush=True)
    last = None
    for _ in range(a.detik):
        devs = list(rs.context().devices)         # context baru tiap putaran: yang lama tidak melihat colok-cabut
        now = [(d.get_info(rs.camera_info.name), d.get_info(rs.camera_info.usb_type_descriptor)) for d in devs]
        if now != last:
            if not now:
                print("[tidak ada RealSense]", flush=True)
            for name, usb in now:
                print(f"{name}: USB {usb}   " + ("<-- BAGUS" if str(usb).startswith("3") else "<-- masih USB 2"),
                      flush=True)
            last = now
        time.sleep(1.0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Demo permintaan teks (meeting 05-10, demo 12-10): ketik brand + produk + spec -> produk, spec, status, posisi
rak / level, dan pesan JSON untuk software TM. Tanpa kamera; inventaris hanya DIBACA, tidak diubah.

  python scripts/demo_request.py                                  # interaktif, ketik permintaan, kosong = keluar
  python scripts/demo_request.py --sekali "ivascular angiolite 2.5 x 29"
  python scripts/demo_request.py --sekali "nagomi 3/24" --json    # hanya pesan TM (satu baris JSON)
  python scripts/demo_request.py --sekali "angiolite 2.5 x 29" --kamera   # SIAP -> lanjut verifikasi kamera

Contoh permintaan: "angiolite,2.5,29"  "terumo accuforce 2,75 x 20"  "nagomi diameter 3 panjang 24"
Tiap permintaan dicatat ke results/demo_log.jsonl (satu baris JSON = satu pesan TM).
Status SIAP berarti robot boleh MENUJU posisi; kotaknya tetap diverifikasi kamera:
  python scripts/test_webcam_box.py --inventaris data/rak/inventaris.csv --request "angiolite,2.5,29"
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.box_demo import SIAP, append_log, load_positions, lookup, to_text, to_tm_message  # noqa: E402
from src.box_inventory import Inventory  # noqa: E402


def run_camera(res, args) -> None:
    """Status SIAP -> serahkan ke verifikasi kamera (test_webcam_box.py). Konfirmasi akhir tetap di dokter."""
    req = res.request
    size = "" if req.diameter is None else f",{req.diameter:g},{req.length:g}"
    cmd = [sys.executable, str(ROOT / "scripts" / "test_webcam_box.py"), "--inventaris", args.inventaris,
           "--request", f"{req.product}{size}"] + args.kamera_arg
    print("Kamera     : " + " ".join(cmd[1:]))
    subprocess.run(cmd, cwd=ROOT)


def run_one(text: str, args, inv: Inventory, positions) -> None:
    inv.reload()                                   # inventaris bisa diubah di jendela lain (TERAMBIL / kembali)
    res = lookup(text, inv, positions)
    append_log(args.log, res)
    if args.json:
        print(json.dumps(to_tm_message(res), ensure_ascii=False))
        return
    print("-" * 64)
    print(to_text(res))
    print("Pesan TM   : " + json.dumps(to_tm_message(res), ensure_ascii=False))
    if res.status == SIAP:
        req = res.request
        size = "" if req.diameter is None else f",{req.diameter:g},{req.length:g}"
        print(f"Verifikasi : python scripts/test_webcam_box.py --inventaris {args.inventaris} "
              f"--request \"{req.product}{size}\"")
        if args.kamera:
            run_camera(res, args)
    elif args.kamera:
        print("Kamera     : tidak dijalankan (status bukan SIAP)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inventaris", default="data/rak/inventaris.csv")
    ap.add_argument("--posisi", default="data/rak/posisi.csv")
    ap.add_argument("--log", default="results/demo_log.jsonl")
    ap.add_argument("--sekali", help="satu permintaan lalu keluar")
    ap.add_argument("--kamera", action="store_true",
                    help="status SIAP -> lanjut ke verifikasi kamera (test_webcam_box.py)")
    ap.add_argument("--kamera-arg", nargs=argparse.REMAINDER, default=[],
                    help="argumen tambahan untuk test_webcam_box.py, taruh PALING AKHIR, mis. --config config/box_pipeline_config_server.yaml --source http://127.0.0.1:18090/video")
    ap.add_argument("--json", action="store_true", help="cetak hanya pesan TM (JSON satu baris)")
    args = ap.parse_args()

    inv_path, pos_path = pathlib.Path(args.inventaris), pathlib.Path(args.posisi)
    if not inv_path.exists():
        sys.exit(f"Inventaris tidak ada: {inv_path} (jalankan dari folder proyek, atau isi --inventaris)")
    inv = Inventory(inv_path)
    positions = load_positions(pos_path)
    if not positions:
        print(f"PERINGATAN: tabel posisi kosong / tidak ada ({pos_path}); semua hasil akan 'posisi tidak diketahui'")
    elif any(p.sumber == "contoh" for p in positions.values()) and not args.json:
        print("Catatan: tabel posisi masih CONTOH (rak/level belum dari tim).")

    if args.sekali:
        run_one(args.sekali, args, inv, positions)
        return
    print(f"Inventaris {inv_path} ({len(inv.boxes)} kotak), posisi {pos_path} ({len(positions)} baris).")
    print("Ketik permintaan (brand + produk + spec). Enter kosong = keluar.")
    while True:
        try:
            text = input("\nPermintaan > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            break
        run_one(text, args, inv, positions)


if __name__ == "__main__":
    main()

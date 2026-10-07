"""
Buat file rencana stress test (dipakai: test_webcam_box.py --rencana FILE).

Tiga bagian, sesuai protokol uji:
  1. semua kotak di inventaris bergiliran (20 kotak x 90 s = 30 menit)
  2. N kali ganti formasi: tiap kali K kotak diubah (pindah slot / balik 180 / berbaring / pindah ke sisi lain
     rak / dimiringkan), lalu K kotak itu diminta lagi
  3. permintaan acak dari semua kotak (posisi apa adanya)
Barang tidak diambil, cukup AMBIL, jadi inventaris tidak berubah.

  venv\\Scripts\\python.exe scripts\\stress_plan.py --out results\\rencana_stress.txt
  venv\\Scripts\\python.exe scripts\\stress_plan.py --formasi 5 --per-formasi 5 --acak 10 --seed 7 --out ...
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.box_inventory import Inventory  # noqa: E402
from src.config import resolve  # noqa: E402

ACTIONS = ["pindah ke slot lain", "balik 180 derajat", "baringkan", "pindah ke sisi lain rak", "miringkan"]


def request_text(b) -> str:
    return b.product if b.diameter is None else f"{b.product} {b.diameter:g} {b.length:g}"


def build_plan(boxes, formasi: int, per_formasi: int, acak: int, rng: random.Random) -> list[str]:
    boxes = sorted(boxes, key=lambda b: (b.order or 999, b.box_id))
    lines = [f"# Bagian 1: {len(boxes)} kotak bergiliran"]
    lines += [f"{request_text(b)}   # {b.box_id}" for b in boxes]
    for k in range(1, formasi + 1):
        chosen = rng.sample(boxes, min(per_formasi, len(boxes)))
        moves = ", ".join(f"{b.box_id} {request_text(b)}: {ACTIONS[(i + k) % len(ACTIONS)]}"
                          for i, b in enumerate(chosen))
        lines += ["", f"# Bagian 2: formasi {k}",
                  f"! formasi {k}: ubah {len(chosen)} kotak -> {moves}. Tangan keluar dari rak, kamera diam"]
        lines += [f"{request_text(b)}   # {b.box_id} ({ACTIONS[(i + k) % len(ACTIONS)]})" for i, b in enumerate(chosen)]
    if acak:
        lines += ["", "# Bagian 3: permintaan acak", "! permintaan acak: biarkan posisi kotak apa adanya"]
        lines += [f"{request_text(b)}   # {b.box_id}" for b in (rng.choice(boxes) for _ in range(acak))]
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inventaris", default="data/rak/inventaris.csv")
    ap.add_argument("--formasi", type=int, default=5, help="berapa kali formasi diubah")
    ap.add_argument("--per-formasi", type=int, default=5, help="berapa kotak diubah tiap formasi")
    ap.add_argument("--acak", type=int, default=10, help="jumlah permintaan acak di bagian akhir")
    ap.add_argument("--seed", type=int, default=None, help="angka acak tetap (rencana bisa diulang persis)")
    ap.add_argument("--out", default="results/rencana_stress.txt")
    args = ap.parse_args()

    inv = Inventory(resolve(args.inventaris))
    boxes = [b for b in inv.boxes.values() if b.product]
    if not boxes:
        sys.exit(f"Inventaris kosong: {inv.path}")
    taken = [b.box_id for b in boxes if b.status == "diambil"]
    lines = build_plan(boxes, args.formasi, args.per_formasi, args.acak, random.Random(args.seed))
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    n_req = sum(1 for ln in lines if ln and not ln.startswith(("#", "!")))
    print(f"rencana: {n_req} permintaan, {args.formasi} jeda ganti formasi -> {out}")
    if taken:
        print(f"PERHATIAN: {', '.join(taken)} berstatus diambil di inventaris. Kalau kotaknya ada di rak, pulihkan dulu:\n"
              f"  venv\\Scripts\\python.exe scripts\\rak_inventaris.py kembali {','.join(taken)}")


if __name__ == "__main__":
    main()

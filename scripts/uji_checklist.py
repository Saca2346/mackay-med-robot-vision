#!/usr/bin/env python3
"""
Uji otomatis 20 permintaan di docs/CHECKLIST_DEMO.md (bagian teks). Membandingkan hasil `lookup()` dengan yang
diharapkan, mencetak tabel LULUS / GAGAL, dan menyimpan laporan. Inventaris hanya dibaca.

  python scripts/uji_checklist.py                    # 20 uji teks + pemeriksaan awal
  python scripts/uji_checklist.py --kamera 5         # uji teks lalu kamera untuk nomor 5 (kotak fisik di depan kamera)
  python scripts/uji_checklist.py --kamera 5 --kamera-arg --config config/box_pipeline_config_server.yaml --source URL

Keluaran: results/uji_checklist_<waktu>.csv dan .md (tempel ke laporan). Kode keluar 0 hanya kalau tidak ada GAGAL.
Uji kamera nomor 1-16 dijalankan satu per satu: kotak fisik harus ada di rak, hasilnya (MATCH / PERLU KONFIRMASI,
dan apakah kotaknya benar) Anda isi sendiri di kolom Hasil kamera pada checklist. Aturan lulus: tidak boleh ada MATCH
pada kotak yang salah.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.box_demo import HABIS, INVALID, SIAP, load_positions, lookup  # noqa: E402
from src.box_inventory import STATUS_DIAMBIL, Inventory  # noqa: E402

# (permintaan, status yang diharapkan, kotak yang diharapkan; "" = tidak ada)
UJI = [
    ("angiolite 2.5 x 29", SIAP, "K15"), ("xperience pro 2.5 x 15", SIAP, "K06"),
    ("essential pro 3 x 40", SIAP, "K04"), ("accuforce 2.5 x 12", SIAP, "K07"),
    ("accuforce 2.75 x 20", SIAP, "K16"), ("accuforce 3.5 x 15", SIAP, "K21"),
    ("ryurei 2 x 10", SIAP, "K09"), ("nagomi 2.75 x 33", SIAP, "K13"),
    ("nagomi 3 x 24", SIAP, "K18"), ("nc emerge 5.5 x 12", SIAP, "K08"),
    ("conqueror 3.5 x 15", SIAP, "K11"), ("conqueror 4 x 8", SIAP, "K12"),
    ("sapphire 3 x 15", SIAP, "K14"), ("scoreflex 3 x 15", SIAP, "K19"),
    ("agent 2.75 x 30", SIAP, "K20"), ("permanent sled bag", SIAP, "K03"),
    ("angiolite 4 x 19", HABIS, ""), ("accuforce 2.75 x 15", HABIS, ""),
    ("terumo xperience 2.5 x 15", INVALID, ""), ("essential pro 3", INVALID, ""),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inventaris", default="data/rak/inventaris.csv")
    ap.add_argument("--posisi", default="data/rak/posisi.csv")
    ap.add_argument("--kamera", type=int, metavar="N", help="setelah uji teks, jalankan kamera untuk nomor N (1-16)")
    ap.add_argument("--kamera-arg", nargs=argparse.REMAINDER, default=[], help="argumen tambahan test_webcam_box.py")
    args = ap.parse_args()

    inv_path = pathlib.Path(args.inventaris)
    if not inv_path.exists():
        sys.exit(f"Inventory missing: {inv_path} (run from the project folder)")
    inv, positions = Inventory(inv_path), load_positions(args.posisi)

    print(f"Inventory : {inv_path} ({len(inv.boxes)} boxes: {inv.counts()})")
    print(f"Positions : {args.posisi} ({len(positions)} rows"
          + (", EXAMPLE" if any(p.sumber == "contoh" for p in positions.values()) else "") + ")")
    k10 = inv.boxes.get("K10")
    k10_taken = k10 is not None and k10.status == STATUS_DIAMBIL
    if not k10_taken:
        print("NOTE      : K10 is not 'taken' -> test 17 skipped (READY is correct then, not OUT OF STOCK)")

    rows, gagal = [], 0
    print(f"\n{'#':>2}  {'request':<28}{'expected':<24}{'result':<24}{'box':<6} verdict")
    for i, (q, want, want_box) in enumerate(UJI, 1):
        if i == 17 and not k10_taken:
            rows.append((i, q, want, "-", "", "SKIPPED"))
            print(f"{i:>2}  {q:<28}{want + ' ' + want_box:<24}{'-':<24}{'':<6} SKIPPED")
            continue
        res = lookup(q, inv, positions)
        got_box = res.box.box_id if res.box else ""
        ok = res.status == want and got_box == want_box
        # keselamatan: permintaan yang harus ditolak tidak boleh pernah menghasilkan kotak
        if want != SIAP and res.box is not None:
            ok = False
        gagal += not ok
        verdict = "PASS" if ok else "FAIL"
        rows.append((i, q, want, res.status, got_box, verdict))
        print(f"{i:>2}  {q:<28}{(want + ' ' + want_box).strip():<24}{res.status:<24}{got_box:<6} {verdict}")

    lulus = sum(r[5] == "PASS" for r in rows)
    print(f"\nPASS {lulus}, FAIL {gagal}, SKIPPED {sum(r[5] == 'SKIPPED' for r in rows)} of {len(UJI)}")

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    with open(out / f"uji_checklist_{stamp}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["no", "request", "expected", "result", "box", "verdict"])
        w.writerows(rows)
    with open(out / f"uji_checklist_{stamp}.md", "w", encoding="utf-8") as f:
        f.write(f"# Checklist test {stamp}\n\nPASS {lulus}, FAIL {gagal} of {len(UJI)}\n\n"
                "| # | Request | Expected | Result | Box | Verdict |\n| --- | --- | --- | --- | --- | --- |\n")
        for r in rows:
            f.write(f"| {r[0]} | `{r[1]}` | {r[2]} | {r[3]} | {r[4]} | {r[5]} |\n")
    print(f"Laporan   : results/uji_checklist_{stamp}.csv / .md")

    if args.kamera:
        # 17 (angiolite 4 x 19, K10) hanya bisa diuji kamera kalau K10 memang ada di rak; pasangan K15/K10 = dua angiolite
        # dengan tampilan hampir sama, kasus "mirip" paling berisiko
        last = 16 if k10_taken else 17
        if not 1 <= args.kamera <= last:
            sys.exit(f"--kamera only for numbers 1-{last} (items on the shelf)")
        q, _, box = UJI[args.kamera - 1]
        if args.kamera == 17:
            box = "K10"
        print(f"\nCamera test {args.kamera}: {q} (must be box {box}). Record the result in the checklist; "
              "a MATCH on any other box = FAIL.")
        cmd = [sys.executable, str(ROOT / "scripts" / "demo_request.py"), "--sekali", q, "--kamera",
               "--inventaris", args.inventaris, "--posisi", args.posisi]
        if args.kamera_arg:
            cmd += ["--kamera-arg"] + args.kamera_arg
        subprocess.run(cmd, cwd=ROOT)
    sys.exit(1 if gagal else 0)


if __name__ == "__main__":
    main()

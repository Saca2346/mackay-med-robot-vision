"""
Cari kode REF yang HAMPIR dikenali di log bacaan (--debug-readings): teks yang mirip awalan REF suatu produk tetapi
ditolak src/box_ref.parse_refs. Ini cara menemukan salah baca huruf yang baru (08-10: "SCCDSR" terbaca "SCCDSA" ->
REF ditolak -> bacaan dianggap kosong -> AMBIL batal). Jalankan setelah setiap uji kamera.

  python scripts/cek_ref_hampir.py results/bukti_laporan_20261008/bacaan_uji1 results/bukti_laporan_20261008/bacaan_uji2

Hasil "HAMPIR" yang muncul berkali-kali dengan pola yang sama = kandidat perbaikan pola di src/box_ref.py. Pola baru
harus diuji dulu (tests/test_box_ref.py): kotak saudara (mis. K10 angiolite 4 x 19) tidak boleh berubah ukuran.
"""
from __future__ import annotations

import argparse
import collections
import csv
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.box_ref import parse_refs  # noqa: E402

# awalan REF yang longgar (huruf yang sering tertukar boleh), panjang sesuai kode asli
LOOSE = {
    "angiolite": r"[S5][CO0]{1,3}[DO0][S5][A-Z0-9][A-Z0-9]{11}",
    "xperience pro": r"[B8]C[PR][RP][A-Z0-9]{2}[NM][A-Z0-9]{9}",
    "essential pro": r"[B8]CD[PR]R[A-Z0-9]{2}[NM][A-Z0-9]{9}",
    "accuforce": r"D[CO0][-_.]?R[MN][A-Z0-9]{4}",
    "ryurei": r"D[CO0][-_.]?RR[A-Z0-9]{4}",
    "ultimaster nagomi": r"D[EF][-_.]?R[S5][A-Z0-9]{4}",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="file log --debug-readings (kolom teks)")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()
    hit, miss = collections.Counter(), collections.Counter()
    for f in args.files:
        with open(f, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                t = re.sub(r"\s+", "", (r.get("teks") or "").upper())
                got = {h.product for h in parse_refs(t)}
                for prod, pat in LOOSE.items():
                    for m in re.finditer(pat, t):
                        (hit if prod in got else miss)[(prod, m.group(0))] += 1
    total = sum(hit.values()) + sum(miss.values())
    print(f"REF dikenali: {sum(hit.values())} dari {total} teks mirip REF"
          + (f" ({100 * sum(hit.values()) / total:.1f}%)" if total else ""))
    per = collections.Counter()
    for (p, _), n in hit.items():
        per[p] += n
    for p, n in per.most_common():
        print(f"  {p:18s} {n}")
    if not miss:
        print("Tidak ada REF yang hampir dikenali.")
        return
    print(f"HAMPIR (mirip REF tapi ditolak), {args.top} terbanyak:")
    for (p, s), n in miss.most_common(args.top):
        print(f"  {n:3d}x  {p:18s} {s}")


if __name__ == "__main__":
    main()

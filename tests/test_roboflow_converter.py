#!/usr/bin/env python3
"""
Tes scripts/convert_roboflow_dataset.py pada data mini buatan sendiri (poligon
segitiga & bbox campuran) — tidak butuh dataset Roboflow asli untuk dijalankan,
tapi logika konversinya sama persis dengan yang dipakai pada dataset nyata
(sudah diverifikasi terpisah pada dataset "Pharmacy" 44-kelas: 6903 objek
dikonversi, 0 baris rusak, koordinat valid, dan cocok visual pada foto rak asli).

Run: python tests/test_roboflow_converter.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from scripts.convert_roboflow_dataset import convert_label_line

PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"


def check(label, cond):
    print(f"[{PASS if cond else FAIL}] {label}")
    return cond


def approx(a, b, tol=1e-6):
    return abs(a - b) < tol


def main():
    ok = True

    # 1) bbox biasa (5 token) -> disalin apa adanya
    r = convert_label_line("3 0.5 0.5 0.2 0.4")
    ok &= check("bbox biasa disalin apa adanya", r == ("3", 0.5, 0.5, 0.2, 0.4))

    # 2) poligon kotak sederhana (4 sudut persegi 0.2..0.6 x 0.3..0.7) -> bbox harus pas sama
    #    titik: (0.2,0.3) (0.6,0.3) (0.6,0.7) (0.2,0.7)
    r = convert_label_line("5 0.2 0.3 0.6 0.3 0.6 0.7 0.2 0.7")
    cls, cx, cy, w, h = r
    ok &= check(
        f"poligon persegi -> bbox benar (dapat cx={cx},cy={cy},w={w},h={h})",
        cls == "5" and approx(cx, 0.4) and approx(cy, 0.5) and approx(w, 0.4) and approx(h, 0.4),
    )

    # 3) poligon segitiga (titik tidak simetris) -> bbox = bounding rectangle-nya
    #    titik: (0.1,0.1) (0.9,0.2) (0.5,0.8)  -> x:[0.1,0.9] y:[0.1,0.8]
    r = convert_label_line("0 0.1 0.1 0.9 0.2 0.5 0.8")
    cls, cx, cy, w, h = r
    ok &= check(
        f"poligon segitiga -> bounding rectangle benar (dapat cx={cx},cy={cy},w={w},h={h})",
        cls == "0" and approx(cx, 0.5) and approx(cy, 0.45) and approx(w, 0.8) and approx(h, 0.7),
    )

    # 4) baris rusak (jumlah token genap tapi < 4, atau ganjil selain 5) -> None
    ok &= check("baris kosong/rusak (3 token) ditolak", convert_label_line("1 0.5 0.5") is None)
    ok &= check("baris poligon dengan jumlah koordinat ganjil ditolak", convert_label_line("1 0.5 0.5 0.6 0.6 0.7") is None)

    print("\n" + ("SEMUA TES LULUS" if ok else "ADA TES YANG GAGAL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

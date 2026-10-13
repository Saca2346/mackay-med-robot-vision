#!/usr/bin/env python3
"""Ringkas log uji kamera (hasil --log-stress): python scripts/ringkas_uji_kamera.py results/uji_kamera_XXXX.csv"""
import csv
import re
import sys

rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
print(f"{'no':>2}  {'permintaan':42} {'diharapkan':10} {'hasil':12} {'detik':>5}  bukti")
for r in rows:
    want = (re.search(r"K\d+", r.get("catatan", "")) or [""])[0]
    print(f"{r['no']:>2}  {r['permintaan']:42} {want:10} {r['hasil']:12} {r['detik']:>5}  {r['bukti']}")
n = len(rows)
ambil = sum(r["hasil"] == "PICK" for r in rows)
print(f"\nAMBIL {ambil}/{n}; sisanya WAKTU HABIS / TIDAK ADA / TIDAK JELAS = sistem tidak menebak (aman).")
print("PENTING: 'AMBIL' berarti sistem memilih satu kotak. Pastikan di rekaman/jendela video bahwa kotak bertanda")
print("hijau tebal itu memang kotak yang diharapkan (kolom 'diharapkan'). Salah kotak = GAGAL KERAS.")

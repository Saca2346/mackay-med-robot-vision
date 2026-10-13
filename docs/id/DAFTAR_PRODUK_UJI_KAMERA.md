# Daftar produk untuk uji kamera / gladi rekaman

Pegang kotak fisik yang sesuai di depan kamera, ucapkan/ketik **permintaan**, cocokkan hasil dengan **kotak
yang diharapkan**. Status harus **SIAP**. Jalankan satu per satu dengan:

```bash
python scripts/demo_request.py --sekali "PERMINTAAN" --kamera
# atau nomor checklist:  python scripts/uji_checklist.py --kamera N   (N = 1-16)
```

Aturan lulus: tidak boleh ada MATCH pada kotak yang salah. Ragu = PERLU KONFIRMASI (itu normal, bukan gagal).

## A. Barang ada di rak (uji kamera, harus SIAP)

| # | Permintaan (ucapkan/ketik) | Brand | Produk | Ukuran (diameter x panjang mm) | Kedaluwarsa | Kotak | Posisi |
|---|---|---|---|---|---|---|---|
| 1 | `ivascular angiolite 2.5 x 29` | iVascular | Angiolite | 2.5 x 29 | – | K15 | R1 / level 1 / kolom 15 |
| 2 | `ivascular xperience pro 2.5 x 15` | iVascular | Xperience Pro | 2.5 x 15 | – | K06 | R1 / level 1 / kolom 6 |
| 3 | `ivascular essential pro 3 x 40` | iVascular | Essential Pro | 3 x 40 | – | K04 | R1 / level 1 / kolom 4 |
| 4 | `terumo accuforce 2.5 x 12` | Terumo | Accuforce | 2.5 x 12 | – | K07 | R1 / level 1 / kolom 7 |
| 5 | `terumo accuforce 2.75 x 20` | Terumo | Accuforce | 2.75 x 20 | – | K16 | R1 / level 1 / kolom 16 |
| 6 | `terumo accuforce 3.5 x 15` | Terumo | Accuforce | 3.5 x 15 | – | K21 | R1 / level 1 / kolom 21 |
| 7 | `terumo ryurei 2 x 10` | Terumo | Ryurei | 2 x 10 | – | K09 | R1 / level 1 / kolom 9 |
| 8 | `terumo ultimaster nagomi 2.75 x 33` | Terumo | Ultimaster Nagomi | 2.75 x 33 | – | K13 | R1 / level 1 / kolom 13 |
| 9 | `terumo ultimaster nagomi 3 x 24` | Terumo | Ultimaster Nagomi | 3 x 24 | 2028-01-31 | K18 | R1 / level 1 / kolom 18 |
| 10 | `boston scientific nc emerge 5.5 x 12` | Boston Scientific | NC Emerge | 5.5 x 12 | 2028-01-03 | K08 | R1 / level 1 / kolom 8 |
| 11 | `apt medical conqueror nc pro 3.5 x 15` | APT Medical | Conqueror NC Pro | 3.5 x 15 | – | K11 | R1 / level 1 / kolom 11 |
| 12 | `apt medical conqueror nc pro 4 x 8` | APT Medical | Conqueror NC Pro | 4 x 8 | – | K12 | R1 / level 1 / kolom 12 |
| 13 | `orbusneich sapphire 3 x 15` | OrbusNeich | Sapphire | 3 x 15 | – | K14 | R1 / level 1 / kolom 14 |
| 14 | `orbusneich scoreflex 3 x 15` | OrbusNeich | Scoreflex | 3 x 15 | – | K19 | R1 / level 1 / kolom 19 |
| 15 | `boston scientific agent 2.75 x 30` | Boston Scientific | Agent | 2.75 x 30 | – | K20 | R1 / level 1 / kolom 20 |
| 16 | `boston scientific permanent sled bag` | Boston Scientific | Permanent Sled Bag | tanpa ukuran | 2029-03-10 | K03 | R1 / level 1 / kolom 3 |

Catatan uji #16: ada 4 kotak Permanent Sled Bag di inventaris (K01, K03, K05, K17) dengan kedaluwarsa
berbeda-beda. Sistem harus memilih **K03** karena kedaluwarsanya paling awal (2029-03-10) — ini uji aturan
FEFO, bukan sekadar "kotak pertama yang ditemukan".

## B. Harus ditolak atau tidak ditemukan (teks saja, tanpa kamera)

| # | Permintaan | Hasil yang diharapkan | Sebab |
|---|---|---|---|
| 17 | `ivascular angiolite 4 x 19` | TIDAK ADA | sudah diambil (K10) — hanya berlaku selama K10 masih berstatus `diambil` |
| 18 | `terumo accuforce 2.75 x 15` | TIDAK ADA | mirip #5 dan #6, kombinasi ini tidak ada di inventaris |
| 19 | `terumo xperience 2.5 x 15` | ULANGI | Xperience Pro bukan produk Terumo (punya iVascular) |
| 20 | `ivascular essential pro 3` | ULANGI | panjang belum ditulis |

## Posisi rak — PENTING

Semua posisi di atas (`R1 / level 1 / kolom N`) berasal dari `data/rak/posisi.csv`, yang **masih data
CONTOH** (belum dikonfirmasi tim). Robot/TM jangan diarahkan secara fisik ke koordinat ini sampai posisi
rak/level sebenarnya dikonfirmasi — lihat pertanyaan #1 di `docs/CHECKLIST_DEMO.md`.

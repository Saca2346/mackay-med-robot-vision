# Protokol data rak: dua rak (rak 1 → rak 2 → kedua rak bersamaan)

Versi Inggris: `docs/DATA_RAK_PROTOCOL.md` bagian 6. Syarat kamera, kalibrasi, dan aturan privasi tetap sama seperti
protokol satu rak (`docs/DATA_RAK_PROTOCOL.md` bagian 1-2): USB 3, tripod, lebar kotak >= 40 px (28-40 masih boleh),
sharpness >= 300, target di tengah gambar, tanpa data pasien / wajah / layar pribadi.

## 1. Urutan kerja

| Fase | Apa yang direkam | Kenapa urutannya begini |
| --- | --- | --- |
| **1. Rak 1** | semua sesi rak 1 | kondisi yang sudah dikenal (20 kotak, uji 8-14 Okt) - pembanding |
| **2. Rak 2** | sesi yang sama di rak 2 | rak baru: latar, cahaya, susunan berbeda |
| **3. Kedua rak bersamaan** | seluruh area dua rak, lalu menyusuri dari rak 1 ke rak 2 | kondisi robot nanti: kamera melihat lebih dari satu rak |

Jangan mencampur fase dalam satu rekaman. Selesaikan fase 1 dulu, baru pindahkan tripod ke rak 2.

## 2. Nama sesi

`r1_` = rak 1, `r2_` = rak 2, `r12_` = kedua rak. Lalu nama sesi seperti biasa.

| Sesi | Rak 1 | Rak 2 | Kedua rak | Kamera | Prioritas |
| --- | --- | --- | --- | --- | --- |
| jauh, seluruh rak | `r1_s1_far` | `r2_s1_far` | `r12_s1_far` | semua kotak terlihat | **minggu ini** |
| sedang, lebar kotak >= 40 px | `r1_s2_mid` | `r2_s2_mid` | `r12_s2_mid` | kiri lalu kanan; untuk r12: menyusuri dari rak 1 ke rak 2 | **minggu ini** |
| dekat, 3-5 kotak | `r1_s3_close` | `r2_s3_close` | - | digeser menyusuri rak | minggu depan |
| susunan diubah | `r1_s4_order` | `r2_s4_order` | - | seperti s2, referensi `r1_s2_mid` / `r2_s2_mid` | minggu depan |
| kasus sulit | `r1_s5_hard` | `r2_s5_hard` | - | kotak miring / tidur / terbalik / slot kosong / tangan | minggu depan |
| sudut 15-20 derajat | `r1_s6_angle` | `r2_s6_angle` | - | dari kiri dan dari kanan | minggu depan |
| **uji (tidak untuk training)** | - | - | `r12_s7_test` | seperti r12_s2, susunan baru di kedua rak | terakhir |

Minggu ini cukup **6 sesi**: `r1_s1_far`, `r1_s2_mid`, `r2_s1_far`, `r2_s2_mid`, `r12_s1_far`, `r12_s2_mid` (masing-masing
1-2 menit rekaman + kalibrasi, total sekitar 1 jam).

## 3. Langkah per sesi

1. `TEST_REALSENSE.bat` -> **D**.
2. **Session name**: mis. `r1_s1_far`.
3. **Boxes in view**: jumlah kotak yang terlihat di kamera (rak 1 saja, rak 2 saja, atau total kedua rak). Enter = 20.
4. **Reference session**: Enter (kosong), kecuali `s4_order` (isi `r1_s2_mid` atau `r2_s2_mid`).
5. Kalibrasi tampil -> atur sampai tidak ada FAIL -> rekam: diam 2-3 detik tiap posisi, geser pelan, `q` untuk berhenti.

## 4. Catatan per fase

**Sebelum fase 1 dan fase 2 (per rak):**
- Foto rak dengan HP dari depan (acuan susunan) dan catat **berapa kotak** dan **kotak apa saja** di rak itu (kode K..).
  Kalau kotak dipindah dari rak 1 ke rak 2, catat juga - tabel posisi `data/rak/posisi.csv` sekarang hanya berisi rak 1
  (contoh), posisi rak 2 belum ada.
- Kalibrasi baru untuk tiap rak (latar dan cahaya berbeda).

**Fase 3 (kedua rak bersamaan):**
- `r12_s1_far`: dua rak dalam satu gambar membuat kotak jadi kecil - lebar kotak kemungkinan **< 28 px (FAIL)**. Rekam
  tetap (pilih `y` saat ditanya), karena sesi ini hanya untuk **deteksi dan hitung kotak**, bukan membaca teks.
- `r12_s2_mid`: kamera cukup dekat (lebar kotak >= 40 px), menyusuri rak 1 sampai ujung, lewat celah/tiang di antara
  rak, lalu rak 2. Tahan diam di celah antar rak 2-3 detik: ini contoh "tiang / sisi rak bukan kotak".
- Tiang dan sisi rak di tengah gambar pernah terdeteksi sebagai kotak (22 bukannya 20) - justru perlu ada di data.

## 5. Setelah merekam

```bat
venv\Scripts\python.exe scripts\ambil_frame_data.py results\data_rak\r1_*.mp4 results\data_rak\r2_*.mp4 results\data_rak\r12_s1_far.mp4 results\data_rak\r12_s2_mid.mp4
```
Jangan masukkan `r12_s7_test` (data uji). Frame masuk ke `labeling\raw_d455\`. Beri tahu Claude: frame dikirim ke
server, diberi draf label, dan disiapkan sebagai batch CVAT.

**Pembagian data:** satu sesi utuh masuk training ATAU validasi. Untuk validasi yang jujur: satu sesi rak 2 (mis.
`r2_s2_mid`) disimpan untuk validasi, supaya terlihat apakah model bisa di rak yang tidak dipakai training.

## 6. Daftar centang

- [ ] Rak 1: foto HP + daftar kotak, `r1_s1_far`, `r1_s2_mid`
- [ ] Rak 2: foto HP + daftar kotak, `r2_s1_far`, `r2_s2_mid`
- [ ] Kedua rak: `r12_s1_far`, `r12_s2_mid`
- [ ] `ambil_frame_data.py` dijalankan, jumlah frame per sesi dicatat
- [ ] Folder `results\data_rak\` (video + kartu kalibrasi) tidak dihapus

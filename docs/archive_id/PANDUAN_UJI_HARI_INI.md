# Panduan uji hari ini — 3 prioritas (laptop, di depan rak)

Semua perintah dijalankan di **cmd laptop**. Salin persis. Urutan yang disarankan: **P2 → P1 → P3** (P2 paling singkat dan
hasilnya menentukan adegan video demo).

## Persiapan (sekali, 5 menit)

1. Charger terpasang, layar laptop **terbuka**, mode daya **Best performance**. Tutup browser dan VS Code (RAM 8 GB adalah batasnya).
2. D455 di port **USB 3**. Semua 20 kotak di rak, **dalam susunan yang Anda pakai sekarang**. Susunan itu **tidak harus** sama dengan urutan
   K01…K21 di inventaris (lihat bagian "Urutan kotak" di bawah). **Ambil satu foto rak dengan ponsel sekarang**: itu patokan untuk mengembalikan kotak.
3. Kamera diam (tripod / penjepit), rak di **tengah** gambar, tiang rak tidak di sebelah kotak paling kiri.
4. Buka cmd, lalu:

```bat
cd C:\Users\sacab\Downloads\medicine_shelf_vision\medicine_shelf_vision
UJI_REALSENSE.bat
```

Menu muncul. Pilih angka lalu Enter. Di jendela kamera, **q** = berhenti. Cek posisi kamera dengan **menu 2**: HUD harus
menunjukkan `boxes 20` dan `sharpness` ≥ 300. Tekan q kalau sudah pas.

---

## P2 — Uji dekat, produk non-iVascular (±15 menit)

**Tujuan:** tahu pada jarak berapa bukti kedua (REF / barcode) terbaca, supaya adegan 6 atau 7 di video demo bisa ditentukan.

1. Menu **4**. Saat diminta `Request`, ketik: `accuforce,3.5,15` lalu Enter.
2. Tunggu sampai ada kotak **biru** (kandidat) atau status `SEARCHING` stabil, kamera diam.
3. **Dekatkan kamera pelan-pelan** ke kotak target (yang diberi nomor `#…`) sampai tulisan di labelnya **besar dan tajam di layar**.
   Kalau gambar jadi buram (angka `sharpness` di HUD turun di bawah 300), mundur sedikit: terlalu dekat juga membuat gambar buram.
4. Tahan diam **10 detik**. Lihat status:
   - **PICK hijau** = berhasil: bukti kedua terbaca dari dekat.
   - tetap biru / kuning = belum cukup.
5. Tekan **q**. Ulangi untuk dua produk lain: menu 4 → `ultimaster nagomi,2.75,33`, lalu menu 4 → `nc emerge,5.5,12`.

Hasil tersimpan di `results\bukti_laporan_20261008\`: `bacaan_uji2_<cap waktu>` dan `rs_uji2_<cap waktu>_vis.mp4`.
**Kabari saya: "menu 4 sudah"**, saya baca filenya.

*(Koreksi: saya sebelumnya menulis "kotak selebar sepertiga layar". Itu terlalu dekat untuk kamera ini. Patokannya: tulisan label besar dan tajam.)*

---

## P1 — Daftar kotak, ambil satu per satu (±30 menit)

**Tujuan:** menguji alur penuh: cari → PICK → Anda ambil → slot kosong terverifikasi (TAKEN) → kotak berikutnya, dan menguji perbaikan
"moved or taken?" secara live.

1. Menu **9**. Program membaca `rencana_20_kotak.txt` (17 permintaan) dan mulai dari `permanent sled bag`.
2. Tunggu `CHECKING SHELF` → `20 boxes`, lalu `SEARCHING`.
3. Saat muncul **PICK hijau** (`FOUND: PICK #…`), **ambil kotak yang berbingkai hijau itu**. Untuk Sled Bag yang kembar: **baca tanggal yang tercetak di kotak yang Anda ambil dan catat**
   (yang benar menurut FEFO: `2029-03-10`). Lihat "Masalah tanggal Sled Bag" di bawah.
4. Taruh kotak di **meja** (jangan dikembalikan ke rak), lalu **jauhkan tangan dari rak ±5 detik**, kamera tetap diam.
5. Status berubah `TAKEN` → permintaan berikutnya mulai otomatis. Layar terminal menulis `[stress] n/17 …`.
6. Kalau suatu kotak tetap biru/kuning **lebih dari 60 detik**, ketik `lewati` lalu Enter **di jendela cmd** (bukan di jendela kamera). Kalau dibiarkan, program menyerah sendiri setelah 180 detik (TIMEOUT).
7. Selesai atau ingin berhenti: **q** di jendela kamera.
8. **Kembalikan semua kotak ke tempat semula**, sesuai foto rak yang Anda ambil sebelum mulai.

Perkiraan: kotak iVascular (K04, K06, K10, K15) cepat; produk lain kemungkinan besar harus di-`lewati` di jarak ini. Itu normal (lihat P2).
Hasil: `hasil_ambil_<cap waktu>.csv` (waktu per kotak), `ambil_<cap waktu>_vis.mp4` (video dengan kotak dan status), `bacaan_ambil_<cap waktu>`.
**Kabari saya: "menu 9 sudah".**

Inventaris asli **tidak berubah** (menu ini memakai salinan baru setiap dijalankan).

---

## P3 — Data rak untuk training, dengan kalibrasi (±15 menit)

**Tujuan:** frame dari D455 asli untuk meningkatkan deteksi kotak, dengan catatan kondisi kamera tiap sesi.

Ulangi untuk tiga sesi (menu **D** setiap kali):

| Sesi | Posisi kamera |
| --- | --- |
| `s1_far` | seluruh rak terlihat |
| `s2_mid` | lebih dekat: lebar kotak di gambar sebesar mungkin tapi tetap tajam, bagian kiri lalu bagian kanan rak |
| `s3_close` | 3–5 kotak, digeser pelan menyusuri rak |

Untuk setiap sesi:

1. Menu **D**. `Session name`: ketik nama sesi (mis. `s1_far`). `Reference session`: tekan **Enter** (kosong).
2. **Pratinjau langsung** muncul dengan daftar PASS / WARN / FAIL. Geser kamera sampai tidak ada FAIL (WARN pada lebar kotak boleh). Tekan **q** atau Enter.
3. Tahan kamera diam **3 detik**. Hasil kalibrasi tercetak. Kalau ada FAIL, program bertanya `Record anyway? (y/n)`: ketik `n` lalu perbaiki posisi, atau `y` untuk tetap merekam.
4. Perekaman mulai. Diam beberapa detik di tiap posisi, lalu geser pelan. **60–90 detik**, lalu **q**.

Setelah ketiga sesi, ambil frame untuk training:

```bat
venv\Scripts\python.exe scripts\ambil_frame_data.py results\data_rak\s1_far.mp4 results\data_rak\s2_mid.mp4 results\data_rak\s3_close.mp4
```

Hasil: kartu kalibrasi `results\data_rak\<sesi>_kalibrasi.json/.jpg`, video, dan frame di `prelabel_size_label_full\raw_d455`.
**Kabari saya: "data rak sudah".**

---

## Urutan kotak: apa yang dimaksud

- **Bukan** foto-foto labelling (23, 28 dan 1 Oktober). Itu foto kotak di dalam kardus dengan susunan berbeda-beda, dan tidak ada hubungannya dengan susunan rak.
- Inventaris (`data\rak\inventaris.csv`) mencatat **urutan hasil pindai 30 September**: K01 Sled Bag, K03 Sled Bag, K04 Essential Pro, K05 Sled Bag, K06 Xperience Pro, K07 Accuforce …
- Rak yang tampak di rekaman D455 **sudah berbeda**: empat iVascular di kiri, lalu empat Sled Bag berdempetan, lalu Accuforce, dan seterusnya.
- Untuk uji kamera, **susunan fisik tidak mempengaruhi hasil**: kotak dikenali dari tulisannya, bukan dari urutannya. Yang penting kotak yang diambil
  **dikembalikan ke tempat semula**, supaya rak di akhir sama dengan di awal. Karena itu: foto dulu, kembalikan sesuai foto.
- Satu-satunya yang memakai urutan adalah tabel posisi demo (`data\rak\posisi.csv`: kolom = nomor urut). Tabel itu masih **CONTOH** sampai
  tim mengirim file posisi asli, jadi posisi yang keluar di demo teks bukan susunan fisik rak.

## Masalah tanggal Sled Bag

Keempat Permanent Sled Bag identik. Hanya tanggal kedaluwarsa tercetaknya yang berbeda: K01 `2029-04-26`, K03 `2029-03-10`, K05 `2029-03-24`,
K17 `2029-03-12`. Menurut FEFO, yang harus diambil duluan adalah **K03 (2029-03-10)**.

- **Di jarak rak (kotak ±33 px), tanggal itu tidak terbaca** oleh OCR. Akibatnya program tahu "ini Sled Bag" (itu mudah), tapi **tidak tahu yang mana**
  yang paling cepat kedaluwarsa. Ia memilih salah satu dari empat berdasarkan nomor pelacakan, jadi **belum tentu K03**.
- Susunan fisik sekarang juga tidak sama dengan inventaris, jadi "K03" di inventaris tidak terikat ke posisi tertentu di rak.
- Karena itu di uji menu 9 dan di video demo: **lihat sendiri tanggal di kotak yang dipilih** dan catat. Kalau bukan `2029-03-10`, itu temuan, bukan kegagalan Anda.
- Perbaikan yang sedang disiapkan: model `expiry_label` (sudah dilatih, mAP50 0,96) untuk menemukan tanggal, lalu dibaca **dari dekat**. Itu langkah setelah Senin.
- Untuk video demo: letakkan satu Sled Bag dengan tanggal `2029-03-10` di tempat yang mudah dikenali, dan sebut hasilnya apa adanya.

---

## Kalau ada masalah

| Gejala | Tindakan |
| --- | --- |
| `Cannot open the RealSense` | menu **1** (cek USB harus 3.x); tutup RealSense Viewer; cabut-colok kabel |
| Laptop lag | tutup program lain; pakai menu 2 dulu sampai lancar |
| Layar kamera tidak muncul | pastikan jendela cmd tidak sedang menunggu input; klik jendela kamera, tekan q |
| Kotak berhenti di biru terus | normal untuk non-iVascular di jarak jauh; lanjutkan dengan P2 |
| Salah ketik di menu | tekan Enter pada menu kosong, atau pilih 0 lalu mulai lagi |

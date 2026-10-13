# Uji live di laptop (lokal, RealSense D455, tanpa server)

Semua berjalan di laptop: kamera, deteksi, OCR, verifikasi, dan layar. Tidak ada tunnel, `stream_webcam.py`, atau
Remote Desktop. Versi server yang setara: `docs/UJI_LIVE_SERVER.md`. Progres dicatat di `docs/CHECKLIST_UJI_LIVE.md`
(lewati bagian A5 dan B, itu khusus server). Daftar 20 uji: `docs/CHECKLIST_DEMO.md`.

Semua perintah dijalankan di **laptop (cmd)**, folder `C:\Users\sacab\Downloads\medicine_shelf_vision\medicine_shelf_vision`.

**Aturan umum**
- Tahap tidak boleh dilompati. **Berhenti dan catat** kalau ada MATCH/AMBIL pada kotak yang salah (satu-satunya kegagalan
  yang tidak boleh terjadi). Kotak yang diragukan harus PERLU KONFIRMASI.
- Kamera hanya bisa dipakai satu program: tutup RealSense Viewer dan jendela uji sebelumnya (`q`) sebelum memulai yang baru.
- Jangan membuka aplikasi berat selama uji (laptop i3 melambat kalau panas). Semua bukti disimpan ke satu folder:
  `results\bukti_laporan_20261008\` (rekaman `.mp4`, `bacaan_*.csv`, `verify_log.csv`, `screens\`). Simpan uji yang gagal apa adanya.
- Tidak perlu `scp`: begitu selesai, beri tahu Claude. File-nya langsung terbaca dari folder itu untuk dibuat slide.

Perbedaan dengan server: model v4 (OpenVINO 960 di iGPU Intel), sekitar 6-8 FPS, OCR di CPU (sekitar 110-130 ms per kotak).
Hasil ini menguji **kebenaran** (tidak boleh salah ambil), bukan kecepatan maksimum. Tampilan lancar karena tidak lewat Remote Desktop.

---

## L0. Persiapan (sekali, tanpa kotak di depan kamera)

```bat
cd /d C:\Users\sacab\Downloads\medicine_shelf_vision\medicine_shelf_vision
venv\Scripts\python.exe scripts\cek_usb_realsense.py --detik 5
venv\Scripts\python.exe -m pytest tests -q
venv\Scripts\python.exe scripts\uji_checklist.py
venv\Scripts\python.exe scripts\rak_inventaris.py lihat
```

| Cek | Syarat lulus |
| --- | --- |
| USB | `RealSense D455: USB 3.x <-- BAGUS` (kalau `2.1`: tidak ada frame, jangan lanjut) |
| Tes unit | semua lulus (213 per 08-10) |
| Uji teks 20 permintaan | 0 gagal (19 lulus + #17 dilewati selama K10 ada di rak; sudah dicek 08-10) |
| Inventaris = isi rak | 20 kotak; hanya kotak yang benar-benar **tidak ada di rak** yang `diambil` |

**Penting, cek sekarang:** per 08-10 inventaris mencatat K10 (angiolite 4 x 19) `diambil`, tetapi kotak angiolite 4/19
terlihat di rak pada foto RealSense. Kalau benar sudah ada di rak, kembalikan statusnya (Anda yang menjalankan, bukan Claude):

```bat
venv\Scripts\python.exe scripts\rak_inventaris.py kembali K10
```
Setelah itu uji #17 (`angiolite 4 x 19`) berubah dari HABIS menjadi SIAP, jadi catat di tabel.

---

## L1. Demo teks (tanpa kamera, 5 menit)

```bat
venv\Scripts\python.exe scripts\demo_request.py --sekali "ivascular angiolite 2.5 x 29"
venv\Scripts\python.exe scripts\demo_request.py --sekali "accuforce 2.75 x 15"
venv\Scripts\python.exe scripts\demo_request.py --sekali "terumo xperience 2.5 x 15"
```
Syarat lulus: K15 SIAP; `accuforce 2.75 x 15` TIDAK ADA; merek salah ULANGI.

---

## L2. Hitung kotak (kamera agak jauh, seluruh rak terlihat, 10 menit)

```bat
venv\Scripts\python.exe scripts\test_webcam_box.py --config results\bukti_laporan_20261008\config_uji.yaml --request "angiolite,2.5,29" --expected 20 --record results\bukti_laporan_20261008\uji_live_tahap3.mp4
```
Di jendela: **`a`** = tandai area rak (menghilangkan gelas/layar yang terdeteksi jadi kotak), **`s`** = simpan tangkapan layar
ke `screens\`, **`q`** = keluar.

Syarat lulus: `tajam` > 100; jumlah kotak 20 dari 20 (19 dicatat sebagai R-21, 21 = ada benda palsu: tekan `a`).

---

## L3. Satu permintaan per kotak (inti uji, kamera DEKAT)

Dekatkan kamera sampai lebar kotak di gambar **>= 28 piksel (idealnya >= 40)**; jalankan `UJI_REALSENSE.bat` menu D untuk mengukurnya. Pada uji 08-10
dengan lebar kotak 24 piksel (19 kotak dalam satu frame) teks ukuran terbaca di 7 % bacaan dan 0 AMBIL; pada 36 piksel 34 % dan AMBIL untuk iVascular.
Setiap kamera dipindah, tekan `a` lagi (area rak berlaku untuk satu kerangka).

Satu perintah per nomor (nomor = kolom `#` di `CHECKLIST_DEMO.md`, 1-16, dan 17 selama K10 ada di rak). Contoh nomor 5 (accuforce 2.75 x 20):

```bat
UJI_LIVE_LOKAL.bat 5
```
Skrip itu menyimpan `uji_live_05.mp4` dan `bacaan_live_05.csv` ke folder bukti. Tidak ada batas waktu otomatis:
pantau sendiri, dan kalau 90 detik berlalu tanpa AMBIL tekan `q` lalu catat **WAKTU HABIS** (jangan dihapus, itu bukti gagal).

**Urutan prioritas** (yang mirip dan paling berisiko dulu): #4, #5, #6 (accuforce), lalu #1, #17 (dua angiolite, K15/K10,
tampilan hampir sama), lalu #11, #12 (conqueror), lalu #8, #9
(nagomi), lalu sisanya sesuai nomor. Hasil tiap nomor dicatat di tabel D `CHECKLIST_UJI_LIVE.md`.

**Uji negatif** (kombinasi yang tidak ada di rak, tidak boleh AMBIL):
```bat
venv\Scripts\python.exe scripts\test_webcam_box.py --config results\bukti_laporan_20261008\config_uji.yaml --request "accuforce,2.75,15" --inventaris data\rak\inventaris.csv --expected 20 --record results\bukti_laporan_20261008\uji_live_negatif.mp4 --debug-readings results\bukti_laporan_20261008\bacaan_live_negatif.csv
```

Syarat lulus: **tidak ada AMBIL pada kotak yang salah**; kotak benar boleh PERLU KONFIRMASI (aman, catat sebagai bahan
perbaikan); uji negatif tidak boleh AMBIL.

---

## L4. Pengambilan fisik (opsional, mengubah inventaris)

Cadangkan dulu, jalankan satu permintaan seperti L3, tarik kotak yang benar dengan tangan, lalu kembalikan.
```bat
copy data\rak\inventaris.csv data\rak\inventaris_sebelum_uji_live.csv
```
Syarat lulus: setelah ditarik muncul TERAMBIL dan kotak itu `diambil` di inventaris. Kembalikan dengan
`venv\Scripts\python.exe scripts\rak_inventaris.py kembali K15` (ganti K15 dengan kotak yang tadi ditarik).

---

## L5. Gladi demo (15 menit)

Alur Senin: ketik permintaan lewat `demo_request.py` (produk, spec, status, rak/level), lanjut kamera dengan `--kamera`,
tunjukkan satu kasus ditolak, hitung waktu total. Rekam layar dengan **Win + Alt + R** (Xbox Game Bar) sebagai cadangan.
Syarat lulus: dua kali berturut-turut tanpa intervensi.

## L6. Kumpulkan bukti

```bat
dir results\bukti_laporan_20261008 /s /b
```
Kalau semua nomor sudah dijalankan, kirim ringkasan: nomor yang gagal, nomor PERLU KONFIRMASI padahal kotaknya benar,
jumlah kotak terhitung, dan angka `tajam`. File di folder bukti dipakai langsung untuk slide.

## Kalau ada masalah

| Gejala | Kemungkinan penyebab | Tindakan |
| --- | --- | --- |
| `RealSense tidak bisa dibuka` | Viewer / jendela uji lain masih terbuka, atau kabel | tutup program lain; cek `cek_usb_realsense.py` |
| `USB 2.1` | kabel / port bukan USB 3 | ganti kabel; tidak ada frame di 2.1 |
| 0 AMBIL, banyak `needs_confirmation` | teks terlalu kecil di frame | dekatkan kamera (lebar kotak >= 28 piksel; kalibrasi: menu D) |
| Jumlah kotak 21 | benda palsu (gelas, layar) | tombol `a` untuk area rak |
| Laptop melambat | panas | jeda 5 menit antar-uji |

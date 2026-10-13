# Checklist progres uji live (RealSense D455; versi lokal dan remote desktop)

Panduan langkahnya: **lokal (laptop saja)** `docs/UJI_LIVE_LOKAL.md`, atau **remote desktop (server)**
`docs/UJI_LIVE_SERVER.md`. Checklist ini dipakai bersama; versi lokal melewati bagian A5 dan B (khusus server).
Daftar 20 uji: `docs/CHECKLIST_DEMO.md`.
Jadwal: laporan Jumat 09-10, demo Senin 12-10. Terakhir diperbarui: 08-10.

Cara memakai: centang `[x]` begitu syarat lulus tahap terpenuhi, isi **Tanggal** dan **Bukti** (nama file di `results/`).
Uji yang gagal tidak dihapus: catat di kolom Catatan, simpan filenya. Itu bahan slide "kesulitan nyata".

## A. Kamera dan jaringan

| Tahap | Syarat lulus | Status | Tanggal | Bukti / catatan |
| --- | --- | --- | --- | --- |
| A1. RealSense di USB 3 | `cek_usb_realsense.py` menampilkan `USB 3.x` | [x] | 08-10 | D455 USB 3.2 (sebelumnya 2.1, tanpa frame; kabel/port diganti) |
| A2. Stream lokal | frame 1280x800 @ 30 fps, `tajam` > 100 | [x] | 08-10 | `tajam` 778-790, frame 30 ms |
| A3. Pipeline jalan di laptop (tanpa server) | JUMLAH KOTAK terhitung, tidak error | [x] | 08-10 | 20 kotak (putaran 1), 21 (putaran 2); 0 AMBIL / 145 dtk: teks terlalu kecil |
| A4. Kerangka kamera untuk baca teks | lebar kotak >= 28 px (idealnya >= 40 px), ketajaman >= 300; jalankan UJI_REALSENSE.bat menu D (kalibrasi) | [ ] |  |  |
| A5. Tunnel + stream ke server | server membuka `http://127.0.0.1:18090/video.mjpg`, `masuk` 10-15 fps | [ ] |  |  |

## B. Server siap (Tahap 0-2)

| Tahap | Syarat lulus | Status | Tanggal | Bukti / catatan |
| --- | --- | --- | --- | --- |
| B1. Kode + data + model di server | `git pull`, `data/rak/inventaris.csv`, `models/box_v4s.pt` ada | [ ] |  |  |
| B2. Tes unit di server | `pytest` lulus semua | [ ] |  |  |
| B3. Uji teks 20 permintaan | `uji_checklist.py`: 0 gagal (19 lulus + #17 dilewati selama K10 ada di rak) | [ ] |  |  |
| B4. Demo teks (Tahap 1) | 3 permintaan sesuai tabel | [ ] |  |  |
| B5. Jalur video / HUD (Tahap 2) | HUD menampilkan `tampil`, `masuk`, `proses maks` | [ ] |  |  |

## C. Uji kamera (Tahap 3-4)

| Tahap | Syarat lulus | Status | Tanggal | Bukti / catatan |
| --- | --- | --- | --- | --- |
| C1. Hitung kotak (Tahap 3) | 20 dari 20 (19 dicatat sebagai R-21), tanpa `KAMERA BERGERAK` | [ ] |  | `uji_live_tahap3.mp4` |
| C2. Satu permintaan per kotak, urutan prioritas | tidak ada MATCH/AMBIL pada kotak yang salah | [ ] |  |  |
| C3. Uji negatif | `accuforce 2.75 x 15`: tidak ada AMBIL | [ ] |  | `uji_live_negatif.mp4` |
| C4. Pengambilan fisik (opsional, Tahap 5) | TERAMBIL muncul, inventaris `diambil`, lalu `kembali` | [ ] |  | cadangkan `inventaris.csv` dulu |
| C5. Gladi demo lengkap (Tahap 6) | alur dua kali berturut-turut tanpa intervensi, layar direkam | [ ] |  |  |

## D. Hasil per kotak (isi saat Tahap 4)

Status: `AMBIL` (sukses), `CEK ULANG` lalu sukses, `PERLU KONFIRMASI` (aman, bahan perbaikan), `WAKTU HABIS` (gagal),
`SALAH` (MATCH pada kotak yang salah: hentikan uji, kirim rekaman).

| # | Permintaan | Hasil | Detik sampai AMBIL | Teks terbaca benar? | Bukti (rekaman / bacaan) |
| --- | --- | --- | --- | --- | --- |
| 4 | accuforce 2.5 x 12 |  |  |  | `uji_live_04.mp4` / `bacaan_live_04.csv` |
| 5 | accuforce 2.75 x 20 |  |  |  | `uji_live_05.mp4` / `bacaan_live_05.csv` |
| 6 | accuforce 3.5 x 15 |  |  |  | `uji_live_06.mp4` / `bacaan_live_06.csv` |
| 11 | conqueror 3.5 x 15 |  |  |  | `uji_live_11.mp4` / `bacaan_live_11.csv` |
| 12 | conqueror 4 x 8 |  |  |  | `uji_live_12.mp4` / `bacaan_live_12.csv` |
| 8 | nagomi 2.75 x 33 |  |  |  | `uji_live_08.mp4` / `bacaan_live_08.csv` |
| 9 | nagomi 3 x 24 |  |  |  | `uji_live_09.mp4` / `bacaan_live_09.csv` |
| 1 | angiolite 2.5 x 29 (K15) |  |  |  | `uji_live_01.mp4` / `bacaan_live_01.csv` |
| 17 | angiolite 4 x 19 (K10; mirip #1) |  |  |  | `uji_live_17.mp4` / `bacaan_live_17.csv` |
| 2 | xperience pro 2.5 x 15 |  |  |  | `uji_live_02.mp4` / `bacaan_live_02.csv` |
| 3 | essential pro 3 x 40 |  |  |  | `uji_live_03.mp4` / `bacaan_live_03.csv` |
| 7 | ryurei 2 x 10 |  |  |  | `uji_live_07.mp4` / `bacaan_live_07.csv` |
| 10 | nc emerge 5.5 x 12 |  |  |  | `uji_live_10.mp4` / `bacaan_live_10.csv` |
| 13 | sapphire 3 x 15 |  |  |  | `uji_live_13.mp4` / `bacaan_live_13.csv` |
| 14 | scoreflex 3 x 15 |  |  |  | `uji_live_14.mp4` / `bacaan_live_14.csv` |
| 15 | agent 2.75 x 30 |  |  |  | `uji_live_15.mp4` / `bacaan_live_15.csv` |
| 16 | permanent sled bag |  |  |  | `uji_live_16.mp4` / `bacaan_live_16.csv` |

Ringkasan (isi setelah tabel D penuh): AMBIL ___ / 17, PERLU KONFIRMASI ___, WAKTU HABIS ___, SALAH ___ (harus 0).

## E. Dokumentasi untuk slide laporan

| Bukti yang diminta | Sumber | Status |
| --- | --- | --- |
| Foto / rekaman pengambilan (AMBIL) | `screens/peristiwa/*.jpg`, `uji_live_*.mp4` | [ ] |
| Contoh uji **berhasil** (log + foto) | baris `AMBIL` di `verify_log.csv` | [ ] |
| Contoh uji **gagal** (log + foto) | baris `WAKTU HABIS` / `BATAL` / `PERLU KONFIRMASI` | [ ] |
| Detail kotak terbaca di rak | `bacaan_live_*.csv` + layar HUD yang diperbesar | [ ] |
| Kesulitan nyata (RealSense USB 2.1, teks terlalu kecil, deteksi palsu) | log 08-10 di bagian A, tangkapan layar RealSense Viewer | [x] |
| Hasil dikirim ke laptop | `hasil_uji_live.tgz` (Tahap 7) lalu `scp` | [ ] |

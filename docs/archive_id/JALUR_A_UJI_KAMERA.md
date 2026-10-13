# Jalur A: uji kamera 16 kotak lewat server (model v4, GPU server)

Kamera ada di laptop, model ada di server. Gambar dikirim lewat tunnel SSH. Semua perintah di bawah bisa
disalin langsung; yang di laptop tidak membutuhkan file baru apa pun (hanya `stream_webcam.py` yang sudah ada).

## Urutan (±30 menit)

### 0. Fisik (sekali)
- 20 kotak di rak, webcam eksternal terpasang DIAM (tripod/dijepit) menghadap rak.
- Tutup program lain yang memakai webcam (aplikasi Kamera, Zoom, Teams, browser dengan kamera).
- Buka remote desktop (Remmina) ke server, biarkan terbuka.

### 1. Laptop, cmd jendela 1: kirim kamera
```bat
cd /d C:\path\ke\medicine_shelf_vision
venv\Scripts\python.exe scripts\stream_webcam.py
```
Tunggu baris `[stream] http://127.0.0.1:8090/video.mjpg (1920x1080 ...)`. Kalau yang terpilih bukan webcam
eksternal, hentikan (Ctrl+C) dan jalankan ulang dengan `--camera 0` atau `--camera 1`.
Jaringan lambat: tambah `--width 1280 --height 720 --quality 60`.

### 2. Laptop, cmd jendela 2: tunnel ke server
```bat
ssh -N -R 18090:127.0.0.1:8090 Zen@140.113.149.94
```
Isi password server, lalu biarkan jendela terbuka dan kosong (itu normal).

### 3. Server, terminal di desktop Remmina
```bash
cd ~/Downloads/mackay-med-robot-vision
bash scripts/jalankan_uji_kamera_server.sh
```
Skrip ini otomatis:
1. memeriksa model dan data, lalu uji teks (harus 20 lulus);
2. menunggu tunnel sampai 3 menit (bisa dinyalakan setelah skrip dimulai);
3. mengambil beberapa frame dan melaporkan resolusi, fps, ketajaman, kecerahan; frame contoh disimpan di
   `results/frame_cek_<waktu>.jpg`. Ada peringatan (buram/gelap/resolusi rendah): skrip bertanya lanjut atau tidak;
4. menjalankan 16 kotak bergiliran (90 detik per kotak), merekam video, lalu mencetak ringkasan.

Opsi: `DEVICE=cpu` bila muncul CUDA out of memory (GPU dipakai bersama; lebih lambat, ±2 FPS),
`WAKTU=60` untuk waktu per kotak, `TUNGGU=300` untuk menunggu tunnel lebih lama.

### 4. Saat jendela video berjalan
- Judul di atas jendela: `STRESS n/16 ... detik`. Sistem berpindah otomatis ke kotak berikutnya.
- Tugas Anda: lihat kotak **hijau tebal + AMBIL**; cocokkan dengan kolom Kotak di
  `docs/DAFTAR_PRODUK_UJI_KAMERA.md` (mis. #1 angiolite 2.5 x 29 harus K15). Catat: benar / salah kotak / tidak memilih.
- "KAMERA BERGERAK - tahan diam" = jangan sentuh kamera. Tombol: `q` selesai, `s` simpan frame, `r` ulang permintaan.
- Barang tidak benar-benar diambil; cukup sistem menunjuk kotak.

### 5. Hasil
Ringkasan tercetak di terminal. File di `results/`: `uji_kamera_<waktu>.csv` (hasil per kotak),
`uji_kamera_<waktu>.mp4` (rekaman, bisa jadi cadangan demo), `bacaan_uji_kamera_<waktu>.csv` (bacaan OCR).
Kirim ke saya: ringkasan di terminal + catatan benar/salah per nomor.

## Aturan lulus
- **Wajib:** tidak ada AMBIL pada kotak yang salah. Satu kali salah kotak = gagal dan harus ditelusuri sebelum demo.
- **Boleh:** WAKTU HABIS / PERLU KONFIRMASI (sistem ragu, tidak menebak).
- Target awal (usulan, bisa diubah tim): minimal 12 dari 16 berhasil AMBIL.

## Bila ada masalah

| Gejala | Penyebab / tindakan |
| --- | --- |
| `menunggu tunnel/stream dari laptop...` lama lalu GAGAL | Jendela 1 atau 2 belum jalan/tertutup, atau password SSH salah. Periksa kedua jendela, skrip menunggu sampai 3 menit. |
| `Gambar masuk ... PERINGATAN gambar buram` | Dekatkan kamera / perbaiki fokus & cahaya. Buka `results/frame_cek_*.jpg` untuk melihat apa yang dilihat server. |
| Gambar yang masuk bukan rak/kamera salah | Ulang jendela 1 dengan `--camera 0` atau `--camera 1`. |
| `CUDA out of memory` | Jalankan dengan `DEVICE=cpu`. |
| Jendela video tidak muncul | Jalankan skrip dari terminal di dalam desktop Remmina, bukan dari SSH biasa. |
| Semua WAKTU HABIS | Teks ukuran tidak terbaca (jarak/fokus/cahaya). Perbaiki lalu ulangi. |

Catatan: posisi rak/level pada pesan TM masih data CONTOH (`data/rak/posisi.csv`); uji ini hanya memeriksa
pemilihan kotak, bukan koordinat fisik.

# Panduan uji kamera 16 kotak (model v4)

Tujuan: membuktikan sistem memilih kotak yang benar dari kamera sungguhan, dan tidak pernah memilih kotak yang
salah. Model yang dipakai: **v4** (`models/box_v4s.pt`). Bila v6 nanti lebih baik, uji teks otomatis diulang
saja; uji kamera ini tidak perlu diulang penuh.

Waktu: sekitar 30 menit (16 kotak x 90 detik + persiapan). Perlu: laptop dengan webcam, 16 kotak fisik, tripod
atau penjepit, akses Remmina ke server.

## Bagian A. Persiapan (sekali)

1. **Rak dan kamera.** Susun 20 kotak di rak (urutan bebas, tapi catat). Pasang webcam DIAM menghadap rak.
   Teks ukuran pada punggung kotak harus terlihat jelas di jendela video (kalau buram, dekatkan kamera atau
   perbaiki fokus; di bawah kira-kira 100 pada ukuran ketajaman = teks tidak terbaca).
2. **Kode di laptop terbaru.** Di folder `medicine_shelf_vision`: `git pull`. Pastikan ada
   `UJI_KAMERA_KIRIM_KE_SERVER.bat` dan `results\rencana_uji_kamera_16.txt`.
3. **Server terbaru** (Remmina, terminal): `cd ~/Downloads/mackay-med-robot-vision && git pull`
   (atau salin 4 file baru kalau belum di-push: `docs/`, `results/rencana_uji_kamera_16.txt`,
   `scripts/jalankan_uji_kamera_server.sh`, `scripts/ringkas_uji_kamera.py`).

## Bagian B. Hubungkan kamera laptop ke server

4. **Laptop:** klik dua kali `UJI_KAMERA_KIRIM_KE_SERVER.bat`. Ikuti layarnya:
   - Jendela 1 menjalankan stream webcam. Tunggu sampai muncul tanda stream aktif.
   - Jendela 2 menjalankan tunnel SSH. Ketik password server. Setelah itu layar kosong = normal.
   - **Jangan tutup kedua jendela** sampai uji selesai.
5. **Server (Remmina), terminal:**
   ```bash
   cd ~/Downloads/mackay-med-robot-vision
   bash scripts/jalankan_uji_kamera_server.sh
   ```
   Skrip memeriksa sendiri: model ada, inventaris ada, uji teks 20/20 lulus, dan stream dari laptop terjangkau.
   Kalau salah satu gagal, pesan di layar menyebut apa yang harus diperbaiki.

## Bagian C. Jalankan uji

6. Jendela video terbuka dalam 10-60 detik. Sistem akan meminta kotak **1 dari 16** secara bergiliran
   (judul permintaan terlihat di bagian atas jendela, contoh `STRESS 1/16 ... 12/90 s`).
7. Untuk tiap permintaan, **tidak perlu melakukan apa pun pada kotak** selain membiarkan kamera melihat rak.
   Yang dicek: apakah kotak bertanda **hijau tebal + AMBIL** adalah kotak yang benar (lihat kolom Kotak di
   `docs/DAFTAR_PRODUK_UJI_KAMERA.md`: mis. permintaan #1 angiolite 2.5 x 29 harus menyorot kotak K15).
   - Barang tidak benar-benar diambil; cukup sistem menunjuk kotak (AMBIL).
   - Berpindah ke permintaan berikutnya terjadi otomatis setelah AMBIL stabil atau 90 detik habis.
8. **Catat di kertas atau di checklist** untuk tiap nomor: hijau tebal menunjuk kotak yang benar (OK),
   salah kotak (GAGAL KERAS), atau tidak memilih apa pun (PERLU KONFIRMASI / WAKTU HABIS = aman, bukan gagal).
9. Tombol di jendela video: `q` selesai lebih awal, `s` simpan frame, `r` ulang permintaan yang sama.
   Kalau tertulis "KAMERA BERGERAK - tahan diam", jangan sentuh kamera.

## Bagian D. Hasil dan penutup

10. Setelah 16 kotak selesai, skrip mencetak ringkasan otomatis. Berkas hasil:
    - `results/uji_kamera_<waktu>.csv` (hasil per kotak), `results/uji_kamera_<waktu>.mp4` (rekaman, cadangan
      untuk demo), `results/bacaan_uji_kamera_<waktu>.csv` (bacaan OCR, untuk analisis bila ada yang gagal).
11. Isi kolom **Hasil kamera** dan **OK** di tabel A `docs/CHECKLIST_DEMO.md`.
12. Kirim ke saya: isi ringkasan di terminal dan file `.csv` di atas. Salin dengan
    `scp Zen@140.113.149.94:~/Downloads/mackay-med-robot-vision/results/uji_kamera_*.csv .` bila perlu.

## Aturan lulus

- **Wajib:** tidak ada satu pun AMBIL pada kotak yang salah. Satu kali salah kotak = uji gagal dan
  kami harus menelusuri penyebabnya sebelum demo.
- **Boleh:** WAKTU HABIS atau PERLU KONFIRMASI. Itu artinya sistem ragu dan meminta dokter, sesuai aturan
  (konfirmasi akhir tetap di dokter). Namun terlalu banyak WAKTU HABIS berarti kamera/pencahayaan perlu
  diperbaiki; target awal: minimal 12 dari 16 berhasil AMBIL.

## Bila ada masalah

| Gejala | Tindakan |
| --- | --- |
| `GAGAL: stream tidak terjangkau` | Jendela 1 atau 2 di laptop tertutup/belum aktif, atau password SSH salah. Ulangi langkah 4. |
| Jendela video tidak muncul | Pastikan Remmina menampilkan desktop server, bukan hanya terminal SSH. |
| Kamera "sedang dipakai" | Ada program lain memakai webcam. Tutup, jalankan ulang `.bat`. |
| Semua WAKTU HABIS | Teks terlalu kecil/buram. Dekatkan kamera atau perbaiki fokus/cahaya, lalu ulangi. |
| `git pull` bentrok | Hentikan, kirim pesan galat ke saya. Jangan hapus file. |

Posisi rak/level dalam pesan ke TM masih data CONTOH (`data/rak/posisi.csv`). Uji ini hanya memeriksa
**pemilihan kotak**, bukan koordinat fisik.

# Tahapan uji live di remote desktop (server)

Webcam tetap di laptop. Laptop mengirim gambar lewat tunnel SSH. Server (RTX A5000) yang memproses, dan hasilnya
dilihat di desktop Remmina. Dasar: `docs/BOX_PIPELINE.md` bagian 3b dan checklist `docs/CHECKLIST_DEMO.md`.

**Aturan umum**
- Urutan tahap tidak boleh dilompati. Tahap berikutnya hanya dimulai kalau syarat lulus tahap sebelumnya terpenuhi.
- **Berhenti dan catat** kalau ada MATCH pada kotak yang salah. Itu satu-satunya kegagalan yang tidak boleh terjadi.
  Kotak yang diragukan harus berstatus PERLU KONFIRMASI, tidak pernah ditebak.
- Setiap blok perintah diberi label tempat menjalankannya: **laptop cmd** atau **server (terminal Remmina)**.
- Kamera hanya bisa dipakai satu program. Tutup `test_webcam_box.py` di laptop sebelum `stream_webcam.py`.
- Jangan menjalankan tes berat di laptop selama uji berlangsung (laptop pernah melambat karena panas).

---

## Tahap 0. Persiapan (sehari sebelumnya, tanpa kamera)

| Cek | Cara | Syarat lulus |
| --- | --- | --- |
| Kode terbaru di server | server: `cd ~/Downloads/mackay-med-robot-vision && git pull` | `Already up to date` atau berhasil |
| Data demo ada | server: `ls data/rak/inventaris.csv data/gtin_catalog.csv` | kedua file ada |
| Model v4 ada | server: `ls -la models/box_v4s.pt` | file ada, sekitar 20 MB |
| Tes unit | server: `python -m pytest tests -q` | **205 lulus** |
| Uji teks 20 permintaan | server: `python scripts/uji_checklist.py` | **20 lulus, 0 gagal** |
| Paket OCR terpasang | server: `pip install rapidocr onnxruntime zxing-cpp` | tanpa error |
| Isi rak = inventaris | laptop: `python scripts\rak_inventaris.py lihat` | 20 kotak, K10 `diambil` kalau memang diambil |

Semua perintah server dijalankan setelah `cd ~/Downloads/mackay-med-robot-vision` dan `source ~/trse_venv/bin/activate`.

---

## Tahap 1. Demo teks di server (tanpa kamera, sekitar 5 menit)

**Server (terminal Remmina):**
```bash
cd ~/Downloads/mackay-med-robot-vision && source ~/trse_venv/bin/activate
python scripts/demo_request.py --sekali "ivascular angiolite 2.5 x 29"
python scripts/demo_request.py --sekali "accuforce 2.75 x 15"
python scripts/demo_request.py --sekali "terumo xperience 2.5 x 15"
```

Syarat lulus: hasil sama dengan tabel 20 uji. Contoh: K15 SIAP; `accuforce 2.75 x 15` TIDAK ADA; merek salah ULANGI.
Layar Remmina harus terbaca jelas (huruf tidak terpotong).

---

## Tahap 2. Jalur video tanpa kamera (replay rekaman, sekitar 10 menit)

Menguji tunnel, kecepatan, dan HUD **tanpa** kotak fisik. Pakai rekaman lama yang ada di laptop.

**Laptop cmd 1** (folder `medicine_shelf_vision`):
```bat
venv\Scripts\python.exe scripts\stream_webcam.py --video results\uji_tahap2.mp4 --video-fps 10.7 --start 220
```
Kalau file rekaman itu sudah dihapus, lewati tahap ini dan langsung ke Tahap 3.

**Laptop cmd 2** (isi password server):
```bat
ssh -N -R 18090:127.0.0.1:8090 Zen@140.113.149.94
```

**Server (terminal Remmina):**
```bash
python scripts/test_webcam_box.py --config config/box_pipeline_config_server.yaml --source http://127.0.0.1:18090/video.mjpg --request "angiolite,2.5,29" --expected 20 --max-frames 300
```

Syarat lulus:
- Jendela terbuka dan HUD menampilkan `tampil`, `masuk`, `proses maks`.
- `masuk` mendekati 10-15 fps. Kalau jauh di bawah itu, jaringan penuh: turunkan dengan `--quality 70 --fps 10` di cmd 1.
- Tidak ada pesan error model atau OCR.

---

## Tahap 3. Kamera live, rak diam, hitung kotak (sekitar 10 menit)

Menyiapkan tripod dan jarak sebelum menguji identitas. Tutup cmd 1 yang lama dulu.

**Laptop cmd 1:** `venv\Scripts\python.exe scripts\stream_webcam.py`
**Laptop cmd 2:** `ssh -N -R 18090:127.0.0.1:8090 Zen@140.113.149.94`

**Server (terminal Remmina):**
```bash
python scripts/test_webcam_box.py --config config/box_pipeline_config_server.yaml --source http://127.0.0.1:18090/video.mjpg --request "angiolite,2.5,29" --expected 20 --record results/uji_live_tahap3.mp4
```

Syarat lulus:
- HUD `tajam` di bagian tengah frame di atas sekitar 100. Geser jarak kamera sampai angkanya tinggi.
- Jumlah kotak terhitung **20 dari 20**. Catatan: 19 dari 20 pernah terjadi (dua punggung Boston Scientific yang sempit
  tergabung, R-21). Kalau 19, catat sebagai R-21 dan lanjut. Sistem tetap aman karena `--expected 20` mencegah TIDAK ADA palsu.
- Tripod stabil: tidak ada pesan `KAMERA BERGERAK`.

---

## Tahap 4. Satu permintaan per kotak (inti uji, sekitar 40 menit)

Untuk tiap nomor di tabel 20 uji (nomor 1-16), jalankan satu permintaan. Pakai wrapper supaya sama dengan demo:

**Server (terminal Remmina):**
```bash
python scripts/uji_checklist.py --kamera 5 --kamera-arg --config config/box_pipeline_config_server.yaml --source http://127.0.0.1:18090/video.mjpg --expected 20 --record results/uji_live_05.mp4
```
Ganti angka `5` dan nama rekaman untuk nomor lain. Tekan `q` di jendela untuk berhenti.

**Urutan prioritas** (kotak yang mirip dan paling berisiko dulu):
1. #4, #5, #6: tiga accuforce.
2. #11, #12: dua conqueror.
3. #8, #9: dua nagomi.
4. Sisanya sesuai nomor.
5. **Uji negatif:** minta `accuforce 2.75 x 15` (kombinasi yang tidak ada di rak) tanpa menyingkirkan K07, K16, K21.

Catat per nomor di tabel pada `CHECKLIST_DEMO.md`: hasil akhir (MATCH / CONFIRM / TIDAK ADA), kotak yang dipilih, dan waktu
sampai AMBIL.

Syarat lulus:
- **Tidak ada MATCH atau AMBIL pada kotak yang salah.** Satu kali saja = gagal, hentikan dan kirim rekamannya.
- Uji negatif tidak boleh AMBIL. Hasil yang benar: tidak ada MATCH, status TIDAK ADA atau PERLU KONFIRMASI.
- Kotak yang benar boleh berstatus PERLU KONFIRMASI. Itu aman, tapi catat: itu bahan untuk perbaikan model.
- Target kasar: sebagian besar kotak benar menjadi AMBIL dalam sekitar 60 detik. Catat angkanya, jangan dipaksakan.

---

## Tahap 5. Pengambilan fisik (opsional, sekitar 15 menit)

Menguji TERAMBIL dan pembaruan inventaris. Ini **mengubah** `inventaris.csv`, jadi cadangkan dulu.

**Server (terminal Remmina):**
```bash
cp data/rak/inventaris.csv data/rak/inventaris_sebelum_uji_live.csv
```
Jalankan satu permintaan seperti Tahap 4, lalu tarik kotak yang benar dari rak dengan tangan.

Syarat lulus: setelah kotak ditarik muncul TERAMBIL, dan kotak itu berstatus `diambil` di inventaris.
Kembalikan kotaknya ke rak, lalu kembalikan statusnya (ganti K15 dengan kotak yang tadi diambil):

**Server (terminal Remmina):**
```bash
python scripts/rak_inventaris.py kembali K15
```

---

## Tahap 6. Gladi demo lengkap (sekitar 15 menit)

Persis seperti yang akan ditunjukkan Senin. Rekam layar Remmina sebagai cadangan.

1. Ketik permintaan lewat `demo_request.py` (teks: produk, spec, status, rak/level).
2. Lanjut ke kamera dengan `--kamera`. Tunjukkan HUD dan hasil verifikasi.
3. Tunjukkan satu kasus ditolak (merek salah atau ukuran tidak ada).
4. Hitung waktu total satu permintaan dari ketik sampai hasil.

Syarat lulus: alur berjalan dua kali berturut-turut tanpa intervensi, dan rekaman layar tersimpan.

---

## Tahap 7. Kumpulkan hasil

**Server (terminal Remmina):**
```bash
ls -la results/ | tail -20
tar czf hasil_uji_live.tgz results/uji_live_*.mp4 results/demo_log.jsonl results/uji_checklist_*.csv
```
Kirim `hasil_uji_live.tgz` ke laptop dengan `scp`, dan kirim ringkasan ini ke Claude: nomor yang gagal, nomor yang
PERLU KONFIRMASI padahal kotaknya benar, jumlah kotak terhitung, dan angka `masuk` / `tampil` dari HUD.

## Kalau ada masalah

| Gejala | Kemungkinan penyebab | Tindakan |
| --- | --- | --- |
| `Video tidak bisa dibuka` | tunnel belum jalan, atau cmd 1 belum aktif | cek cmd 1 dan cmd 2 masih terbuka; pastikan port **18090**, bukan 8090 |
| Model tidak ditemukan | `models/box_v4s.pt` tidak ada di folder proyek | kirim lagi dengan `scp` |
| Gambar di server tertinggal | jaringan penuh | `--quality 70 --fps 10` di cmd 1 |
| Kamera tidak terbuka di laptop | program lain masih memakai kamera | tutup `test_webcam_box.py` di laptop |
| Core dumped saat OCR | OCR di GPU tidak stabil | pastikan `ocr.device: cpu` di config server |
| `Inventaris tidak ada` | folder hasil `git clone` tanpa data | unzip `data_demo_20261007.zip` di folder proyek |

# Checklist demo permintaan teks

Jadwal: laporan progres **Jumat 09-10 sore**, meeting lanjutan + demo **Senin 12-10**.
Dasar: meeting 05-10. Yang dinilai adalah program yang berjalan. Input brand + produk + spec, output produk, spec,
dan posisi (rak/level). Konfirmasi akhir tetap di dokter.

Terakhir diperbarui: 07-10.

## Selesai

- [x] Ringkasan meeting dicatat. Klaim yang hanya ada di rangkuman AI ditandai bukan keputusan meeting:
      lengan ganda, RealSense, OCR tanggal kedaluwarsa, barcode sebagai cadangan.
- [x] `src/box_demo.py`: tabel posisi, `lookup()`, lima status, pesan TM (`to_tm_message`), teks layar, dan log JSONL.
- [x] `data/rak/posisi.csv`: **CONTOH**. Rak R1, level 1, kolom = urutan di rak. Belum dari tim.
- [x] `scripts/demo_request.py`: mode interaktif, `--sekali`, dan `--json`. Inventaris hanya dibaca, tidak diubah.
- [x] `scripts/buat_paket_server.py`: daftar file tetap, sha256 per file, dan menolak membuat paket kalau ada modul `src` yang tertinggal.
- [x] `pick_box()` memakai **FEFO ketat** (keputusan Wardana, Opsi A): selalu kotak yang kedaluwarsanya paling awal.
      Kalau posisinya tidak ada di tabel, status menjadi PERLU KONFIRMASI. Sistem tidak pindah ke kotak lain.
- [x] `tests/test_box_demo.py`: 11 tes. Seluruh tes di laptop: **205 lulus**.
- [x] Uji skenario demo di laptop: keenam skenario sesuai harapan. Satu permintaan per status:

  | Permintaan | Hasil yang diharapkan |
  | --- | --- |
  | `ivascular angiolite 2.5 x 29` | SIAP, K15 |
  | `angiolite 4x19` | HABIS (K10 sudah diambil) |
  | `angiolite 3 x 20` | HABIS (tidak ada di inventaris) |
  | `angiolite 2.5` | ULANGI (panjang tidak ada) |
  | `sled bag` | SIAP, kotak dengan kedaluwarsa paling awal (barang tanpa ukuran) |
  | `terumo xperience 2.5 x 15` | ULANGI (merek salah) |

- [x] Paket server dibuat: `box_pipeline_demo_20261007.zip`.

## Belum

- [ ] Jalankan `python scripts/uji_checklist.py` di server: harus 20 lulus.
- [ ] Kirim paket ke server, cek sha256, lalu jalankan `pytest` di server. Harus 205 lulus.
- [ ] Gladi demo lewat Remmina, lalu rekam layarnya sebagai cadangan.
- [x] Mode kamera di program demo: `--kamera` meneruskan status SIAP ke `test_webcam_box.py` (diuji sampai skrip kamera terbuka; belum dengan kamera/stream asli).
- [ ] Uji mode kamera sungguhan: laptop `stream_webcam.py` + tunnel, server `demo_request.py --sekali "angiolite 2.5 x 29" --kamera --kamera-arg --config config/box_pipeline_config_server.yaml --source http://127.0.0.1:18090/video.mjpg`. Langkah lengkapnya ada di `docs/UJI_LIVE_SERVER.md` (Tahap 0-7).
- [ ] Laporan Jumat: tangkapan layar demo, tabel status, dan pertanyaan untuk tim.
- [ ] Ganti `posisi.csv` contoh dengan data rak/level dari tim.

## Daftar 20 uji (isi kolom Hasil dan OK saat menguji)

Otomatis (uji teks 1-20, laporan ke `results/`): `python scripts/uji_checklist.py`. Hasil di laptop: **20 lulus, 0 gagal**.
Kamera per nomor: `python scripts/uji_checklist.py --kamera N` (N = 1-16, kotak fisik di depan kamera).

A. Barang ada di rak: harus **SIAP** (teks), lalu uji kamera untuk nomor 1-16.

| # | Permintaan | Kotak | Catatan | Hasil teks | Hasil kamera | OK |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `angiolite 2.5 x 29` | K15 | satu-satunya angiolite di rak |  |  | [ ] |
| 2 | `xperience pro 2.5 x 15` | K06 |  |  |  | [ ] |
| 3 | `essential pro 3 x 40` | K04 | ukuran panjang |  |  | [ ] |
| 4 | `accuforce 2.5 x 12` | K07 | mirip #5/#6 |  |  | [ ] |
| 5 | `accuforce 2.75 x 20` | K16 | mirip #4/#6 |  |  | [ ] |
| 6 | `accuforce 3.5 x 15` | K21 | mirip #4/#5 |  |  | [ ] |
| 7 | `ryurei 2 x 10` | K09 | ukuran terkecil |  |  | [ ] |
| 8 | `nagomi 2.75 x 33` | K13 | mirip #9 |  |  | [ ] |
| 9 | `nagomi 3 x 24` | K18 | kedaluwarsa 2028-01-31 |  |  | [ ] |
| 10 | `nc emerge 5.5 x 12` | K08 | kedaluwarsa 2028-01-03 |  |  | [ ] |
| 11 | `conqueror 3.5 x 15` | K11 | mirip #12 |  |  | [ ] |
| 12 | `conqueror 4 x 8` | K12 | mirip #11 |  |  | [ ] |
| 13 | `sapphire 3 x 15` | K14 |  |  |  | [ ] |
| 14 | `scoreflex 3 x 15` | K19 |  |  |  | [ ] |
| 15 | `agent 2.75 x 30` | K20 |  |  |  | [ ] |
| 16 | `permanent sled bag` | K03 | 4 kotak (K01,K03,K05,K17); uji FEFO: kedaluwarsa 2029-03-10 paling awal |  |  | [ ] |

B. Harus ditolak atau tidak ditemukan (uji teks saja, tanpa kamera).

| # | Permintaan | Hasil yang diharapkan | Catatan | Hasil teks | OK |
| --- | --- | --- | --- | --- | --- |
| 17 | `angiolite 4 x 19` | TIDAK ADA, sudah diambil (K10) | hanya jika K10 masih `diambil` |  | [ ] |
| 18 | `accuforce 2.75 x 15` | TIDAK ADA, kombinasi tidak ada | mirip #5 dan #6 |  | [ ] |
| 19 | `terumo xperience 2.5 x 15` | ULANGI, merek tidak cocok dengan produk |  |  | [ ] |
| 20 | `essential pro 3` | ULANGI, panjang belum ditulis |  |  | [ ] |

Aturan lulus uji kamera: tidak boleh ada MATCH pada kotak yang salah. Kalau ragu, hasilnya harus PERLU KONFIRMASI.

## Pertanyaan untuk tim

1. Format file posisi rak/level: kolom apa saja, dan per kotak atau per jenis barang?
2. Bentuk data yang diterima software TM: JSON, teks, atau socket/ROS? Sementara ini kita pakai `to_tm_message()`.
3. Siapa yang memegang antarmuka konfirmasi suara (warna + "execute")?

## Perintah

Laptop, cmd, di folder `medicine_shelf_vision`:

```bat
venv\Scripts\python.exe -m pytest tests/test_box_demo.py -q
venv\Scripts\python.exe scripts\demo_request.py
venv\Scripts\python.exe scripts\buat_paket_server.py
scp C:\Users\sacab\Downloads\box_pipeline_demo_20261007.zip Zen@140.113.149.94:~/Downloads/
```

Server, terminal di desktop Remmina:

```bash
cd ~/Downloads/box_pipeline && unzip -o ~/Downloads/box_pipeline_demo_20261007.zip
sha256sum -c --quiet paket_sha256.txt
source ~/trse_venv/bin/activate
python -m pytest tests -q
python scripts/demo_request.py
```

Tanggal pada nama zip mengikuti hari paket dibuat. Pakai nama yang dicetak oleh `buat_paket_server.py`.

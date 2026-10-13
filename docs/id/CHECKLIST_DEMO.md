# Checklist demo permintaan teks

Jadwal: laporan progres **Jumat 09-10 sore**, meeting lanjutan + demo **Senin 12-10**.
Dasar: meeting 05-10. Yang dinilai adalah program yang berjalan. Input brand + produk + spec, output produk, spec,
dan posisi (rak/level). Konfirmasi akhir tetap di dokter.

Terakhir diperbarui: 08-10 (hasil server hucenrotia-ai + audit Codex + K10 kembali ke rak).

## Selesai

- [x] Ringkasan meeting dicatat. Klaim yang hanya ada di rangkuman AI ditandai bukan keputusan meeting:
      lengan ganda, RealSense, OCR tanggal kedaluwarsa, barcode sebagai cadangan. RealSense D455 sekarang memang
      dipakai (sejak 08-10), tapi itu pilihan perangkat, bukan keputusan meeting.
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
  | `angiolite 4x19` | SIAP, K10 (sejak 08-10 K10 kembali ke rak; sebelumnya HABIS saat K10 diambil) |
  | `angiolite 3 x 20` | HABIS (tidak ada di inventaris) |
  | `angiolite 2.5` | ULANGI (panjang tidak ada) |
  | `sled bag` | SIAP, kotak dengan kedaluwarsa paling awal (barang tanpa ukuran) |
  | `terumo xperience 2.5 x 15` | ULANGI (merek salah) |

- [x] Paket server dibuat: `box_pipeline_demo_20261007.zip`.

## Belum

- [x] Jalankan `python scripts/uji_checklist.py` di server: **20 lulus, 0 gagal** (08-10, sebelum K10 kembali;
      `results/uji_checklist_20261007_235241.md`). Setelah K10 kembali (08-10, sore): **19 lulus, 0 gagal, 1 dilewati**
      (#17 dilewati otomatis selama K10 ada di rak — lihat tabel B/A).
- [x] Kode lewat `git clone`, data lewat `data_demo_20261007.zip` (36/36 file cocok), model `models/box_v4s.pt`
      (sha256 `21c4bf01…`). `pytest` di server: **205 lulus**. Perlu `pip install pytesseract==0.3.*` di `trse_venv`.
- [x] Gladi teks di server: keenam skenario demo memberi hasil yang sama dengan laptop.
- [ ] Gladi demo lewat Remmina, lalu rekam layarnya sebagai cadangan.
- [x] Mode kamera di program demo: `--kamera` meneruskan status SIAP ke `test_webcam_box.py` (diuji sampai skrip kamera terbuka; belum dengan kamera/stream asli).
- [ ] Uji mode kamera sungguhan dengan RealSense D455 (lihat `docs/UJI_LIVE_LOKAL.md` untuk laptop-saja, atau
      `docs/UJI_LIVE_SERVER.md` untuk lewat Remmina). Percobaan 08-10 di laptop dengan 20 kotak dalam satu frame:
      0 AMBIL dalam 145 detik (teks terlalu kecil) — perlu kamera didekatkan, belum diuji ulang.
- [ ] Laporan Jumat: draf di `docs/LAPORAN_PROGRES_20261009.md` (tabel status, hasil uji, model, risiko, pertanyaan).
      Perlu diperbarui: baris K10/#17 di draf itu masih versi lama (sebelum K10 kembali), dan angka v6 perlu dikoreksi
      (lihat catatan v6 di bawah).
- [x] v6 dilatih dan dievaluasi (08-10). **Koreksi penting:** skor CV yang benar-benar sebanding dengan v4 adalah
      mAP50-95 **0,706 ± 0,032** (v4: 0,696 ± 0,044) — naik tipis, bukan lompatan besar. Deteksi kotak nyata (out-of-fold):
      v4 100% (123/123), v6 95,1% (117/123, 4 foto malah pakai bobot cadangan). 0 salah MATCH tetap terjaga.
      **Tidak dipakai**; demo tetap v4. Detail: `D_model_v6/best_model.json` (paket 08-10).
- [ ] Ganti `posisi.csv` contoh dengan data rak/level dari tim.
- [ ] Audit Codex (4 Okt, `prelabel_size_label_full`) menemukan 8 hal yang perlu dibenahi di pipeline label: 7 foto
      masih "perlu dicek manual" tapi ikut terhitung training, 25 pasang foto berurutan bocor antara train/val, dan
      cache training yang tidak selalu batal saat eksperimen berubah. Detail: `B_prelabel_perbaikan_tools/LAPORAN_AUDIT.md`
      (paket 08-10). 7 foto sedang dikerjakan ulang.

## Daftar 20 uji (isi kolom Hasil dan OK saat menguji)

Otomatis (uji teks 1-20, laporan ke `results/`): `python scripts/uji_checklist.py`. Hasil di laptop dan server
(sebelum K10 kembali): **20 lulus, 0 gagal**. Sejak K10 kembali (08-10): **19 lulus, 0 gagal, 1 dilewati** (#17).
Kamera per nomor: `python scripts/uji_checklist.py --kamera N` (N = 1-16, kotak fisik di depan kamera; N = 17
juga bisa selama K10 ada di rak — lihat `UJI_LIVE_LOKAL.bat`/`UJI_LIVE_LOKAL.md`).

A. Barang ada di rak: harus **SIAP** (teks), lalu uji kamera untuk nomor 1-17.

| # | Permintaan | Kotak | Catatan | Hasil teks | Hasil kamera | OK |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `angiolite 2.5 x 29` | K15 | sejak 08-10 K10 (angiolite 4 x 19) kembali di rak: pasangan mirip, lihat #17 | SIAP K15 (server, LULUS) |  | [ ] |
| 2 | `xperience pro 2.5 x 15` | K06 |  | SIAP K06 (server, LULUS) |  | [ ] |
| 3 | `essential pro 3 x 40` | K04 | ukuran panjang | SIAP K04 (server, LULUS) |  | [ ] |
| 4 | `accuforce 2.5 x 12` | K07 | mirip #5/#6 | SIAP K07 (server, LULUS) |  | [ ] |
| 5 | `accuforce 2.75 x 20` | K16 | mirip #4/#6 | SIAP K16 (server, LULUS) |  | [ ] |
| 6 | `accuforce 3.5 x 15` | K21 | mirip #4/#5 | SIAP K21 (server, LULUS) |  | [ ] |
| 7 | `ryurei 2 x 10` | K09 | ukuran terkecil | SIAP K09 (server, LULUS) |  | [ ] |
| 8 | `nagomi 2.75 x 33` | K13 | mirip #9 | SIAP K13 (server, LULUS) |  | [ ] |
| 9 | `nagomi 3 x 24` | K18 | kedaluwarsa 2028-01-31 | SIAP K18 (server, LULUS) |  | [ ] |
| 10 | `nc emerge 5.5 x 12` | K08 | kedaluwarsa 2028-01-03 | SIAP K08 (server, LULUS) |  | [ ] |
| 11 | `conqueror 3.5 x 15` | K11 | mirip #12 | SIAP K11 (server, LULUS) |  | [ ] |
| 12 | `conqueror 4 x 8` | K12 | mirip #11 | SIAP K12 (server, LULUS) |  | [ ] |
| 13 | `sapphire 3 x 15` | K14 |  | SIAP K14 (server, LULUS) |  | [ ] |
| 14 | `scoreflex 3 x 15` | K19 |  | SIAP K19 (server, LULUS) |  | [ ] |
| 15 | `agent 2.75 x 30` | K20 |  | SIAP K20 (server, LULUS) |  | [ ] |
| 16 | `permanent sled bag` | K03 | 4 kotak (K01,K03,K05,K17); uji FEFO: kedaluwarsa 2029-03-10 paling awal | SIAP K03 (server, LULUS) |  | [ ] |
| 17 | `angiolite 4 x 19` | K10 | sejak 08-10 K10 kembali di rak; pasangan mirip dengan #1 — kasus paling berisiko tertukar | SIAP K10 (laptop, LULUS, 08-10) |  | [ ] |

B. Harus ditolak atau tidak ditemukan (uji teks saja, tanpa kamera).

| # | Permintaan | Hasil yang diharapkan | Catatan | Hasil teks | OK |
| --- | --- | --- | --- | --- | --- |
| 18 | `accuforce 2.75 x 15` | TIDAK ADA, kombinasi tidak ada | mirip #5 dan #6 | TIDAK ADA (server, LULUS) | [x] |
| 19 | `terumo xperience 2.5 x 15` | ULANGI, merek tidak cocok dengan produk |  | ULANGI (server, LULUS) | [x] |
| 20 | `essential pro 3` | ULANGI, panjang belum ditulis |  | ULANGI (server, LULUS) | [x] |

Aturan lulus uji kamera: tidak boleh ada MATCH pada kotak yang salah. Kalau ragu, hasilnya harus PERLU KONFIRMASI.

## Pertanyaan untuk tim

1. Format file posisi rak/level: kolom apa saja, dan per kotak atau per jenis barang?
2. Bentuk data yang diterima software TM: JSON, teks, atau socket/ROS? Sementara ini kita pakai `to_tm_message()`
   (lihat contoh di `docs/LAPORAN_PROGRES_20261009.md` bagian 1) - format **sementara**, belum dikonfirmasi tim.
3. ~~Siapa yang memegang antarmuka konfirmasi suara~~ **Keputusan 08-10: pakai antarmuka TEKS dulu untuk demo**
   (`demo_request.py`); UI suara (warna + "execute") ditunda sampai pemiliknya ditentukan tim.

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

# Sumber Data untuk Deteksi Buku, Rak (Shelf) & Pena

Pendamping `notebooks/train_book_on_kaggle.ipynb`, mengikuti pola yang sama dengan
`docs/DATASET_SOURCES.md` (domain obat). Ditambahkan setelah meninjau dek
**"Overall Progress NSTC – Mackay"** (Hucenrotia Lab / NYCU): foto rak farmasi nyata mereka
menunjukkan kemasan berdiri tegak, berjajar rapat, punggung menghadap kamera — visualnya sama
persis dengan buku di rak buku, dan mereka sendiri menandai kondisi ini sebagai kelemahan
("Non-Box Workpieces", "Uncertain Positioning"). Dokumen ini menjelaskan cara memperluas
pipeline deteksi proyek ini (tanpa mengubah kode) ke domain buku, sebagai pembuktian bahwa
arsitekturnya generik.

**Revisi:** versi dokumen ini sebelumnya merekomendasikan 2 dataset publik generik untuk
diunduh manual. Anda kemudian mengunggah **10 file dataset Roboflow nyata** yang sudah Anda
kumpulkan sendiri (termasuk 1 proyek Roboflow milik Anda sendiri) — dokumen ini sudah ditulis
ulang sepenuhnya berdasarkan pemeriksaan langsung terhadap 10 file tersebut, bukan lagi
rekomendasi generik.

## Level 0 — sudah jalan HARI INI, tanpa dataset atau training apa pun

Model placeholder yang sudah ada di proyek ini, `models/yolo26n.onnx`, adalah **YOLO26n
COCO-pretrained** (80 kelas umum). Salah satu dari 80 kelas itu **adalah `"book"`** — cek
langsung di `src/pipeline.py`, daftar `COCO_CLASSES`. Artinya, tanpa mengunduh dataset apa
pun atau melatih apa pun:

```bash
python scripts/test_webcam.py --config config/book_pipeline_config.yaml
```

...sudah bisa mendeteksi buku secara generik di kamera Anda hari ini. Ini bukan solusi akhir
(akurasinya seadanya COCO, tidak dilatih khusus kondisi rak Anda, dan COCO tidak punya kelas
"shelf" atau "pen" sama sekali), tapi ini validasi pipeline paling murah yang bisa dilakukan
sebelum menghabiskan waktu/kuota GPU untuk training.

## Level 1 — 10 dataset nyata yang Anda upload, diperiksa satu per satu

Setiap file Roboflow diekspor dengan nama kelas yang **tidak selalu bersih** (beberapa berisi
judul proyek + tanggal ekspor, bukan nama kelas asli, atau placeholder `"-"` untuk kelas versi
lama yang sudah tidak dipakai). Supaya penggabungan tidak salah kelas, tiap file diperiksa
lewat 3 hal — bukan cuma dipercaya dari nama filenya:

1. isi `data.yaml` (jumlah kelas, nama kelas tertulis),
2. provenance asli (`roboflow.workspace` / `roboflow.project` — tertulis di dalam file,
   tidak berubah walau file di-rename/diunggah ulang),
3. **distribusi pemakaian tiap index kelas di file label asli** (`.txt`) — ini satu-satunya
   cara memastikan arti index kelas yang sebenarnya ketika nama kelasnya korup.

| # | File yang Anda upload | Proyek Roboflow asli | Kelas asli (index: nama) | Objek terpakai | Dipetakan ke |
|---|---|---|---|---|---|
| 1 | `book.v1i.yolo26(1).zip` | zebra-learn/book-4abtl v1 | 0: book | 2.384 | `book` |
| 2 | `book.v1i.yolo26.zip` | **saca-setya-wardana**/book-omfeo-r6slr v1 (proyek Anda sendiri) | 0: book | 1.881 (poligon) | `book` |
| 3 | `book.v2i.yolo26(1).zip` | yrden/book-zbbr0 v2 | 0: "book_image - v6..." (nama proyek, bukan kelas asli) | 52.095 | `book` |
| 4 | `book.v2i.yolo26.zip` | library-rjncz/book-6wkuh v2 | 0 & 1: nama korup ("---...", "book segmentation - v5...") | 36.101 (poligon) | `book` (keduanya) |
| 5 | `Book spline detection.v1i.yolov8.zip` | books-26cz6/book-spline-detection v1 | 0: "Book spine..." (dominan), 1: "object" (129x, noise) | 22.852 (poligon) | `book` (keduanya) |
| 6 | `Book_Pen_Model.v1i.yolo26.zip` | working-vce8g/book-pen-model v1 | 0: Book, 1: Pen (bersih) | 2.707 + 380 | `book` + `pen` |
| 7 | `book.v2-roboflow-instant-1--eval-.yolo26.zip` | sadang/book-bzuxs v2 | 0: book (3x), 1: books (808x) | 811 | `book` (keduanya) |
| 8 | `shelf.v3i.yolo26.zip` | university-of-science-vnu-zlz2c/shelf-tzfro v3 | 0: shelf (bersih) | 518 | `shelf` |
| 9 | `shelf.v2-shelfv5.yolo26.zip` | robocup2022-kogzd/shelf-6pwdx v2 | 0: shelf (bersih) | 573 | `shelf` |
| 10 | `BOOK.v1i.yolo26.zip` | blueworkspace/book-c8llt v1 | 0: "-" (2x), **1: "1dcode" (890x, dominan)** | 892 | **dikecualikan** |

**Total yang digabung (file #1-9, diverifikasi lewat simulasi merge lokal, bukan estimasi):**

| Metrik | Nilai |
|---|---|
| Total gambar | **7.931** |
| Total objek berlabel (setelah poligon → bbox) | **120.302** |
| Baris label rusak/tidak terbaca | **0** |
| Objek kelas `book` | **118.831** |
| Objek kelas `shelf` | **1.091** |
| Objek kelas `pen` | **380** |

**Semua 10 dataset berlisensi CC BY 4.0** — atribusi wajib bila dipakai untuk model akhir;
daftar workspace/project di tabel di atas sudah cukup sebagai kutipan (lengkap dengan link
per proyek ada di Bagian 9 laporan `.docx`).

### Kenapa file #10 dikecualikan (dan kenapa TIDAK dibuang begitu saja)

Meski dinamai "BOOK.v1i", isi labelnya didominasi kelas `1dcode` (kode batang 1 dimensi /
barcode di sampul buku), bukan objek "buku" itu sendiri — kalau ikut digabung sebagai `book`,
model akan salah belajar (bentuk barcode dianggap bentuk buku). Dataset ini **disimpan
terpisah, bukan dihapus**: tahap "Verification" pada visi 7-tahap NSTC-Mackay (OCR + barcode
GS1) saat ini masih berstatus "belum ada" di proyek ini (lihat laporan `.docx`, Bagian 6) —
dataset ini adalah aset siap-pakai kalau/ketika kebutuhan verifikasi barcode itu digarap.

### Kenapa file #3 dan #4 kelas aslinya "korup" — dan kenapa aman dipetakan ke `book`

Ini pola umum Roboflow: kalau sebuah project cuma 1 kelas dan pernah di-rename/re-export,
nama kelas kadang tertimpa jadi judul proyek + tanggal ekspor, atau (kasus #4) placeholder
`---` untuk kelas versi lama yang sudah tidak dipakai tapi datanya masih ada. Karena sudah
dicek langsung distribusi pemakaian index (bukan cuma nama), pemetaan ke `book` di
`notebooks/train_book_on_kaggle.ipynb` (tabel `REMAP`, dikunci per `(workspace, project)`,
bukan per nama file) aman dilakukan.

### Ketimpangan kelas (`class imbalance`) — apa artinya untuk hasil training

Kelas `book` (118.831 objek) jauh lebih banyak daripada `shelf` (1.091) dan `pen` (380).
Ini konsekuensi wajar dari menggabungkan dataset publik yang memang fokus ke buku, ditambah
2 dataset shelf yang lebih kecil dan 1 dataset pen kecil sebagai "bonus". Implikasi praktis:

- Deteksi `book` kemungkinan besar akurat sejak training pertama (data melimpah).
- Deteksi `shelf` dan `pen` kemungkinan **butuh lebih banyak epoch atau data tambahan**
  sebelum recall-nya sebaik `book` — ini normal untuk kelas minoritas, bukan tanda kode salah.
- Setelah training, cek metrik **per kelas** (bukan cuma mAP50 rata-rata) di sel "Validasi
  hasil" pada notebook — kalau `shelf`/`pen` jauh di bawah `book`, itu sinyal untuk menambah
  data kelas tersebut nanti, bukan mengubah arsitektur pipeline.
- Kalau tujuan utama Anda hanya "deteksi buku + hitung per tingkat rak", kelas `shelf`/`pen`
  yang lebih lemah tidak menghalangi kegunaan model — keduanya kelas tambahan (bonus), bukan
  kebutuhan inti permintaan awal ("model buku dan rak buku").

## Kenapa "rak buku" SEKARANG bisa dilatih sebagai kelas YOLO — dan kenapa itu TIDAK mengganti pendekatan tier-scan

**Koreksi dari versi dokumen sebelumnya:** versi awal dokumen ini menyatakan tidak ditemukan
dataset publik yang melabeli rak sebagai objek. Itu keliru — 2 dari 10 file yang Anda upload
(#8 dan #9 di tabel atas) adalah dataset `shelf` yang bersih, masing-masing 1 kelas saja,
total 1.091 objek berlabel. Jadi `shelf` sekarang benar-benar menjadi salah satu dari 3 kelas
YOLO yang dilatih di `notebooks/train_book_on_kaggle.ipynb`.

**Tapi ini tidak menggantikan `src/shelf_scanner.py`.** Alasannya bukan soal ada/tidaknya
data, melainkan soal **pertanyaan yang dijawab berbeda**:

- Kelas `shelf` (YOLO) menjawab: *"apakah ada struktur rak yang terlihat di sini?"* — berguna
  sebagai sanity-check visual atau bantuan kalibrasi (mis. mendeteksi otomatis di mana batas
  atas/bawah rak dalam bingkai kamera), tapi **tidak tahu rak itu terbagi berapa tingkat**.
- `src/shelf_scanner.py` menjawab: *"buku/obat yang terdeteksi ini ada di TINGKAT rak yang
  mana?"* — ini pertanyaan geometris (posisi relatif dalam bingkai kamera dibagi
  `shelf.tiers`/`tier_bounds`), yang tidak bisa dijawab hanya dari satu bounding box "shelf".

Jadi kedua pendekatan **saling melengkapi, bukan saling menggantikan**: deteksi `shelf` YOLO
adalah kemampuan baru yang didapat "gratis" dari data yang Anda kumpulkan (bisa dipakai nanti
sebagai bantuan kalibrasi/validasi), sementara logika tier-scan yang sudah ada dan sudah lulus
uji (`tests/test_shelf_scanner.py`) tetap menjadi mekanisme utama untuk menjawab "tingkat
berapa" — **tidak ada kode di `src/` yang perlu diubah**. `config/book_pipeline_config.yaml`
bagian `shelf` tetap menunjuk ke jumlah/posisi tingkat rak buku fisik Anda, diisi manual
seperti sebelumnya (kalibrasi otomatis dari kelas `shelf` adalah pengembangan lanjutan
opsional, belum diimplementasikan).

## Cara pakai end-to-end (identik alurnya dengan domain obat)

```
Level 0: python scripts/test_webcam.py --config config/book_pipeline_config.yaml
                (langsung jalan, kelas "book" bawaan COCO)
                        |
                        v  (kalau akurasi kurang cukup, atau butuh kelas shelf/pen)
        9 file dataset yang sudah Anda upload (lihat tabel di atas, semua SUDAH DI TANGAN --
        tidak perlu diunduh lagi) -> lihat docs/JALUR_B_CHECKLIST_BUKU.md untuk urutan upload
                        |
                        v
        upload ke Kaggle sebagai Kaggle Dataset(s) / + Add Input
                        |
                        v
        notebooks/train_book_on_kaggle.ipynb  (GPU gratis Kaggle T4/P100, 3 kelas: book/shelf/pen)
                        |
                        v
        best.onnx + classes.txt  (download dari panel Output Kaggle)
                        |
                        v
        copy ke models/book_yolo26n.onnx + data/book_classes.txt
                        |
                        v
        python scripts/test_webcam.py --config config/book_pipeline_config.yaml \
            --onnx models/book_yolo26n.onnx --classes data/book_classes.txt
        python scripts/run_shelf_search.py --config config/book_pipeline_config.yaml \
            --target <id> --live --image foto_rak_buku.jpg
```

## Katalog per-judul (opsional, analog `medicine_db.sqlite`)

Kalau tujuan Anda berkembang dari "deteksi + hitung buku per tingkat" menjadi "cari judul buku
X di rak", Anda perlu katalog referensi per judul — sama seperti `scripts/init_db.py` untuk
obat (nama, teks label untuk OCR, histogram warna sampul). `config/book_pipeline_config.yaml`
sudah menyiapkan field `database.path` untuk ini; buat skrip serupa `init_db.py` kalau
kebutuhan itu muncul. Untuk sekarang (fokus permintaan: deteksi buku & rak buku), bagian ini
tidak wajib diisi — `scripts/test_webcam.py` tidak memakainya sama sekali, dan
`run_shelf_search.py` tetap berjalan tanpanya (hanya tidak akan menemukan match "judul
dikenal").

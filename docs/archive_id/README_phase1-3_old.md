# Medicine Shelf Vision System

Implementasi kode untuk Fase 1–3 dari roadmap di dokumen desain **"Vision System
Design: Medicine Shelf Robot on Lenovo V14 G2 ITL"** (bagian 10). Dibangun mengikuti
arsitektur di bagian 2 dokumen itu: Kamera → Model Visi Ringan (YOLO-nano, ONNX) →
Verifikasi Objek (OCR + warna + bentuk) → Database Lokal → Estimasi Posisi 3D.

## Status jujur: apa yang sudah teruji vs yang masih perlu hardware/data asli Anda

Kode ini ditulis dan **diuji end-to-end di sandbox cloud tanpa GPU, tanpa webcam,
dan tanpa kamera RealSense** — jadi ada dua lapis pengujian yang perlu dibedakan:

1. **Kebenaran kode terbukti** — `tests/test_detector_accuracy.py` membandingkan
   hasil decoder ONNX custom (`src/detector.py`) langsung terhadap
   `ultralytics.YOLO(...).predict()` pada foto asli (bus.jpg bawaan Ultralytics):
   **5/5 deteksi cocok persis** (kelas, confidence, kotak — lihat log run terakhir).
   Ini membuktikan matematika letterbox/NMS/decode-nya benar.
2. **Domain obat belum diuji dengan model/dataset asli** — karena sandbox ini tidak
   punya foto kemasan obat asli, `scripts/generate_test_assets.py` membuat gambar
   sintetis (kotak digambar + teks) hanya untuk memvalidasi *jalur kode* (I/O,
   OCR, database lookup, matematika depth) tidak crash — **bukan** untuk mengklaim
   sistem sudah bisa mendeteksi obat sungguhan. Model yang dipakai saat ini adalah
   YOLO11n **COCO-pretrained** (80 kelas umum), sesuai catatan di
   `scripts/export_model.py`. Anda tetap perlu mengumpulkan dataset kemasan obat
   Anda sendiri dan melatihnya (bagian 4 dokumen desain, `scripts/train.py` sudah
   disiapkan) sebelum ini benar-benar mengenali SKU obat.

Singkatnya: **infrastrukturnya sudah lengkap dan teruji jalan; bagian yang secara
inheren butuh Anda adalah dataset foto obat + hardware Anda** (laptop MX350, kamera,
lengan robot) — itu bukan sesuatu yang bisa disubstitusi dari sandbox ini.

## Struktur proyek

```
medicine_shelf_vision/
├── config/pipeline_config.yaml   # semua parameter (model, kamera, threshold, depth) — domain OBAT
├── config/book_pipeline_config.yaml  # NEW: profil sama, domain BUKU & rak buku (lihat bagian di bawah)
├── src/
│   ├── detector.py        # inferensi YOLO ONNX (bagian 2, 3)
│   ├── verifier.py         # OCR + warna + bentuk vs database (bagian 5)
│   ├── sku_retrieval.py    # NEW: retrieval vektorisasi numpy untuk katalog besar (>500 SKU, target 110k)
│   ├── database.py         # SQLite medicine DB (bagian 2)
│   ├── depth_estimator.py  # depth->3D: RealSense asli + mode mock (bagian 6, 8)
│   ├── pipeline.py         # orkestrator Camera->Detector->Verifier->Depth
│   ├── shelf_scanner.py    # NEW: cari SKU target di rak 4-tingkat, laporkan tingkat mana (bagian 7 diperluas)
│   └── utils.py             # abstraksi sumber frame (webcam/video/sintetis) + drawing
├── scripts/
│   ├── generate_test_assets.py  # buat gambar uji sintetis (karena sandbox ini tanpa kamera)
│   ├── init_db.py                # isi database SQLite dengan contoh 3 SKU
│   ├── export_model.py           # .pt -> ONNX
│   ├── train.py                  # scaffold training pada dataset obat Anda sendiri
│   ├── run_phase1.py             # Kamera -> Deteksi -> Tampilan
│   ├── run_phase2.py             # + OCR + Database
│   ├── run_phase3.py             # + Depth + Posisi 3D
│   ├── run_shelf_search.py       # NEW: "cari paracetamol, ada di tingkat berapa?" (rak 4 tingkat)
│   ├── convert_roboflow_dataset.py  # NEW: konversi label poligon Roboflow -> bbox (1 dataset)
│   ├── merge_datasets.py         # NEW: gabung beberapa dataset Roboflow jadi 1 (union kelas + reindex label)
│   └── test_webcam.py            # NEW: jalankan di laptop Anda sendiri — uji live webcam + FPS
├── notebooks/
│   ├── train_on_kaggle.ipynb     # NEW: notebook siap-pakai, gabung 3 dataset Pharmacy + training YOLO26n di GPU gratis Kaggle
│   └── train_book_on_kaggle.ipynb # NEW: kloning notebook di atas untuk domain buku & rak buku
├── docs/
│   ├── DATASET_SOURCES.md        # NEW: daftar dataset publik + cara kumpulkan foto SKU sendiri (obat)
│   └── DATASET_SOURCES_BUKU.md   # NEW: idem, domain buku & rak buku
├── tests/
│   ├── test_components.py         # verifier+db+depth, tanpa perlu YOLO/kamera
│   ├── test_detector_accuracy.py  # decoder ONNX vs referensi Ultralytics (foto asli)
│   ├── test_shelf_scanner.py      # NEW: pencarian tingkat rak, kasus tajam & jauh/blur — LULUS keduanya
│   ├── test_sku_retrieval.py         # NEW: skor retrieval vektorisasi vs cv2.compareHist — identik (selisih <1e-6)
│   ├── test_large_catalog_verifier.py # NEW: pipeline verify() dua-tingkat pada katalog 4003 SKU — SKU asli tetap ditemukan
│   ├── test_roboflow_converter.py    # NEW: konversi poligon->bbox pada kasus buatan
│   └── test_merge_datasets.py        # NEW: union kelas + reindex label lintas-dataset — 8/8 lulus
└── data/  (dibuat otomatis oleh script-script di atas)
```

## Model deteksi: YOLO26n (diganti dari YOLO11n)

Detektor default sekarang **YOLO26n** (`models/yolo26n.onnx`, dirilis Ultralytics Januari
2026) — diuji langsung di sandbox ini: **40.6ms CPU** vs 47.6ms YOLO11n, mAP 40.9 vs
39.5, arsitektur NMS-free (lebih ringan untuk MX350). Output ONNX-nya identik bentuk
(1, 84, 8400) dengan YOLO11n, jadi `src/detector.py` tidak perlu diubah sama sekali.
Butuh `ultralytics>=8.4.0` (lihat `requirements.txt`). Kalau versi ultralytics Anda
lebih lama, `scripts/export_model.py --weights yolo11n.pt` tetap tersedia sebagai
fallback.

## Dataset training: 4 dataset Roboflow "Pharmacy" dikumpulkan, 3 digabung

Selain "Pharmacy v7" (44 SKU) yang sudah diverifikasi sebelumnya, 3 dataset Roboflow
tambahan sudah diperiksa:

| Dataset | Kelas | Gambar | Status |
|---|---|---|---|
| Pharmacy v7 (`sito-s02gz/pharmacy-bk2qv`) | 44 SKU nyata | 1454 | Digabung |
| pharmacy-shelf-detect v1 | 4 SKU nyata (Panadol_coldflu, alphintern, augmentin_1gm, panadol_extra) | 411 | Digabung |
| pharmacy robot v2 | 4 SKU nyata (panadol, paymol, revanin, riva-n) | 414 | Digabung |
| Pharmacy v3 (`object-detection-hhx5h/pharmacy-jl5t1`) | 10 kelas berlabel angka `1`-`10` saja | 1538 | **Dikeluarkan** -- tidak ada identitas SKU |

`scripts/merge_datasets.py` (diuji: `tests/test_merge_datasets.py`, 8/8 lulus) menggabungkan
3 dataset yang bisa dipakai menjadi **52 kelas gabungan**, menulis ulang index label tiap
dataset ke index gabungan (bukan asal salin folder -- itu akan bikin model belajar label
yang salah). Diverifikasi lokal: 2279 gambar, 8182 objek, 0 kesalahan index.

**3 kemiripan nama lintas-dataset yang perlu keputusan Anda** (sengaja TIDAK digabung
otomatis jadi satu kelas, karena tidak ada cara aman menebak dari nama saja):
`alphain` (v7) vs `alphintern` (shelf-detect); `panadol` (robot) vs `Panadol_coldflu`
dan vs `panadol_extra` (keduanya dari shelf-detect). Kalau Anda tahu ini produk yang
sama, beri tahu supaya di-merge manual jadi satu kelas.

**Catatan:** dataset "pharmacy robot v2" cuma py split `train` (sudah kena augmentasi
Roboflow: flip/crop/rotasi/brightness/blur), jadi 4 kelasnya tidak akan muncul di
mAP validasi -- ikut training saja. `notebooks/train_on_kaggle.ipynb` melakukan seluruh
konversi + penggabungan ini otomatis di Kaggle (dari file zip asli yang Anda upload),
karena hasil gabungannya (~264MB) terlalu besar untuk dikirim lewat chat.

## Skala katalog 110.000 SKU: kenapa perlu SkuRetrievalIndex

Target katalog dikonfirmasi ~110.000 SKU. Loop pencocokan warna per-record yang
dipakai `verifier.py` sejak awal (`cv2.compareHist` satu per satu) terukur
**~39ms/objek/frame** di skala ini (lihat `scripts/benchmark_sku_scale.py`) — lebih
lambat dari anggaran satu frame real-time (~33ms @ 30fps), dan itu BELUM termasuk
shape matching/OCR. `src/sku_retrieval.py` (`SkuRetrievalIndex`) memperbaikinya
dengan satu operasi matriks numpy yang skornya **matematis identik** dengan
`cv2.compareHist` (dibuktikan di `tests/test_sku_retrieval.py`, selisih maks
7.77e-16), mempersempit ke top-K kandidat (default 50) dalam **~2.1ms bahkan di
110.000 SKU** (terukur langsung, bukan ekstrapolasi) sebelum OCR/shape matching
yang mahal per-item dijalankan hanya pada shortlist itu. `verifier.py` otomatis
memakai jalur ini di atas 500 SKU (`large_catalog_threshold`); di bawah itu (termasuk
`medicine_db.sqlite` yang ada sekarang, 3 SKU) perilakunya persis sama seperti
sebelumnya. **Catatan jujur:** ini baru diuji pada katalog sintetis (4000 SKU
pengecoh), belum pada 110.000 SKU nyata dengan foto obat asli — dan barcode (belum
diimplementasikan) tetap direkomendasikan sebagai lapisan pertama paling andal di
skala ini, bukan mengandalkan visual saja.

## Hardware robot: sudah dikonfirmasi (belum diintegrasikan)

- **Lengan**: Igus ReBel cobot, 6 DOF, payload 2kg, reach maks 664mm / nominal 400mm.
- **AMR**: A100 (uni-innovate), payload 100kg, dimensi 480x480x250mm.

Payload kedua perangkat jauh melebihi kebutuhan (obat + wadah biasanya <500g), jadi
aman dari sisi kapasitas. Pertanyaan terbuka yang butuh pengukuran lapangan (bukan
sesuatu yang bisa dijawab dari spesifikasi saja): apakah reach 664mm lengan cukup
menjangkau ke-4 tingkat rak dari satu titik docking AMR, atau perlu aktuasi vertikal
tambahan/beberapa titik docking. Kode kontrol (ROS2/MoveIt2) untuk Fase 4 belum
dibangun — lihat bagian "Yang BELUM ada di paket ini" di bawah.

## Instalasi

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
sudo apt install tesseract-ocr tesseract-ocr-eng   # untuk pytesseract
```

**Di laptop Lenovo+MX350 Anda** (bukan di sandbox ini), untuk inferensi dipercepat GPU:
mengikuti bagian 9 dokumen desain, install `onnxruntime-gpu` (bukan `onnxruntime` biasa)
dengan **CUDA 12.6 atau 12.8** (BUKAN CUDA 13.x — CUDA 13 menghapus dukungan Pascal/MX350
sepenuhnya). `config/pipeline_config.yaml` sudah diset agar otomatis pakai
`CUDAExecutionProvider` kalau tersedia, dan jatuh ke CPU kalau tidak (seperti di sandbox ini).

## Cara menjalankan (urutan yang sudah diuji)

```bash
# 1. Siapkan model (placeholder COCO — ganti dengan model obat Anda setelah training)
python scripts/export_model.py

# 2. Siapkan data uji (di laptop asli dengan kamera, lewati langkah ini dan set
#    camera.source: webcam di config/pipeline_config.yaml)
python scripts/generate_test_assets.py
python scripts/init_db.py

# 3. Validasi komponen (tidak butuh kamera/GPU)
python tests/test_components.py
python tests/test_detector_accuracy.py

# 4. Jalankan tiap fase
python scripts/run_phase1.py --frames 10
python scripts/run_phase2.py --frames 10
python scripts/run_phase3.py --frames 10
# tambahkan --show kalau punya display (bukan headless) untuk lihat jendela live
```

## Pencarian tingkat rak (rak 4 tingkat, cari SKU tertentu)

Skenario yang Anda tambahkan: satu rak berisi campuran botol pil, tablet, dan
kemasan dus di **4 tingkat**, dan robot perlu tahu **tingkat mana** yang ada
obat target (mis. Paracetamol) sebelum lengan bergerak mendekat untuk
konfirmasi jarak-dekat (Stage 2). Ini diimplementasikan di `src/shelf_scanner.py`
dan diuji di `tests/test_shelf_scanner.py` — **LULUS untuk kasus tajam/dekat
MAUPUN jauh/blur** (pada kasus blur, OCR gagal karena teks buram, tapi sistem
otomatis jatuh ke pencocokan warna+bentuk dan tetap menemukan tingkat yang benar
— sesuai ekspektasi Anda bahwa akurasi dari jauh lebih rendah tapi tetap cukup
untuk memilih tingkat, karena Stage 2 akan mengonfirmasi dari dekat).

```bash
python tests/test_shelf_scanner.py                          # validasi logika (ground-truth sintetis)
python scripts/run_shelf_search.py --target MED-001          # demo: cari di kondisi dekat/tajam
python scripts/run_shelf_search.py --target MED-001 --far    # demo: cari di kondisi jauh/blur
python scripts/run_shelf_search.py --target MED-001 --live --image foto_rak_asli.jpg  # dgn model & foto asli
```

Catatan kalibrasi: `config/pipeline_config.yaml` bagian `shelf.tier_bounds` default
membagi bingkai kamera jadi 4 pita sama tinggi. Setelah rak fisik Anda diukur,
set ini secara eksplisit sebagai rasio tinggi bingkai (contoh ada di komentar
file config) untuk hasil pembagian tingkat yang lebih akurat. Pengambilan/grasping
itu sendiri **sengaja tidak** termasuk di modul ini — sesuai catatan Anda, itu
tugas terpisah setelah lengan sudah diarahkan ke tingkat yang benar.

## Jalur A: dari laptop kosong sampai model obat sungguhan jalan

Panduan langkah-demi-langkah lengkap (dengan cara cek keberhasilan tiap langkah) ada di
`docs/JALUR_A_CHECKLIST.md`. Ringkasannya:

```bash
python scripts/check_setup.py     # LANGKAH 1 BARU — cek semua dependensi/kamera/GPU/tesseract
                                   # dulu, sebelum pegang kamera, supaya error jelas bukan crash
python scripts/test_webcam.py     # uji kamera hidup + kumpulkan foto (tombol 's')
```

**Catatan penting:** `requirements.txt` memakai `opencv-python` (BUKAN
`opencv-python-headless`) karena `test_webcam.py` membuka jendela GUI (`cv2.imshow`) —
build headless tidak punya GUI backend sama sekali dan akan gagal di langkah ini.
`scripts/check_setup.py` memeriksa ini otomatis.

## Melatih model obat Anda sendiri (bagian 4 dokumen desain)

1. **Kumpulkan foto** kemasan obat Anda sendiri — jalankan `scripts/test_webcam.py`
   di **laptop Anda** (bukan di sandbox ini, yang tanpa kamera) untuk sekaligus
   memverifikasi pipeline jalan live di kamera+MX350 Anda DAN menyimpan foto
   (tombol `s`) sebagai awal dataset. Lihat `docs/DATASET_SOURCES.md` untuk
   daftar dataset publik (Kaggle/Roboflow) yang bisa dikombinasikan, dan berapa
   banyak foto per SKU yang realistis (~100-300).
2. Label bounding box format YOLO (Roboflow/LabelImg/CVAT), buat `data.yaml`
   (lihat contoh di `notebooks/train_on_kaggle.ipynb`).
3. **Latih di GPU cloud gratis Kaggle** — buka `notebooks/train_on_kaggle.ipynb`
   di Kaggle Notebooks (Settings -> Accelerator -> GPU T4), Run All. Notebook ini
   sudah lengkap: cek GPU, install ultralytics, load dataset, training 120 epoch,
   validasi mAP, export ke ONNX, dan bungkus hasil jadi zip siap-download.
   Alternatif command-line (kalau training di mesin lain yang punya GPU):
   ```bash
   python scripts/train.py --data path/ke/data.yaml --epochs 120
   ```
4. Download `medicine_model_export.zip` dari Kaggle, copy `best.onnx` &
   `classes.txt` ke `models/` dan `data/` di laptop Anda.
5. Update `scripts/init_db.py` dengan SKU obat Anda sendiri (ganti 3 contoh SKU),
   supaya tahap verifikasi hybrid (bagian 5) punya referensi nama/warna/bentuk
   yang sinkron dengan kelas yang dikenali model baru.
6. Uji ulang: `python scripts/test_webcam.py --onnx models/medicine_yolo11n.onnx --classes data/classes.txt`
   lalu `python scripts/run_shelf_search.py --target <SKU> --live --image <foto_rak>`.

## Menyalakan kamera & depth asli (bukan mode sintetis/mock)

Di `config/pipeline_config.yaml`:
```yaml
camera:
  source: "webcam"        # dari "synthetic"
depth:
  source: "realsense"     # dari "mock" — butuh pip install pyrealsense2 + D435i/D415 terpasang
```

## Perluasan domain: deteksi buku & rak buku (mengikuti visi NSTC-Mackay)

Ditambahkan setelah meninjau dek **"Overall Progress NSTC – Mackay"** (Hucenrotia Lab/NYCU):
foto rak farmasi nyata mereka berisi kemasan berdiri tegak, berjajar rapat, punggung menghadap
kamera — visualnya sama persis dengan buku di rak buku, dan mereka sendiri menandai kondisi ini
sebagai kelemahan model mereka saat ini ("Non-Box Workpieces", "Uncertain Positioning" pada
slide "Current Progress: Vision"). Perluasan ini membuktikan arsitektur proyek ini (YOLO26n +
ONNX + tier-scan) **tidak spesifik-obat** — bisa dipakai untuk objek berjajar tegak apa pun,
termasuk buku, **tanpa mengubah satu baris pun kode inti di `src/`**. Yang ditambahkan hanya:

| File baru | Fungsi |
|---|---|
| `config/book_pipeline_config.yaml` | Profil config domain buku (model, kamera, tier rak) |
| `notebooks/train_book_on_kaggle.ipynb` | Notebook training 3 kelas (`book`/`shelf`/`pen`) dari 9 dataset nyata Anda |
| `docs/DATASET_SOURCES_BUKU.md` | Rincian 10 dataset nyata yang Anda upload, pemetaan kelas, statistik terverifikasi |

Panduan langkah-demi-langkah lengkap untuk upload dataset ke Kaggle, training, sampai
kalibrasi rak buku fisik ada di `docs/JALUR_B_CHECKLIST_BUKU.md` (format sama dengan
`docs/JALUR_A_CHECKLIST.md` untuk domain obat).

**Dataset Level 1 sudah nyata, bukan lagi rekomendasi generik**: 10 file Roboflow yang Anda
upload sudah diperiksa satu per satu (`data.yaml`, provenance, dan distribusi pemakaian kelas
di label asli — bukan cuma nama file). 9 file digabung (1 dikecualikan karena isinya barcode,
disimpan terpisah untuk R&D verifikasi barcode nanti) menjadi dataset 3-kelas terverifikasi:

| Kelas | Objek | Sumber |
|---|---|---|
| `book` | 118.831 | 7 dataset buku (termasuk 1 proyek Roboflow milik Anda sendiri) |
| `shelf` | 1.091 | 2 dataset shelf bersih, 1 kelas |
| `pen` | 380 | 1 dataset book+pen |

Total **7.931 gambar, 120.302 objek, 0 baris label rusak** (diverifikasi lewat simulasi merge
lokal terhadap file zip asli, bukan estimasi). Rincian lengkap tiap file + lisensi (semua
CC BY 4.0) ada di `docs/DATASET_SOURCES_BUKU.md`.

`scripts/test_webcam.py` dan `scripts/run_shelf_search.py` mendapat tambahan flag `--config`
supaya bisa memilih profil obat vs buku tanpa duplikasi skrip:

```bash
# Level 0 -- jalan HARI INI, model COCO placeholder yang sudah ada SUDAH punya kelas "book"
# (lihat src/pipeline.py, COCO_CLASSES) -- tanpa dataset atau training apa pun:
python scripts/test_webcam.py --config config/book_pipeline_config.yaml

# Level 1 -- setelah training khusus, 3 kelas book/shelf/pen (lihat notebooks/train_book_on_kaggle.ipynb):
python scripts/test_webcam.py --config config/book_pipeline_config.yaml \
    --onnx models/book_yolo26n.onnx --classes data/book_classes.txt
python scripts/run_shelf_search.py --config config/book_pipeline_config.yaml \
    --target <id> --live --image foto_rak_buku.jpg
```

**"Tingkat berapa" tetap dijawab oleh tier-scan geometris, bukan kelas `shelf` yang baru** —
kelas `shelf` hasil training di atas menjawab "apakah ada struktur rak di sini", bukan "ini
tingkat ke berapa". Pertanyaan tingkat tetap dijawab `src/shelf_scanner.py` (geometri bingkai
kamera dibagi per tingkat, `shelf.tiers`/`tier_bounds`, sudah teruji di
`tests/test_shelf_scanner.py`) — **tidak ada kode di `src/` yang berubah**. Penjelasan lengkap
kenapa keduanya saling melengkapi (bukan salah satu menggantikan yang lain) ada di
`docs/DATASET_SOURCES_BUKU.md`. Laporan gabungan yang menjelaskan kedua subsistem (obat &
buku) secara naratif tersedia terpisah sebagai dokumen `.docx`.

## Yang BELUM ada di paket ini (di luar target minggu 3)

Sesuai roadmap (bagian 10 dokumen desain), **Fase 4 — integrasi ROS2 + MoveIt2 +
kontrol lengan robot sungguhan** belum dibangun di sini karena itu target
setelah Fase 1–3 tervalidasi dengan hardware/dataset asli Anda. Hardware-nya
sendiri **sudah dikonfirmasi** (lengan Igus ReBel cobot 6-DOF + AMR A100
uni-innovate, lihat bagian di atas), jadi yang tersisa bukan lagi menunggu
keputusan jenis hardware, melainkan: (1) pengukuran lapangan reach-vs-tinggi-rak,
dan (2) menulis kode ROS2/MoveIt2 itu sendiri. Modul `depth_estimator.py` sudah
menyiapkan `grasp_point_from_detection()` sebagai titik awal untuk itu, mengikuti
bagian 7–9 dokumen desain.

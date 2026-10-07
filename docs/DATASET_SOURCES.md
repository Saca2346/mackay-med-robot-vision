# Sumber Data untuk Training Model Deteksi Obat

Ini pendamping `notebooks/train_on_kaggle.ipynb` (bagian "Dataset"). Sesuai dokumen desain
bagian 4-5: dataset publik bagus untuk mulai cepat dan melatih kemampuan umum model
mengenali "ini kemasan obat" / "ini botol pil", tapi **jarang punya SKU/merk yang persis
sama** dengan yang ada di rak Anda — jadi rekomendasinya adalah kombinasi: dataset publik
+ foto SKU spesifik Anda sendiri.

## 0. Contoh nyata sudah diverifikasi: Roboflow "Pharmacy" (dua versi)

Dua dataset ini sudah diunduh, dibuka, dan diperiksa langsung (bukan cuma dari deskripsi
halaman Roboflow) — termasuk ditemukan satu jebakan format yang penting untuk diketahui.

**a) `Pharmacy` v7 oleh sito-s02gz — 44 kelas produk NYATA, ekspor "YOLOv8"**
<br>Link: https://universe.roboflow.com/sito-s02gz/pharmacy-bk2qv/dataset/7
- 1454 foto (1045 train / 274 valid / 135 test), rak/etalase apotek sungguhan.
- 44 kelas nama produk asli (`zedd_easy`, `zedd_mini`, `sendo_primo`, `stamatin_kids_apple`, dst)
  — persis tipe data yang dibutuhkan sistem ini (bukan kelas generik "pill"/"box").
- **Jebakan format ditemukan**: walau diekspor sebagai "YOLOv8", **85% label (6039 dari 6903
  objek) ternyata berupa POLIGON**, bukan bounding box biasa (project Roboflow-nya dianotasi
  sebagai instance segmentation, bukan kotak lurus). Label deteksi YOLO txt (`class cx cy w h`)
  sebenarnya SAMA PERSIS di YOLOv5/v8/v9/v11 — jadi ekspor "YOLOv8" itu sendiri tetap valid
  untuk training YOLO11n — TAPI baris label berbentuk poligon (>5 angka per baris) akan
  membingungkan training deteksi kalau dipakai apa adanya.
- **Sudah diperbaiki**: `scripts/convert_roboflow_dataset.py` (baru) mengonversi tiap poligon
  ke bounding rectangle-nya (min/max titik x,y), sisanya disalin apa adanya. Diuji langsung
  pada dataset ini: 6903 objek dikonversi, 0 baris rusak, semua koordinat valid, dan diverifikasi
  visual — kotak hasil konversi menempel rapi pada kemasan produk di foto rak sungguhan.
  ```bash
  python scripts/convert_roboflow_dataset.py --input path/hasil_unzip_roboflow --output data/pharmacy_converted
  python scripts/train.py --data data/pharmacy_converted/data.yaml --epochs 120
  ```

**b) `Pharmacy` v3 oleh object-detection-hhx5h — 10 kelas, ekspor "YOLO11"**
<br>Link: https://universe.roboflow.com/object-detection-hhx5h/pharmacy-jl5t1/dataset/3
- 1591 foto (1080 train / 301 valid / 157 test), satu kemasan obat per foto, kemasan
  berbahasa Korea, format label bounding box BERSIH (bukan poligon, tidak perlu konversi).
- **Keterbatasan**: nama kelasnya cuma angka `'1'`..`'10'` (tanpa nama produk asli), jadi
  berguna sebagai data tambahan generik ("ini bentuk kemasan obat") tapi TIDAK bisa langsung
  dipetakan ke SKU tertentu di `medicine_db.sqlite` tanpa Anda cek manual produk apa di balik
  tiap nomor kelasnya di halaman Roboflow-nya.

**Kesimpulan praktis**: dataset (a) lebih berharga untuk sistem ini (nama SKU asli), tapi WAJIB
lewat `convert_roboflow_dataset.py` dulu. Dataset (b) plug-and-play tapi kurang informatif
untuk pencarian per-SKU. Keduanya bisa digabung dengan foto SKU Anda sendiri (bagian 3 di bawah)
untuk hasil yang paling sesuai kebutuhan Anda.

**Catatan umum ekspor Roboflow**: apa pun label format-nya di UI ("YOLOv8", "YOLO11", dst),
selalu cek dulu isi satu-dua file `.txt` di `train/labels/` sebelum training — kalau baris
labelnya lebih dari 5 angka, itu poligon dan perlu `convert_roboflow_dataset.py` lebih dulu.

## 1. Dataset publik (Kaggle)

| Dataset (slug Kaggle) | Isi | Cocok untuk |
|---|---|---|
| `trainingdatapro/pills-detection-dataset` | Foto pil dengan bounding box | Deteksi bentuk tablet/kapsul generik |
| `vencerlanz09/1k-pharmaceutical-pill-image-dataset` | ~1000 foto pil per kelas, berbagai obat | Klasifikasi bentuk/warna pil |
| `gunavenkatdoddi/medicine-tablet-pack-image-dataset` | Foto kemasan tablet/dus | Deteksi kemasan dus (bentuk kotak) |
| `trainingdatapro/mobile-captured-pharmaceutical-medication-packages` | Foto kemasan diambil dari kamera HP (bukan studio) | **Paling relevan** — kondisi pencahayaan/sudut mirip kamera webcam/lengan robot Anda |
| cari juga: "the drug name detection dataset" | Foto kemasan + label nama obat | Membantu jika mau gabungkan OCR-training |

Cara pakai di notebook: `+ Add Data` di UI Kaggle, atau `kagglehub.dataset_download("slug")`
(lihat sel di `train_on_kaggle.ipynb`).

**Catatan lisensi**: cek lisensi tiap dataset di halaman Kaggle-nya sebelum dipakai untuk
sistem yang akan dipakai di lingkungan kerja/komersial (rumah sakit, apotek, dll) — beberapa
dataset riset punya batasan non-komersial.

## 2. Dataset publik (Roboflow Universe)

Roboflow Universe (https://universe.roboflow.com) punya ratusan dataset deteksi objek
siap-pakai yang sudah dalam format YOLO, termasuk:
- `medicine-box-a33sn` — deteksi kotak/dus obat
- `medicine-box-jdjcw-m35zq` — varian lain deteksi kemasan dus obat
- cari juga dengan kata kunci "pill bottle detection", "pharmaceutical packaging",
  "tablet blister pack" — banyak project komunitas dengan ukuran dataset bervariasi (ratusan
  hingga puluhan ribu gambar)

Roboflow juga menyediakan tool **anotasi + augmentasi + export langsung ke format YOLO11**
secara gratis (tier gratis: sampai batas tertentu gambar/bulan) — sangat berguna untuk
Opsi B di bawah.

## 3. Foto SKU spesifik Anda sendiri (RECOMMENDED untuk produksi)

Karena target akhir adalah mendeteksi merk/SKU obat yang PERSIS ada di rak Anda (contoh:
Paracetamol merk tertentu, kemasan tertentu), dataset publik di atas paling berguna sebagai
**pre-training tambahan / augmentasi**, bukan pengganti data asli SKU Anda. Langkah praktis:

1. **Ambil foto** dengan `scripts/test_webcam.py` di laptop Anda — tekan tombol `s` saat
   preview berjalan untuk menyimpan frame ke `data/test_assets/webcam_capture_NNN.png`.
   - Foto tiap SKU dari **berbagai sudut** (depan, miring, dari atas — mensimulasikan sudut
     kamera lengan robot saat mendekat)
   - Berbagai **jarak** (dekat & jauh, sesuai skenario 2-tahap di dokumen bagian 7: kamera
     lebar dari jauh untuk cari tingkat rak, kamera wrist dari dekat untuk konfirmasi)
   - Berbagai **pencahayaan** (lampu ruangan, dekat jendela, dll)
   - Target realistis untuk hasil awal yang layak: **100-300 foto per SKU**. Lebih banyak
     lebih baik terutama untuk SKU dengan kemasan mirip satu sama lain.
2. **Label bounding box** — dua opsi:
   - **Roboflow** (https://roboflow.com, gratis untuk skala kecil): upload foto, gambar
     bounding box di browser, lalu export langsung ke format "YOLO11" — paling cepat, ada
     auto-augmentation (rotate/brightness/blur) yang membantu simulasikan kondisi "jauh/blur"
     seperti pada `four_tier_shelf_far_blurry.png` di proyek ini.
   - **LabelImg** (https://github.com/heartexlabs/labelImg) atau **CVAT**
     (https://github.com/opencv/cvat) — open-source, jalan lokal, cocok jika data tidak
     boleh diunggah ke layanan pihak ketiga (pertimbangan privasi data kesehatan/apotek).
3. **Gabungkan** dengan dataset publik di atas jika ingin model juga tetap bisa mengenali
   bentuk umum (misal generalisasi ke SKU baru yang belum difoto) — campur folder
   `images/train` dan sesuaikan `data.yaml` (lihat notebook, sel 4).
4. Upload dataset gabungan sebagai **Kaggle Dataset privat** (`+ Add Data -> Upload`) supaya
   bisa dipakai langsung di `train_on_kaggle.ipynb`.

## 4. Menghubungkan SKU dataset ke `medicine_db.sqlite`

Urutan `names` di `data.yaml` (dan `classes.txt` hasil export) HARUS sinkron dengan SKU di
database verifikasi (`scripts/init_db.py`, dipakai `src/verifier.py` untuk OCR/warna/bentuk).
Setelah model dilatih dengan SKU baru, perbarui `scripts/init_db.py` (atau isi
`medicine_db.sqlite` langsung) supaya setiap SKU yang bisa dideteksi model juga punya entri
referensi nama/warna/bentuk untuk tahap verifikasi hybrid (dokumen desain bagian 5).

## Ringkasan alur end-to-end

```
foto webcam Anda (scripts/test_webcam.py, tombol 's')
        + dataset publik (opsional, Kaggle/Roboflow)
                |
                v
        label bbox (Roboflow / LabelImg / CVAT)
                |
                v
        upload sbg Kaggle Dataset privat
                |
                v
        notebooks/train_on_kaggle.ipynb  (GPU gratis Kaggle T4/P100)
                |
                v
        best.onnx + classes.txt  (download dari panel Output Kaggle)
                |
                v
        copy ke models/ + data/classes.txt di laptop
                |
                v
        scripts/test_webcam.py --onnx ... --classes ...   (verifikasi cepat)
        scripts/run_shelf_search.py --live --image ...     (uji pencarian tingkat rak)
```

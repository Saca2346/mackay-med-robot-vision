# Jalur B: Checklist Langkah-demi-Langkah — Dataset & Training Deteksi Buku, Rak & Pena

Pendamping `docs/DATASET_SOURCES_BUKU.md` (kenapa dataset ini dipilih/dipetakan begini) — di
sini fokusnya urutan konkret, satu per satu, dengan cara cek "apakah langkah ini berhasil?" di
tiap tahap. Formatnya sengaja disamakan dengan `docs/JALUR_A_CHECKLIST.md` (domain obat).

**Update penting:** versi checklist ini sebelumnya meminta Anda mengunduh 2 dataset publik
generik secara manual dari Roboflow Universe (kendala `403 Forbidden` untuk unduhan tanpa
akun, sama seperti kasus Pharmacy dulu). **Itu sudah tidak berlaku** — Anda sudah mengunggah
**10 file dataset nyata** langsung ke percakapan ini, jadi langkah unduh dihapus. Checklist di
bawah langsung mulai dari file yang sudah Anda punya.

## Langkah 0 — File yang sudah Anda punya (tidak perlu diunduh lagi)

10 file berikut sudah diperiksa (lihat `docs/DATASET_SOURCES_BUKU.md` untuk detail tiap
file). **Unggah 9 file ini** ke Kaggle (file ke-10 sengaja dikecualikan):

| # | Nama file | Untuk diunggah? |
|---|---|---|
| 1 | `book.v1i.yolo26(1).zip` | ✅ Ya |
| 2 | `book.v1i.yolo26.zip` | ✅ Ya (proyek Roboflow Anda sendiri) |
| 3 | `book.v2i.yolo26(1).zip` | ✅ Ya |
| 4 | `book.v2i.yolo26.zip` | ✅ Ya |
| 5 | `Book spline detection.v1i.yolov8.zip` | ✅ Ya |
| 6 | `Book_Pen_Model.v1i.yolo26.zip` | ✅ Ya |
| 7 | `book.v2-roboflow-instant-1--eval-.yolo26.zip` | ✅ Ya |
| 8 | `shelf.v3i.yolo26.zip` | ✅ Ya |
| 9 | `shelf.v2-shelfv5.yolo26.zip` | ✅ Ya |
| 10 | `BOOK.v1i.yolo26.zip` | ❌ **Jangan** — isinya barcode (`1dcode`), bukan buku; lihat `DATASET_SOURCES_BUKU.md` untuk alasannya. Simpan terpisah untuk R&D verifikasi barcode di masa depan. |

**Cek:** Anda punya 9 file zip di komputer (dari unduhan/ekspor yang sama seperti yang Anda
kirim ke percakapan ini) sebelum lanjut ke Langkah 1.

## Langkah 1 — Upload ke Kaggle sebagai Dataset

Dua opsi, pilih salah satu (identik dengan yang sudah berhasil untuk Pharmacy):

**Opsi A — lewat UI Kaggle langsung (lebih sederhana):**
1. Buka https://www.kaggle.com/datasets -> **New Dataset**.
2. Upload **kesembilan file** dari tabel Langkah 0 sekaligus sebagai **satu Kaggle Dataset**
   (boleh juga dipecah jadi beberapa Dataset terpisah kalau upload sekaligus lambat/gagal —
   sel dataset di `notebooks/train_book_on_kaggle.ipynb` otomatis memindai SEMUA dataset yang
   di-attach ke notebook, tidak peduli dipisah atau digabung).
3. Beri nama, mis. `book-shelf-pen-dataset`, set **Private** kalau tidak ingin publik.
4. **Create**.

**Opsi B — lewat Kaggle API (kalau Opsi A lambat/gagal karena ukuran total ~430 MB):**
Sama seperti proses `medicine-model-export` sebelumnya — lihat riwayat sesi Anda untuk
langkah `kaggle datasets create` kalau perlu mengulang metode ini.

**Cek:** dataset muncul di https://www.kaggle.com/YOUR_USERNAME/datasets dengan 9 file
tercantum, masing-masing ukurannya sesuai (tidak ada yang 0 byte / gagal upload).

## Langkah 2 — Attach dataset ke notebook & jalankan training

1. Buka `notebooks/train_book_on_kaggle.ipynb` di kaggle.com/code -> **New Notebook**
   (upload file ini, atau copy-paste isinya).
2. **Settings -> Accelerator -> GPU T4 x2** (atau P100).
3. **Settings -> Internet -> On** (dibutuhkan untuk `pip install ultralytics`).
4. Panel kanan -> **+ Add Input** -> cari Dataset yang Anda buat di Langkah 1 -> klik
   **+ / Add** -> tunggu muncul di panel Input.
5. **Run All**.

**Cek:** sel `!nvidia-smi` menunjukkan Tesla T4 (bukan "no devices found"); sel dataset
(Bagian 3 notebook) mencetak `Ditemukan 9 dataset ter-attach` (atau jumlah sesuai cara Anda
membagi upload di Langkah 1) dan daftar `[OK] ...: digabung (...)` untuk tiap dataset —
**bukan** pesan `[LEWATI]` untuk 9 dataset yang seharusnya ikut (pesan `[LEWATI]` untuk
dataset ke-10/barcode yang sengaja tidak diunggah itu wajar, tapi seharusnya tidak muncul
sama sekali kalau Anda memang tidak mengunggahnya); totalnya harus mendekati **7.931 gambar,
120.302 objek** (book≈118.831, shelf≈1.091, pen≈380) — kalau jauh berbeda, kemungkinan ada
file yang belum ter-attach.

## Langkah 3 — Perhatikan hasil training per kelas (bukan cuma rata-rata)

Kelas `book` jauh lebih banyak datanya daripada `shelf` dan `pen` (lihat "Ketimpangan kelas"
di `docs/DATASET_SOURCES_BUKU.md`). Setelah sel training & validasi selesai:

**Cek:** mAP50 kelas `book` diharapkan tinggi (data melimpah). mAP50 kelas `shelf` dan `pen`
kemungkinan lebih rendah — ini **normal untuk kelas minoritas**, bukan tanda training gagal.
Kalau `book` juga rendah, baru itu tanda ada masalah (cek ulang Langkah 2).

## Langkah 4 — Download & pasang model hasil training

Download `book_model_export.zip` dari panel Output Kaggle, lalu di laptop Anda:

```bash
unzip book_model_export.zip -d ~/Downloads/book_model_export
cp ~/Downloads/book_model_export/best.onnx  medicine_shelf_vision/models/book_yolo26n.onnx
cp ~/Downloads/book_model_export/classes.txt medicine_shelf_vision/data/book_classes.txt
```

**Ingat kalau Anda sudah pernah kena masalah zip lama/duplikat di `~/Downloads`** (seperti
yang terjadi pada `medicine_shelf_vision.zip` sebelumnya) — cek dulu dengan
`ls -la ~/Downloads/book_model_export.zip*` kalau menemukan lebih dari satu file dengan
nama serupa, pastikan yang diekstrak adalah unduhan Kaggle yang baru. `classes.txt` yang
benar harus berisi tepat 3 baris: `book`, `shelf`, `pen` (dalam urutan ini).

## Langkah 5 — Uji dengan model & data buku Anda sendiri

```bash
cd medicine_shelf_vision
python scripts/test_webcam.py --config config/book_pipeline_config.yaml \
    --onnx models/book_yolo26n.onnx --classes data/book_classes.txt
```

**Cek:** kotak deteksi sekarang berlabel `book`, `shelf`, atau `pen` dengan confidence yang
tampak lebih stabil/tinggi dibanding Level 0 (model COCO umum, yang tidak punya kelas `shelf`
atau `pen` sama sekali) — khususnya untuk `book` pada buku yang berjajar rapat.

## Langkah 6 — Kalibrasi rak buku fisik & uji pencarian tingkat

Ini **tetap memakai logika tier-scan geometris yang sudah ada** (`src/shelf_scanner.py`) —
kelas `shelf` hasil training di atas TIDAK menggantikan langkah ini (lihat penjelasan di
`docs/DATASET_SOURCES_BUKU.md` bagian "Kenapa rak buku sekarang bisa dilatih... tapi tidak
menggantikan tier-scan").

1. Ukur rak buku fisik Anda (kalau tersedia): dari foto kamera lebar, catat rasio
   tinggi bingkai untuk tiap tingkat.
2. Isi `config/book_pipeline_config.yaml` -> `shelf.tier_bounds` dengan rasio tersebut
   (format contoh ada sebagai komentar di `config/pipeline_config.yaml`, identik untuk
   kedua domain).
3. Foto rak buku sungguhan, lalu:
   ```bash
   python scripts/run_shelf_search.py --config config/book_pipeline_config.yaml \
       --target <id_buku> --live --image foto_rak_buku.jpg
   ```

**Cek:** output menunjukkan tingkat yang benar sesuai posisi fisik buku target di foto
Anda. (Match `sku`/`id_buku` akan selalu "tidak dikenal" kecuali Anda mengisi
`book_db.sqlite` — lihat catatan opsional di `docs/DATASET_SOURCES_BUKU.md` bagian
terakhir; untuk sekadar deteksi + hitung + lokasi tingkat, ini tidak wajib.)

---

Kalau Langkah 0-2 (upload & training) terlalu memakan waktu untuk sekarang, ingat Level 0
tetap berjalan tanpa semua langkah di atas: `python scripts/test_webcam.py --config
config/book_pipeline_config.yaml` sudah mendeteksi buku hari ini juga, memakai kelas "book"
bawaan COCO — lihat `docs/DATASET_SOURCES_BUKU.md` bagian "Level 0". Jalur B di atas hanya
diperlukan kalau akurasi Level 0 terasa kurang, atau Anda butuh kelas `shelf`/`pen` yang tidak
ada di COCO.

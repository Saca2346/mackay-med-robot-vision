# Jalur A: Checklist Langkah-demi-Langkah (di Laptop Anda Sendiri)

Semua langkah ini dijalankan di **Lenovo V14 G2 ITL Anda sendiri** — sandbox pengembangan
tempat kode ini ditulis tidak punya kamera, GPU NVIDIA, atau layar, jadi langkah-langkah
ini tidak bisa dijalankan dari sana. Ini pendamping praktis untuk `README.md` (arsitektur)
dan `docs/DATASET_SOURCES.md` (dari mana dapat foto obat) — di sini fokusnya urutan
konkret, satu per satu, dengan cara cek "apakah langkah ini berhasil?" di tiap tahap.

## Langkah 0 — Pindahkan proyek & install dependensi

```bash
# ekstrak zip proyek ke laptop Anda, lalu:
cd medicine_shelf_vision
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
sudo apt install tesseract-ocr tesseract-ocr-eng   # Ubuntu; macOS: brew install tesseract
```

**Cek:** `pip list | grep opencv` harus menunjukkan `opencv-python` (BUKAN
`opencv-python-headless` — kalau keduanya sekaligus terpasang, `pip uninstall
opencv-python-headless` supaya tidak konflik).

## Langkah 1 — Preflight check (baru, cek semua sebelum pegang kamera)

```bash
python scripts/check_setup.py
```

Ini memeriksa: versi Python, semua paket Python, apakah OpenCV punya GUI backend
(dibutuhkan untuk jendela live preview), binary `tesseract`, provider ONNX Runtime
(CPU vs CUDA/MX350), apakah kamera terdeteksi di sistem, apakah model `.onnx` ada.

**Cek:** targetkan **0 [GAGAL]**. [PERINGATAN] soal CUDA tidak apa-apa untuk mulai (bisa
jalan di CPU dulu, lebih lambat tapi tetap bisa dipakai) — tapi tangani sebelum training
skala besar/produksi kalau ingin FPS lebih tinggi.

## Langkah 2 — Uji kamera hidup

```bash
python scripts/test_webcam.py
```

Jendela preview harus muncul menampilkan feed kamera + FPS + jumlah deteksi (model
COCO bawaan — wajar kalau tidak mengenali kemasan obat sebagai kelas spesifik, lihat
catatan jujur di README).

**Cek:** jendela muncul, FPS di atas 0, tidak crash selama beberapa menit. Tekan `q`
untuk keluar.

Kalau kamera index 0 tidak terbuka: coba `--camera 1`, `--camera 2`, dst.

## Langkah 3 — Kumpulkan foto SKU obat Anda

```bash
python scripts/test_webcam.py --save-dir data/test_assets
# tekan 's' berulang kali sambil menggeser/memutar tiap obat & jarak kamera
```

Target per SKU (lihat `docs/DATASET_SOURCES.md` untuk detail): **100-300 foto**,
variasi sudut/jarak/pencahayaan. Ulangi untuk tiap SKU yang ingin dikenali sistem.

**Cek:** `ls data/test_assets/webcam_capture_*.png | wc -l` — hitung berapa foto
sudah terkumpul per sesi.

## Langkah 4 — Label bounding box

Upload foto ke [Roboflow](https://roboflow.com) (gratis untuk skala kecil), gambar
kotak di sekeliling tiap obat, kasih label nama SKU, lalu **Export -> format "YOLO11"**.

**Cek:** hasil export punya struktur `images/train`, `images/val`, `data.yaml` dengan
`names:` berisi SKU Anda.

## Langkah 5 — Training di Kaggle (GPU gratis)

1. Upload dataset hasil Langkah 4 sebagai Kaggle Dataset privat (`+ Add Data -> Upload`).
2. Buka `notebooks/train_on_kaggle.ipynb` di kaggle.com/code -> New Notebook.
3. **Settings -> Accelerator -> GPU T4 x2**, **Settings -> Internet -> On**.
4. Sesuaikan path dataset & `names` di sel "data.yaml" dengan punya Anda.
5. **Run All**.

**Cek:** sel `!nvidia-smi` menunjukkan Tesla T4 (bukan "no devices found"); sel training
selesai tanpa error; sel validasi menunjukkan angka mAP50 (semakin tinggi semakin baik,
> 0.5 sudah layak untuk mulai iterasi).

## Langkah 6 — Download & pasang model hasil training

Download `medicine_model_export.zip` dari panel Output Kaggle, lalu:

```bash
unzip medicine_model_export.zip -d /tmp/medicine_model_export
cp /tmp/medicine_model_export/best.onnx  models/medicine_yolo11n.onnx
cp /tmp/medicine_model_export/classes.txt data/classes.txt
```

## Langkah 7 — Update database referensi

Edit `scripts/init_db.py`: ganti/tambahkan SKU sesuai `data/classes.txt` (nama, warna
dominan, bentuk kemasan) supaya tahap verifikasi hybrid (OCR+warna+bentuk) di
`src/verifier.py` punya data referensi yang sinkron dengan kelas model baru. Lalu:

```bash
python scripts/init_db.py
```

## Langkah 8 — Uji ulang dengan model & data Anda sendiri

```bash
python scripts/test_webcam.py --onnx models/medicine_yolo11n.onnx --classes data/classes.txt
```

**Cek:** kotak deteksi sekarang berlabel nama obat sungguhan Anda (bukan lagi kelas
COCO generik seperti "bottle"/"book").

## Langkah 9 — Kalibrasi rak fisik & uji pencarian tingkat

1. Ukur rak fisik Anda: dari foto kamera lebar, catat di piksel/rasio tinggi bingkai
   di mana tiap tingkat mulai & berakhir.
2. Isi `config/pipeline_config.yaml` -> `shelf.tier_bounds` dengan rasio tersebut
   (contoh format sudah ada sebagai komentar di file itu).
3. Foto rak sungguhan (semua 4 tingkat kelihatan), lalu:
   ```bash
   python scripts/run_shelf_search.py --target <SKU_ANDA> --live --image foto_rak.jpg
   ```

**Cek:** output menunjukkan tingkat yang benar sesuai posisi fisik obat target di
foto Anda.

---

Setelah Langkah 9 berhasil dengan foto/rak sungguhan, Jalur A selesai — sistem visi
sudah bisa: mendeteksi SKU obat asli Anda, memverifikasinya, dan menentukan tingkat
rak yang benar dari foto kamera nyata. Langkah selanjutnya (Fase 4: gerakkan lengan
ke tingkat itu + picking) menunggu kepastian jenis kontroler lengan robot Anda.

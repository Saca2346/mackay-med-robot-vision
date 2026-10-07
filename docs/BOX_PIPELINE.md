# Pipeline BOX + SIZE_LABEL + OCR — cara pakai

Tambahan ini tidak mengubah file lama (detector/verifier/test_webcam versi buku & obat tetap ada).
Semua file baru berawalan `box_`.

| File | Fungsi | Dijalankan di |
|---|---|---|
| `config/box_pipeline_config.yaml` | semua angka yang bisa diubah: model, threshold, OCR, katalog produk | - |
| `data/box_classes.txt` | urutan kelas: 0 = box, 1 = size_label | - |
| `src/box_geometry.py` | IoU poligon, crop miring jadi lurus, baca file label | - |
| `src/box_detector.py` | jalankan model (.onnx / .pt), pasangkan size_label ke box induknya | - |
| `src/box_ocr.py` | baca teks crop size_label (RapidOCR, cadangan Tesseract) | - |
| `src/box_verifier.py` | teks -> produk / diameter / panjang -> MATCH / IGNORED / CONFIRM | - |
| `scripts/check_box_labels.py` | cari label longgar (box saling tumpang tindih) | server / laptop |
| `scripts/export_box_onnx.py` | best.pt -> models/box_obb.onnx + uji kecepatan CPU | server |
| `scripts/test_webcam_box.py` | uji live webcam / foto: Tahap 1 (deteksi) dan Tahap 2 (--request) | laptop |
| `scripts/eval_box_manual.py` | tabel TP / FP / FN + precision / recall pada foto uji | laptop / server |
| `tests/test_box_*.py` | 24 unit test (geometri, parsing, keputusan, kasus mentor A/B/C) | - |

## 0. Pasang (sekali, di laptop dan di server)

```bash
cd medicine_shelf_vision
source venv/bin/activate            # venv yang sudah ada
pip install -r requirements_box.txt
python -m pytest tests/test_box_geometry.py tests/test_box_verifier.py tests/test_box_eval.py -q   # harus 24 passed
```

## 1. Server — cek label, latih ulang, ekspor

```bash
# 1a. cari foto dengan label longgar (kotak lurus pada kemasan miring)
python scripts/check_box_labels.py --labels <folder labels> --images <folder images>
#     -> results/label_check/label_check.csv + gambar foto yang perlu digambar ulang

# 1b. setelah label diperbaiki: latih ulang (pakai data.yaml yang sama dengan run bootstrap;
#     lihat baris "data:" di runs/bootstrap_v.../args.yaml)
yolo obb train data=<data.yaml> model=yolo26n-obb.pt imgsz=640 epochs=150 batch=16 device=0 project=runs name=obb_v2

# 1c. ekspor ke ONNX + uji kecepatan di CPU
python scripts/export_box_onnx.py --weights runs/obb_v2/weights/best.pt --image <satu foto uji>
```

Catatan: model bootstrap yang sudah ada juga bisa langsung diekspor (langkah 1c saja) supaya uji laptop bisa mulai hari ini.

## 2. Pindahkan model ke laptop

```bash
# dari terminal laptop (server harus menyalakan SSH: sudo systemctl enable --now ssh)
scp Zen@hucenrotia-ai:<path proyek di server>/models/box_obb.onnx models/
# kalau nama host tidak dikenali, ganti hucenrotia-ai dengan IP server (cek: hostname -I di server)
```
Cadangan kalau SSH tidak bisa: unggah `box_obb.onnx` ke Google Drive dari server, unduh di laptop.

## 3. Laptop — uji webcam

```bash
# Tahap 1: deteksi saja. Taruh 4 kotak seukuran + 7 item di depan webcam, hitung x/11 dan catat FPS.
python scripts/test_webcam_box.py

# Tahap 2: pilih kotak sesuai permintaan, boleh kalimat biasa
python scripts/test_webcam_box.py --request "pilih ivascular angiolite diameter 4 mm panjang 19"
python scripts/test_webcam_box.py --request "angiolite,2.5,29"

# dari foto (tanpa webcam), hasil disimpan ke results/screens/*_result.png
python scripts/test_webcam_box.py --source <folder foto> --request "accuforce,2.75,20" --no-window
```
Tombol: `q` keluar · `s` simpan frame asli + beranotasi (untuk slide) · `l` catat keputusan ke `results/selection_log.csv` · `o` paksa OCR ulang · `r` permintaan yang sama sekali lagi · `b` pindai detail satu kotak.

**Permintaan** dipahami dalam kalimat biasa: "pilih ivascular angiolite diameter 4 mm panjang 19", "terumo accuforce
2,75 x 20", "nagomi 3/24", "angiolite Ø2.5 L29", "boston scientific permanent sled bag". Merek boleh ditulis (harus
cocok dengan produknya); angka tanpa kata kunci = diameter lalu panjang. Program mencetak cara ia membaca permintaan
("permintaan dibaca sebagai: angiolite (ivascular), diameter 4 mm, panjang 19 mm"); permintaan yang tidak lengkap /
tidak masuk akal ditolak dengan penjelasan. Saat program berjalan, permintaan BARU bisa diketik di terminal + Enter:
seluruh rak dibaca ulang untuk permintaan itu.

**Hanya kotak DI RAK yang bisa AMBIL**: kotak harus diam terhadap kotak-kotak lain (bukan dipegang / ditarik) dan
tidak menutupi kotak lain (bukan di depan rak). Di video uji 2026-09-28 13:06 kotak yang sudah diambil lalu dipegang
di depan rak sempat dipilih lagi dan slotnya menempel ke kotak lain; sekarang kotak itu ditolak ("tidak di rak").
Diukur pada rekaman w300: dengan kamera diam dan tanpa tangan, 100 % kotak rak lolos aturan ini.

**Pindai detail (tombol `b`)**: dekatkan SATU kotak ke kamera (sisi mana saja, juga belakang), tahan diam, tekan `b`.
Semua kode dibaca - barcode 1D GS1-128 (Terumo), DataMatrix / QR (Boston Scientific, juga tanpa tanda kurung dan QR
"GS1 Digital Link"), QR biasa (alamat web) - plus seluruh teks (OCR beberapa skala). Kartu data: GTIN + isi katalog
(terverifikasi / DRAF), kedaluwarsa, LOT, nomor seri, REF -> ukuran, produk & ukuran dari teks, dan keputusan terhadap
permintaan. Angka ukuran yang terbaca berbeda antar lintasan OCR ditandai ragu (tidak dipakai). Disimpan ke
`results/screens/pindai/*.jpg + .json`.

**Barcode baru dipakai untuk MATCH kalau GTIN-nya ada di katalog DAN diverifikasi orang** (`data/gtin_catalog.csv`).
Terumo (teks ukuran kecil) dan Boston Scientific (tanpa kode REF berukuran) sangat bergantung pada ini:
`python scripts/enroll_gtin.py --review` untuk memeriksa draf (nama pemeriksa ditanyakan di terminal), `--camera auto` untuk mendaftarkan kotak
baru (tekan `e` saat barcode terlihat), `--source <folder foto>` untuk foto barcode dari ponsel (paling tajam),
`--belum` = daftar produk/ukuran yang belum punya GTIN, `--review --semua` = tinjau ulang juga entri terverifikasi
(pilihan `e` mengubah produk, ukuran, merek dan REF / tanda kemasan), `--ganti-pemeriksa "NAMA LAMA"` (nama baru ditanyakan). Contoh yang tersalin apa adanya ("nama anda", "NamaAsliAnda",
`<...>`) ditolak.
REF di katalog hanya keterangan (tidak dipakai untuk keputusan). Boston Scientific: REF diisi "MR" (tulisan MONORAIL
di ujung punggung, sama untuk semua ukuran) - ukurannya dipastikan lewat GTIN di DataMatrix/QR + teks ukuran.
Katalog diubah HANYA di laptop, lalu dikirim ke server sendiri setiap kali berubah
(`scp data\gtin_catalog.csv Zen@<server>:~/Downloads/box_pipeline/data/`). Paket zip server TIDAK berisi katalog,
jadi unzip paket tidak pernah mengembalikan katalog lama.

Warna kotak Tahap 2:

| warna | arti | diambil? |
|---|---|---|
| hijau tebal + `AMBIL` | kotak yang diminta, terverifikasi (>= 2 bukti independen, 2 bacaan sepakat) | ya, hanya kotak ini |
| hijau tipis `MATCH` | juga cocok, tapi ada kotak cocok lain yang kedaluwarsa lebih dulu (FEFO) | nanti |
| biru muda `KANDIDAT` | baru 1 bukti cocok (mis. teks ukuran saja) | tidak, dekatkan / tunjukkan barcode |
| kuning `CONFIRM` | ragu: bukti bertentangan, tidak terbaca, terlalu jauh | tidak, sistem tidak menebak |
| abu-abu | pasti bukan target (produk ATAU ukuran terbaca berbeda) | tidak (diabaikan) |
| putih `membaca...` / `cocok? 1/2` / `cek ulang` | belum cukup bacaan / nomor kotak sempat diragukan | belum |
| ungu `KEDALUWARSA` | tanggal kedaluwarsa (barcode) sudah lewat | tidak pernah |

### Alur penyeleksi (`src/box_session.py`)

Banner baris ke-2 HUD menunjukkan langkahnya (hijau = ketemu / selesai, merah = tidak ada / rak kosong,
kuning = perlu konfirmasi, putih = sedang bekerja):

1. **CEK RAK** — arahkan kamera ke SELURUH rak, tahan diam. 0 kotak selama 2 detik -> **RAK KOSONG** (merah),
   OCR tidak dijalankan. Kotak terlihat stabil 1,5 detik -> jumlah kotak di rak dicatat (`JUMLAH KOTAK` di log).
2. **MENCARI** — tiap kotak dibaca. Kotak yang diminta terverifikasi -> **DITEMUKAN** (hijau, `AMBIL #n`, 19 kotak
   lain diabaikan). **TIDAK ADA** (merah) hanya kalau SEMUA kotak yang terlihat SEKARANG pasti bukan target DAN
   jumlahnya >= jumlah kotak di rak, jadi mundurkan kamera sampai seluruh rak terlihat. Satu kotak saja yang ragu /
   belum terbaca / belum terlihat -> **PERLU KONFIRMASI** (kuning) dengan nomor kotaknya; sistem tidak pernah
   menebak "tidak ada".
3. **AMBIL -> sedang diambil -> TERAMBIL** — slot kotak AMBIL dikunci ke posisinya di RAK (ikut gerak kamera dan
   kotak-kotak lain, tidak ikut kotak yang ditarik). Kotak yang bergerak keluar dari slot = "sedang diambil".
   Setelah keluar, sistem memastikan slotnya kosong 2 detik (slot terlihat utuh, kamera diam, tidak ada kotak di
   slot, gambar slot berbeda dari kotak itu dan diam, jadi tangan yang lewat atau detektor yang sesaat gagal tidak
   dihitung). Nomor kotak itu lalu dibuang.
   Selama DITEMUKAN kotak AMBIL dibaca ulang paling dulu; bacaan sebagian / kosong (tangan menutupi, buram) tidak
   membatalkan kunci, satu bantahan -> kuning (JANGAN diambil) dan dibaca lagi, dua bantahan -> BATAL AMBIL.
   **CEK ULANG** (kuning, "AMBIL? cek ulang"): nomor kotak AMBIL diragukan (tangan meraih / menutupi kotak, kotak
   tersenggol), satu kotak diam bernomor lain menempati slotnya (nomornya berganti saat tertutup), atau tampilan
   kotak di slot berubah. Kunci TETAP dipegang tapi robot tidak boleh mengambil; hanya bacaan BARU yang dipakai:
   bacaan yang menunjuk tepat produk + ukuran yang diminta (teks lengkap, REF, barcode atau GTIN) tanpa bantahan
   -> **AMBIL DIPASTIKAN** (hijau lagi). "angiolite" saja tidak cukup; kotak saudara (angiolite 4x19) membantah
   lewat ukurannya -> BATAL. Tidak terpastikan 10 detik selama kotak ada di slotnya (`relock_s`) -> BATAL AMBIL.
   Kotak yang ditarik keluar selama CEK ULANG tetap diawasi sampai TERAMBIL (dulu: BATAL tepat saat perawat
   meraih kotak, lalu TERAMBIL tidak tercatat). Kotak sebelah yang miring / bergeser mengisi celah kotak yang
   diambil (terbaca 2x bukan target, kotak AMBIL tidak terlihat lagi, dan memang ada gerakan: kotak AMBIL ditarik
   atau kotak sebelah itu bergeser >= 0,5 x lebar) -> TERAMBIL; tanpa gerakan -> BATAL AMBIL biasa.
   Nomor kotak (tracker): kotak yang pusatnya tetap di tempat (geser ke samping <= 0,3 x lebar, memanjang
   <= 0,15 x panjang, tidak ambigu) tetap nomor lama walau sudut / panjang deteksinya berubah. Punggung kotak
   yang miring 6-8 derajat saja sudah membuat IoU < 0,5, dan dulu setiap kali itu terjadi bacaannya dibuang
   (uji tripod 30-09: AMBIL 39,5 s). Aturan ini hanya dipakai kalau kamera tenang: gerak terukur < 0,1 x lebar
   kotak selama 5 frame berturut-turut, dan pasangan yang pasti tidak bergeser bersama. Versi pertama (syarat
   gerak < 0,5 x lebar) membuat 3 nomor kotak pindah ke kotak sebelah di replay kamera yang dipegang tangan. Saat
   kamera bergerak, aturan IoU lama tetap berlaku.
   **Kotak terbalik (180 derajat):** classifier arah RapidOCR membalik baris panjang (REF, tanggal) tapi tidak angka
   pendek, jadi "2,5" dan "29" pada kotak terbalik terbaca "25 29" / "62 55". Kalau bacaan belum lengkap, label
   ukuran dibaca juga dalam arah terbalik; kalau arah itu membaca teks jauh lebih jelas, angka arah biasa dibuang.
   Saat mencari, hanya untuk kotak yang produk / REF-nya cocok dengan permintaan atau yang baru terbaca angkanya
   saja; saat memindai rak, semua kotak. Replay tripod 30-09 (Angiolite 2.5x29 dikembalikan terbalik): tanpa ini
   tidak ada AMBIL dalam 91 s, dengan ini AMBIL 10,1 s dan tetap terkunci; OCR +22 % per kotak. eval_verify:
   0 MATCH salah dari 1353, MATCH benar 14 -> 16 dari 123.
4. **Permintaan yang sama lagi** (`r` atau `--qty 2`) — dicari kotak LAIN; slot yang sudah kosong tidak pernah
   dipilih. Tidak ada lagi -> **TIDAK ADA** "tidak ada lagi". Slot kosong yang terisi lagi dalam 5 detik -> peringatan.
5. **Pesanan** (`src/box_order.py`, di `--request` atau diketik di terminal saat berjalan):
   `angiolite 2.5 29` satu kotak; `2 kotak permanent sled bag` (atau `jumlah 2`) jumlah;
   `angiolite 2.5 29; accuforce 2.75 20 jumlah 2` beberapa barang (pemisah `;` `+` `dan` `serta` `lalu`);
   `+ essential pro 3 40` tambah ke antrean; `lewati` barang berikutnya. Baris tanpa `+` MENGGANTI pesanan.
   Barang berikutnya dimulai otomatis setelah SELESAI, atau TIDAK ADA bertahan 3 s; PERLU KONFIRMASI menunggu
   `lewati`. Saat permintaan berganti, bacaan rak TIDAK dibuang: bukti tiap kotak diputuskan ulang untuk permintaan
   baru (`TrackState.redecide`), dan bacaan OCR yang selesai sesudah pergantian dinilai terhadap permintaan baru.
   Satu barang yang tidak jelas membatalkan seluruh baris (tidak ada barang yang diam-diam hilang).
6. **Area rak** (kamera tripod): tombol `a` di jendela video -> tarik kotak di sekitar rak -> Enter. Disimpan ke
   `results/area_rak.json` dan dipakai lagi otomatis untuk kamera live (bukan untuk rekaman lama); `--roi none`
   mematikan, `--roi x1,y1,x2,y2` (pecahan 0-1) mengisi langsung. Deteksi di luar area diabaikan: uji 30-09 kotak
   kertas dan kotak GPU di samping rak terhitung sebagai kotak (24, padahal 20); dengan area rak rekaman yang sama 19.
   Kamera dipindah -> pilih ulang area.
7. **Diameter bulat iVascular** ("4" lalu "19", satuan mm di baris lain): `Identity.diameter_hint`, hanya boleh
   MENYEPAKATI diameter yang diminta, tidak pernah membantah (potongan "3,5" yang terbaca "3" tidak membuat kotak 3.5
   dianggap bukan target). Teks saja tetap 1 bukti. eval_verify: 0 MATCH salah dari 1353.
8. **Salah baca yang disaring** (replay 30-09): angka yang menempel ke huruf adalah bagian kode, bukan ukuran
   (REF "SCCDSR1415…" terbaca "5CCD5A1415…" di sebelah "mm" dulu jadi diameter 5 dan membatalkan AMBIL);
   "4mm", "19rnm", "2.5x29" tetap terbaca. REF dengan diameter bukan kelipatan 0,25 mm ("259" = 2,59) dibuang.
   Salah baca angka panjang di REF (sekitar 3 % bacaan, mis. 4x19 terbaca 4x10) tetap dihitung bantahan: aman,
   hanya lebih lambat. `--debug-readings` mencatat teks OCR mentah (kolom `teks`: label || REF || punggung)
   untuk mencari asal salah baca berikutnya.

Merek dan nama sama, ukuran beda (mis. Angiolite 2.5x29 vs 2.5x24 vs 2.25x29) tidak pernah MATCH: produk DAN
diameter DAN panjang harus sama persis dari 2 bukti independen (teks ukuran + kode REF, atau barcode terverifikasi);
satu bukti dengan ukuran lain = abu-abu / kuning. Evaluasi 123 kotak: 92 permintaan "produk sama ukuran beda",
0 MATCH salah.

Jumlah kotak di rak diketahui dari inventori? Pakai `--expected 20`: kalau detektor hanya melihat 18, sistem tidak
akan menyimpulkan TIDAK ADA. Setiap peristiwa (JUMLAH KOTAK, AMBIL, TERAMBIL, TIDAK ADA, ...) dicatat ke
`results/verify_log.csv` dan gambarnya disimpan ke `results/screens/peristiwa/` (bukti audit).

Default laptop: model OpenVINO `models/box_v1b_1280_openvino_model` di iGPU Intel (`intel:gpu.0`), sekitar 10 FPS
pada frame 1280x720 (.pt di CPU: sekitar 2 FPS). Kamera dibaca di thread sendiri (hanya frame terbaru dipakai), dan
pada sumber live OCR berjalan di thread latar, jadi jendela tidak "Not Responding" selama OCR membaca.
Akhir setiap run mencetak **Ringkasan** (FPS nyata, ms deteksi, iterasi > 1 s, ms OCR per crop) untuk perbandingan.

## 3b. Jalankan di server GPU (dilihat lewat Remmina) - kalau di laptop patah-patah

Webcam tetap di laptop. Laptop hanya mengirim gambar (stream MJPEG 1080p lewat tunnel SSH, hanya terbuka di
127.0.0.1); server (RTX A5000) yang mendeteksi, melacak, membaca OCR (3 thread) dan menjalankan alur penyeleksi;
hasilnya dilihat di desktop Remmina. Port 8090 di server sudah dipakai layanan lain, jadi di server stream-nya ada
di port **18090**. Kamera hanya bisa dipakai satu program (tutup test_webcam_box.py di laptop).

Sekali saja (atau tiap ada paket baru):

```bash
# laptop, cmd (isi password server): kirim paket
scp C:\Users\sacab\Downloads\box_pipeline_server.zip Zen@140.113.149.94:~/Downloads/
# server, terminal di desktop Remmina
mkdir -p ~/Downloads/box_pipeline && cd ~/Downloads/box_pipeline && unzip -o ~/Downloads/box_pipeline_server.zip
source ~/trse_venv/bin/activate
pip install rapidocr onnxruntime zxing-cpp
python -m pytest tests -q
```

Tiap kali uji (3 jendela, biarkan semuanya terbuka):

```bash
# laptop, cmd 1: kirim webcam (tiap 5 detik mencetak FPS kamera dan Mbit/s)
venv\Scripts\python.exe scripts\stream_webcam.py
# laptop, cmd 2: tunnel (isi password server)
ssh -N -R 18090:127.0.0.1:8090 Zen@140.113.149.94
# server, terminal Remmina
cd ~/Downloads/box_pipeline && source ~/trse_venv/bin/activate
python scripts/test_webcam_box.py --config config/box_pipeline_config_server.yaml --source http://127.0.0.1:18090/video.mjpg --request "angiolite,2.5,29" --expected 20 --record results/uji_server.mp4
```

- Jendela tampilan di server diperkecil ke lebar 1280 (`output.view_width`) supaya RDP lancar; deteksi, OCR dan
  gambar bukti tetap resolusi penuh. Remmina: kualitas "Good" cukup.
- Baris atas HUD: `tampil X fps` (frame yang benar-benar diproses), `masuk Y fps` (frame yang sampai dari laptop),
  `proses maks Z fps` (kecepatan satu iterasi tanpa menunggu). "masuk" rendah = jaringan laptop -> server penuh.
- OCR server: 3 mesin paralel, tiap mesin dapat bagian core sendiri (`ocr.threads` 0 = otomatis). Sebelumnya tiap
  mesin memakai semua core sehingga saling berebut (terukur 0,7-1,9 detik per kotak). OCR di GPU (opsional):
  `pip uninstall -y onnxruntime && pip install onnxruntime-gpu`, lalu `ocr.device: "cuda"`; kalau GPU tidak bisa
  dipakai, otomatis kembali ke CPU.
- Stream default 1080p JPEG 80, maks 15 fps (sekitar 25 Mbit/s). Jaringan lambat (cmd 1 mencetak FPS < 15, atau
  gambar di server tertinggal): `--quality 70` atau `--fps 10`. Resolusi jangan diturunkan kalau bisa: teks ukuran
  butuh 1080p.
- Uji server tanpa webcam dengan rekaman yang sama: laptop cmd 1
  `venv\Scripts\python.exe scripts\stream_webcam.py --video results\uji_tahap2.mp4 --video-fps 10.7 --start 220`
  (rekaman baru dari `--record` punya `.times.csv`, jadi `--video-fps` tidak perlu).
- Sudah diuji di laptop dengan rekaman w300 lewat stream http + 3 thread OCR: hitung kotak, AMBIL pada kotak yang
  benar, rekaman `--record` di sisi penerima lengkap dengan waktu tiap frame.

## 3c. Webcam eksternal (HP w300, 1920x1080)

Config `camera.index: "auto"` memilih kamera dengan resolusi terbesar (w300 = 1920x1080 30 fps lewat MSMF + MJPG;
kamera laptop hanya 640x480), jadi tidak perlu `--camera`. Paksa manual: `--camera 0` (w300) / `--camera 1`.
Diukur di laptop (i3-1115G4, iGPU), frame 1080p asli: model v2m 960 = 7.1 FPS nyata, 1280 = 3.8 FPS.
OCR dan barcode selalu memakai crop dari frame 1080p penuh (bukan dari gambar kecil untuk detektor).
w300 fokus tetap: HUD menampilkan `tajam N` (bagian tengah frame). Geser jarak kotak ke kamera sampai angkanya
paling tinggi (di bawah ~100 = buram, teks ukuran tidak akan terbaca). `stream_webcam.py` juga memakai w300 1080p
(maks 15 fps, sekitar 35 Mbit/s; jaringan lambat: `--width 1280 --height 720`).

### Tahap 2 dengan kamera dipegang tangan

- Nomor kotak (`#12`) mengikuti gerakan kamera (optical flow, `src/box_motion.py`), jadi bacaan OCR yang sudah
  terkumpul untuk kotak itu tidak hilang saat kamera digeser. Kalau ragu (satu deteksi menutupi dua kotak, gerakan
  tidak terukur), kotak diberi nomor baru: bacaan hilang, tapi TIDAK PERNAH pindah ke kotak sebelah.
- Frame yang diambil saat kamera bergerak (> `ocr.max_motion_px` px per 1/30 detik, diukur dengan waktu
  sebenarnya) tidak di-OCR; HUD menulis `KAMERA BERGERAK - tahan diam`. Ringkasan akhir mencetak berapa persen frame
  yang bergerak. Gerakan yang tidak terukur (buram berat): kotak di frame itu tidak dicocokkan ke nomor lama; lebih
  dari `ocr.max_lost_frames` frame -> semua nomor dilupakan dan kotak dibaca ulang (aman, tidak tertukar).
- Kotak yang keluar layar tetap diingat (posisinya ikut digeser), jadi rak bisa diperiksa per bagian dari dekat.
  Masuk layar lagi: dicocokkan hanya kalau pas dan tidak ambigu; kotak yang tadinya cocok wajib dibaca ulang
  (`cek ulang`) sebelum boleh AMBIL.
- Nomor kotak yang diragukan: pada rekaman w300 terukur nomor kotak kadang pindah ke kotak SEBELAH setelah kamera
  bergerak jauh / menjauh. Setiap kali pasangan nomor-kotak lemah (IoU < 0,5, lewat aturan longgar, atau masuk layar
  lagi), SEMUA bacaan lama kotak itu dibuang (juga hasil OCR dari crop lama yang baru selesai) -> `cek ulang`, perlu
  2 bacaan baru. Pada rekaman w300 3 dari 3 perpindahan yang terjadi lewat pasangan lemah tertangkap aturan ini.
- Size_label hanya dipakai untuk box yang memuat pusatnya (config `grouping.min_label_overlap`); label di luar box
  tidak lagi dipasangkan ke box terdekat, karena di rak itu hampir selalu label kotak sebelah.
- OCR mendahulukan kotak yang belum pasti dan paling dekat ke TENGAH layar. Cara uji: arahkan kotak target ke
  tengah, tahan diam 2-3 detik per posisi (OCR laptop sekitar 1-3 detik per kotak).
- `#5 cocok? 1/2` = baru 1 dari 2 bacaan yang dibutuhkan (belum boleh dipakai); `...` = sedang dibaca.
- `--record results/uji.mp4` menyimpan SEMUA frame kamera asli (tanpa anotasi) di thread sendiri, plus waktu asli
  tiap frame di `results/uji.mp4.times.csv` (kamera sering memberi fps lebih rendah dari 30). Putar ulang dengan
  `--source results/uji.mp4`: frame dilompati seperti kamera live dengan waktu aslinya (`--all-frames` = semua
  frame). Rekaman lama tanpa `.times.csv`: beri fps sebenarnya, mis. `--video-fps 10.7`. Frame ini juga bahan latih
  jarak dekat w300 (jangan pakai rekaman layar: HUD ikut terekam).

## 3d. Inventaris rak (persiapan sebelum operasi) - scripts/rak_inventaris.py

Alur operasional: SEBELUM hari operasi rak dipetakan dan diperiksa perawat; SAAT operasi kotak yang diminta sudah
diketahui dan hanya kotak itu yang dipastikan ulang.

```bash
# (opsional) daftar isi rak yang sudah ada: CSV kolom produk,diameter,panjang,jumlah[,gtin,lot,kedaluwarsa]
python scripts/rak_inventaris.py impor daftar_isi_rak.csv
# 1. pindai: kamera DIAM (tripod) di depan rak, tunggu sampai kotak hijau, q = simpan
python scripts/rak_inventaris.py pindai
# 2. perawat memeriksa tiap kotak dengan kotak aslinya (y benar / e ubah / n hapus); nama pemeriksa dicatat
python scripts/rak_inventaris.py cek
python scripts/rak_inventaris.py lihat
# 3. operasi: permintaan dicari lewat inventaris
python scripts/test_webcam_box.py --inventaris data/rak/inventaris.csv --request "angiolite,2.5,29"
```

- Identitas kotak saat pindai: PASTI hanya kalau >= 2 bukti independen sepakat (barcode 2 / teks ukuran 1 / REF 1 /
  teks GTIN 1) dari >= 2 bacaan tanpa satu pun bacaan bertentangan; potongan teks dari beberapa bacaan boleh
  dirangkai. Satu bukti saja = USULAN (cek teliti). Semuanya berstatus draf sampai diperiksa perawat; saat operasi
  HANYA entri `terverifikasi` yang dipakai.
- Saat operasi: kotak inventaris yang cocok diurutkan FEFO; tiap kotak di layar dibandingkan dengan ciri tampilan
  punggung kotak itu (warna per bagian, dua arah) -> yang mirip dibaca lebih dulu ("mirip K03"). Posisi lama TIDAK
  dipakai, jadi kotak yang dipindah tetap ditemukan. AMBIL tetap hanya setelah verifikasi ulang. TERAMBIL -> entri
  ditandai `diambil`. HUD dan ringkasan menampilkan detik sejak permintaan sampai AMBIL.
- Pindai ulang = petakan ulang seluruh rak (inventaris lama dicadangkan). `--lanjut` = bagian rak berikutnya.
- Kamera harus diam saat pindai: setiap kali posisi kamera hilang semua nomor kotak diulang; catatan ganda
  digabung otomatis (identitas + tampilan sama, satu per segmen), sisanya ditandai "mungkin kotak yang sama".
- File: `data/rak/inventaris.csv`, `data/rak/punggung/Kxx.jpg`, `data/rak/lembar_cek.jpg` (cetak untuk perawat).

## 3e. Stress test (semua kotak bergiliran, ganti formasi, permintaan acak)

`scripts/stress_plan.py` membuat rencana dari inventaris: (1) semua kotak bergiliran, (2) N kali ganti formasi,
tiap kali K kotak diubah (pindah slot / balik / berbaring / pindah sisi / miring) lalu diminta lagi, (3) permintaan
acak. `test_webcam_box.py --rencana` menjalankannya: tiap barang diberi batas waktu (`--waktu-per-barang`, default
90 s); belum AMBIL -> dilewati dan dicatat. Barang tidak diambil (cukup AMBIL yang bertahan 2 s), jadi inventaris
tidak berubah. Tiap barang dibaca dari nol (waktu per kotak bisa dibandingkan); `--pakai-ulang-bacaan` = seperti
operasi biasa. Saat jeda formasi: ubah kotak, tangan keluar dari rak, ketik `lanjut` di terminal; `lewati` =
barang sekarang dilewati.

```bat
:: laptop, cmd
venv\Scripts\python.exe scripts\stress_plan.py --formasi 5 --per-formasi 5 --acak 10 --out results\rencana_stress.txt
venv\Scripts\python.exe scripts\test_webcam_box.py --rencana results\rencana_stress.txt --waktu-per-barang 90 --expected 20 --inventaris data/rak/inventaris.csv --log-stress results\stress_01.csv --debug-readings results\bacaan_stress_01.csv --record results\uji_stress_01.mp4
```

```bash
# server, terminal Remmina (stream + tunnel dari laptop tetap terbuka, lihat 3b)
python scripts/stress_plan.py --formasi 5 --per-formasi 5 --acak 10 --out results/rencana_stress.txt
python scripts/test_webcam_box.py --config config/box_pipeline_config_server.yaml --source http://127.0.0.1:18090/video.mjpg --rencana results/rencana_stress.txt --waktu-per-barang 90 --expected 20 --inventaris data/rak/inventaris.csv --log-stress results/stress_server_01.csv --debug-readings results/bacaan_stress_server_01.csv --record results/uji_stress_server_01.mp4
```

Hasil per barang (`--log-stress`): hasil (AMBIL / WAKTU HABIS / TIDAK ADA / DILEWATI), detik, nomor kotak, bukti
(teks / REF / barcode), teks terbaca, berapa kali CEK ULANG / BATAL, jumlah kotak di rak, jumlah bacaan dan ms per
bacaan. Ringkasan per bagian dicetak di akhir. Kirim CSV hasil + file bacaan + rekaman untuk dianalisis.

## 4. Evaluasi manual (foto uji yang tidak pernah dipakai training)

```bash
python scripts/eval_box_manual.py --images <foto uji>/images --labels <foto uji>/labels
```
Hasil: `results/eval_*/tabel_box.csv` (isi kolom `kenapa_gagal` sendiri), `ringkasan.txt` (rumus + angka),
`vis/` (hijau TP, merah FP, kuning FN). Buka gambarnya dan cocokkan dengan angka di tabel.

## Batasan yang sudah diketahui (dari uji di sandbox)

- OCR adalah bagian paling lambat (sekitar 0,7-2 detik per crop di CPU sandbox). Karena itu OCR
  hanya dijalankan pada crop size_label, maksimal 2 crop per frame, dan tiap box dibaca ulang
  paling cepat tiap 15 frame. Ukur ulang di laptop.
- MATCH butuh nama produk terbaca. Kalau nama produk tidak berada di dalam label size_label mana
  pun (contoh Terumo Accuforce: angka ukuran di satu blok, nama produk di blok lain), sistem akan
  CONFIRM, bukan MATCH. Solusinya di data: beri size_label juga pada blok nama produk.
- Label yang buram / tidak fokus tidak terbaca -> CONFIRM (aman, tidak menebak).

# Sisa pekerjaan setelah 14 Okt 2026: langkah demi langkah

Versi Inggris: `docs/REMAINING_WORK.md`. Kerjakan berurutan dari atas; centang kotak di bagian akhir.

| # | Tugas | Di mana | Waktu | Selesai bila |
| --- | --- | --- | --- | --- |
| 0 | Pasang tool review baru di server | laptop, 1 perintah | 5 menit | output `done` |
| 1 | Koreksi box 33 foto close-up | CVAT di server, lewat browser laptop | 1,5-2 jam | `33 converted, 0 to check` |
| 2 | Review label tanggal 145 foto, lalu training 3 kelas | terminal server | 5-7 jam review (6 sesi) + 1,5 jam training otomatis | hasil `bandingkan_model.py` dikirim ke Claude |
| 3 | Uji live dan rekam data di depan rak | laptop + D455 | 2-3 jam per kunjungan | folder sesi + `LAPORAN_SESI.md` |

Tugas 3 tidak bergantung pada tugas 1-2; boleh dikerjakan di hari mana pun Anda berada di depan rak.

## Cara membuka terminal server (dipakai di tugas 0-2)

Pilih satu cara:
* **VS Code Remote-SSH** (disarankan): `F1` -> `Remote-SSH: Connect to Host...` -> `trse-server` -> `Terminal > New Terminal`.
* **cmd laptop**: `ssh trse-server`.

Setiap kali terminal server dibuka, jalankan dulu:
```bash
cd ~/Downloads/prelabel_size_label_full && source ~/trse_venv/bin/activate
```
Semua perintah `python tools/...` di bawah dijalankan dari sini.

---

## Tugas 0. Pasang tool review baru (5 menit)

Paketnya sudah diunggah ke server. Di **cmd laptop**:
```bat
cd C:\Users\sacab\Downloads\mackay_vision\labeling
RUN_ON_SERVER.bat langkah9_tools_review
```
Benar bila baris terakhir `done`. Isinya:
* `tools/reviewed_to_spec.py` (baru): mengubah hasil koreksi CVAT kembali menjadi spec, supaya tool tanggal dan training
  memakai box yang sudah dikoreksi;
* `tools/review_tanggal.py`: melewati foto yang box-nya masih draf, dan menampilkan alamat lembar gambarnya.

---

## Tugas 1. Koreksi box 33 foto close-up (1,5-2 jam)

Foto: 28 Sep jam 16, satu kotak per foto (Sled Bag, iVascular, Terumo, dll.) di depan meja, monitor, dan botol minum.
Draf box dibuat oleh model v4 dan **sering salah**: botol atau monitor dianggap box, dan box hanya menutupi bagian yang
berwarna. Tugas Anda hanya **box** dan **size_label**. Tanggal dikerjakan di tugas 2.

### 1.1 Buka CVAT di browser laptop
1. Di cmd laptop (biarkan jendelanya terbuka selama bekerja):
   ```bat
   ssh -N -L 8080:localhost:8080 trse-server
   ```
2. Buka browser di laptop: `http://localhost:8080`, lalu login dengan akun CVAT server.

### 1.2 Project (sekali saja)
Bila project **TRSE box size_label** sudah ada, langsung ke 1.3. Bila belum: **Projects -> + -> Create a new project**,
nama `TRSE box size_label`, tambahkan label `box` dan `size_label` (tipe **Rectangle**), lalu **Submit & Open**.

### 1.3 Buat task
1. Di dalam project: **+ -> Create a new task**, nama `baru_20260928_16h`.
2. **Select files -> My computer**, pilih **ke-33 foto** dari folder laptop
   `C:\Users\sacab\Downloads\mackay_vision\labeling\batches\baru_20260928_16h\images`.
   (Isinya identik dengan `raw/` di server; sudah dicek dengan hash 14 Okt.)
3. **Advanced configuration -> Image quality** = `95`, lalu **Submit & Open**.

### 1.4 Muat draf
**Actions -> Upload annotations**, format **CVAT 1.1**, file
`C:\Users\sacab\Downloads\mackay_vision\labeling\batches\baru_20260928_16h\cvat_annotations.xml`.

### 1.5 Koreksi tiap foto (2-3 menit per foto)
Tombol: **F** / **D** foto berikut / sebelumnya, **Ctrl+S** simpan (sering), **Delete** hapus, **N** gambar lagi,
pegangan bulat di atas box untuk memutar.

Untuk setiap foto:
- [ ] Hapus box dan size_label yang menempel pada **botol, monitor, keyboard, meja, atau tutup botol**.
- [ ] Box = **seluruh punggung kemasan yang terlihat**, dari ujung ke ujung, sejajar punggung. Bila punggung keluar dari
      foto, box berhenti di tepi foto. Jangan masukkan sisi samping, ujung kemasan, atau jari.
- [ ] Satu `size_label` pada kelompok nama produk + ukuran, selalu di dalam box-nya.
      Terumo: dua (nama+ukuran, dan blok logo TERUMO). iVascular: panel berwarna. Sled Bag: logo Boston Scientific +
      tulisan "Permanent Sled Bag".
- [ ] REF, LOT, barcode, atau tanggal saja **tidak** diberi size_label. Jangan membuat label `expiry_label` di sini.
- [ ] Kemasan kedua yang hanya terlihat sebagai potongan tipis (< 1 % lebar foto) tidak dilabel.

Foto yang paling perlu perhatian:

| Foto (akhir nama) | Masalah draf |
| --- | --- |
| 162600_587 | tidak ada box sama sekali: gambar box kemasan Boston Scientific |
| 162335_447, 163033_361, 163052_907 | ada box yang kempis jadi garis di tepi foto: hapus |
| 162031_994, 162154_336, 162159_717, 163611_481, 163640_459, 163052_907 | 2-3 box di foto satu kotak: biasanya satu di botol |
| 162552_123, 163517_241, 163605_296, 163710_214 | box terlalu pendek: panjangkan sampai menutup seluruh punggung |

### 1.6 Ekspor dan kirim ke server
1. **Actions -> Export task dataset**, format **Ultralytics YOLO Oriented Bounding Boxes 1.0**, **Save images** mati, **OK**.
   (Jangan "YOLO 1.1" atau "Ultralytics YOLO Detection": rotasinya hilang dan akan ditolak.)
2. Unduh zip dari menu **Requests**; masuk ke folder Downloads laptop.
3. Cmd laptop (ganti `NAMA_FILE`):
   ```bat
   scp "%USERPROFILE%\Downloads\NAMA_FILE.zip" trse-server:~/Downloads/baru_16h_cvat.zip
   ```

### 1.7 Masukkan ke proyek (terminal server)
```bash
python tools/import_reviewed.py ~/Downloads/baru_16h_cvat.zip --dry
```
* `KESALAHAN - tidak ada file yang ditulis` -> perbaiki foto yang disebut di CVAT, ekspor lagi.
* `BELUM DIKOREKSI` / `sama persis dengan draf model` -> foto itu belum dikoreksi. Bila drafnya memang sudah benar,
  tambahkan `--allow-unchanged`.
* `PERINGATAN` (mis. size_label di luar box) -> cek foto itu sekali lagi; tetap boleh lanjut.
* Bila bersih, jalankan tanpa `--dry`:
```bash
python tools/import_reviewed.py ~/Downloads/baru_16h_cvat.zip
python tools/reviewed_to_spec.py --batch baru_20260928_16h
```
Benar bila baris terakhir **`33 converted, 0 to check. Next: python tools/review_tanggal.py`**. Tool ini juga langsung
membuat usulan tanggal baru pada box yang sudah dikoreksi (sekitar 2 menit).

Cek akhir (harus `0`): `grep -l "SEDANG DIKERJAKAN" label_specs/*.json | wc -l`

---

## Tugas 2. Review label tanggal, lalu training 3 kelas

145 foto masih berstatus `auto`: 112 foto lama (termasuk 26 yang dibuka lagi 12 Okt karena ada tanggal terlewat) dan 33
foto dari tugas 1. Foto ikut training **hanya** setelah tanggalnya di-`ok`. Satu tanggal terlihat yang tidak dilabel akan
diajarkan ke model sebagai "bukan tanggal", jadi jangan `ok` bila ragu.

### 2.1 Mulai sesi review
Paling mudah lewat **desktop Ubuntu (Remmina, Quality: Best)**: lembar gambar terbuka otomatis. Lewat **VS Code
Remote-SSH** juga bisa: klik alamat `lembar: preview_expiry/..._cek.jpg` yang tercetak (Ctrl+klik), gambarnya terbuka
di tab VS Code.
```bash
python tools/review_tanggal.py
```
Judul menunjukkan kemajuan, mis. `===== IMG_... (selesai 51/196, sisa 145) =====`. Urutan foto otomatis (paling
bermasalah dulu). Foto yang sudah `ok` tidak muncul lagi; berhenti kapan saja dengan `keluar`.

### 2.2 Cara membaca lembar
* Satu strip = satu kotak, **ujung jauh di kiri, ujung dekat di kanan**. Nomor box `N` tertulis di bawah strip.
* **Magenta** = label lama, **cyan** = usulan baru, **bingkai merah** = PERIKSA (strip tajam tapi belum ada label).
* Tanggal kedaluwarsa = tanggal di **sebelah ikon jam pasir**. iVascular mencetak 3 baris: LOT (atas), **KEDALUWARSA
  (tengah)**, produksi (bawah); label hanya pada baris tengah.

### 2.3 Perintah per kotak

| Yang Anda lihat | Ketik |
| --- | --- |
| Label pas pada tanggal, angkanya benar | (tidak perlu apa-apa) |
| Label pas, tapi angkanya salah baca | `tgl N 2028-01-31` |
| Label menempel pada LOT / tanggal produksi / bukan tanggal | `hapus N` |
| Tanggal terlihat jelas, tidak ada label | `tambah N T1 T2 2028-09-30` |
| Tanggal buram / tidak terbaca, ada label | `buang N` |
| Bingkai merah, memang tidak ada tanggal terlihat | `kosong N` |
| Semua tanggal yang terlihat sudah berlabel benar | `ok` (foto berikutnya) |
| Ragu | `lewati` |
| Istirahat | `keluar` |

`T1 T2` = posisi tanggal sepanjang strip (0 = kiri, 1 = kanan). Contoh: tanggal di 45 %-60 % panjang strip box 5:
`tambah 5 0.45 0.60 2028-09-30` (label selebar kotak). Label lebih rapat: tambahkan batas melintang `V1 V2` (0..1),
mis. `tambah 5 0.45 0.60 0.3 0.6 2028-09-30`; lihat di lembar sisi mana yang 0, lalu ulangi bila perlu.
Lembar digambar ulang setelah setiap perintah, jadi posisi bisa langsung dicek dan diulang (`hapus N` lalu `tambah` lagi).

Tanggal mencurigakan yang sudah diketahui: `2038-01-31`, `2037-06-30`, `2036-09-30` (kemungkinan 2028 / 2027 / 2028
salah baca), dan `2026-09-30` (sudah lewat; cek di lembar).

### 2.4 Target dan ritme
* 25 foto per sesi (45-60 menit), kurang lebih 6 sesi. Banyak foto cukup dilirik lalu `ok`.
* **Minimal sebelum training:** ke-112 foto lama selesai (perbandingan dengan v6 hanya adil bila hampir semua foto ikut).
  33 foto baru boleh menyusul.

### 2.5 Cek kesiapan training (aman, tidak mengubah apa-apa)
```bash
CEK=1 bash tools/mulai_training_3kelas.sh
```
Benar bila **`SEMUA SYARAT TERPENUHI (CEK=1: tidak training)`**. Bila berhenti:

| Pesan | Tindakan |
| --- | --- |
| `foto 'selesai' masih punya kotak 'periksa' terbuka` | buka foto itu lagi: `python tools/set_expiry.py NAMA --redo`, lalu review |
| `seed N foto (minimal 60)` | review lebih banyak foto |
| `Ada fold dengan < 5 foto` | review juga foto dari tanggal lain (23 Sep, 28 Sep, 1 Okt) |
| `VRAM kurang` | GPU dipakai orang lain; coba lagi nanti |

### 2.6 Training (otomatis, 1-1,5 jam)
```bash
bash tools/mulai_training_3kelas.sh
tail -3 logs/train_3kelas.log          # cek kapan saja; selesai bila ada baris "model final"
```
Skrip mencadangkan hasil lama dulu ke `~/Downloads/cadangan_sebelum_3kelas_<waktu>`. Terminal boleh ditutup.

### 2.7 Bandingkan dan kirim hasil
```bash
python tools/bandingkan_model.py
grep -A6 "Hasil cross-validation" logs/train_3kelas.log
cat logs/best_model.json
```
Kirim ketiga output ke Claude. Aturan: model baru boleh menggantikan v6 hanya bila mAP50-95 dan recall untuk `box` dan
`size_label` turun tidak lebih dari 0,02, **dan** uji keselamatan (0 MATCH salah) lolos. Model demo (v4 di laptop) tidak
diganti sebelum itu.

---

## Tugas 3. Di depan rak (laptop + RealSense D455)

### 3.1 Persiapan (10 menit)
- [ ] Charger terpasang, mode daya **Best performance**, VS Code dan browser ditutup (RAM terbatas).
- [ ] D455 di port **USB 3**, di tripod, seluruh rak terlihat. Cek: `TEST_REALSENSE.bat` menu **1** harus `3.x`.
- [ ] Ke-20 kotak di rak. Foto rak dengan HP (acuan mengembalikan kotak).

### 3.2 Sesi uji terekam (60-90 menit)
```bat
cd C:\Users\sacab\Downloads\mackay_vision\code
DEMO.bat
```
Pilih **S**. Jendela kecil "SCREEN RECORDER" muncul: biarkan, berhenti sendiri di akhir.

| Langkah | Lolos bila | Bila gagal |
| --- | --- | --- |
| 1 Cek perangkat | `READY: 0 FAIL` | baca baris FAIL; boleh lanjut (`y`), tercatat |
| 2 Verifikasi | `... passed`, tidak ada `failed` | berhenti, laporkan |
| 3 Kalibrasi | tidak ada FAIL (WARN lebar box boleh) | geser kamera, ulangi sesi |
| 4 Uji perangkat 1 | HUD `boxes 20`, sharpness >= 300 selama 10 detik | atur kamera / cahaya, `q`, ulangi |
| 5 Uji semua kotak | tiap kotak berakhir TAKEN, PICK, NOT FOUND, atau TIMEOUT | `lewati` di cmd untuk melewati satu kotak |
| 6 Permintaan acak | tiap permintaan dijawab | Enter di baris kosong mengakhiri sesi |

Hasil yang wajar (uji 10-13 Okt): iVascular (nomor 5, 9, 14) mencapai PICK dari jarak rak; merek lain biasanya tetap
**candidate**. Itu aturan keselamatan yang bekerja (PICK butuh dua sumber), bukan error.

### 3.3 Uji khusus tanggal Sled Bag (di langkah 6 sesi S)
Ini menguji perbaikan 14 Okt (ikon jam pasir yang terbaca sebagai "8").

| Ketik | Kotak | Hasil yang benar |
| --- | --- | --- |
| `2` | K03, 2029-03-10 | PICK **hanya** setelah tanggal K03 terbaca |
| `16` | K17, 2029-03-12 (beda satu angka dari K03) | PICK K17, **tidak pernah** K03 |
| `0` | Sled Bag mana saja | yang kedaluwarsa paling awal dulu (2029-03-10) |

Cara: setelah mengetik nomor, dekatkan kamera (geser tripod) ke punggung Sled Bag sampai tanggal di sebelah jam pasir
**terbaca jelas di layar live**, tahan diam 3-5 detik. Dari jarak rak penuh tanggal tidak terbaca, jadi hasilnya
candidate (wajar). Catat untuk tiap nomor: PICK kotak yang benar / candidate / PICK kotak yang SALAH.
**PICK kotak yang salah = hentikan uji dan laporkan segera.**

### 3.4 Rekam data rak (menu D, 10-15 menit per sesi)
```bat
TEST_REALSENSE.bat
```
Pilih **D**, ketik nama sesi. Minggu ini dua sesi:

| Sesi | Kamera |
| --- | --- |
| `s1_far` | seluruh rak (20 kotak) |
| `s2_mid` | lebih dekat, lebar box >= 40 px: setengah kiri lalu setengah kanan |

Kalibrasi tampil dulu: atur sampai tidak ada FAIL. Lalu rekam 1-2 menit: kamera **diam 3 detik** di tiap posisi,
digeser pelan di antaranya; `q` untuk berhenti. File masuk `results\data_rak\`. Jangan hapus sesi yang gagal.

Ambil frame untuk training (cmd laptop, folder `code`):
```bat
venv\Scripts\python.exe scripts\ambil_frame_data.py results\data_rak\s1_far.mp4 results\data_rak\s2_mid.mp4
```
Hasilnya di `C:\Users\sacab\Downloads\mackay_vision\labeling\raw_d455\`. Beri tahu Claude: frame dikirim ke server dan
disiapkan sebagai draf + batch CVAT seperti 33 foto di tugas 1.

### 3.5 Setelah kunjungan
Beri tahu Claude nama folder `results\sesi_<tanggal_waktu>\`; Claude membaca `LAPORAN_SESI.md`, `05_stress.csv`, dan
`verify_log.csv` lalu menulis ringkasannya. Inventaris asli (`data\rak\inventaris.csv`) tidak pernah diubah oleh sesi.

---

## Daftar centang

- [ ] 0 `RUN_ON_SERVER.bat langkah9_tools_review` -> `done`
- [ ] 1.1-1.4 task CVAT `baru_20260928_16h` dengan draf
- [ ] 1.5 33 foto dikoreksi
- [ ] 1.6-1.7 impor + `reviewed_to_spec.py` -> `33 converted, 0 to check`
- [ ] 2 review tanggal: sesi 1 [ ] 2 [ ] 3 [ ] 4 [ ] 5 [ ] 6 [ ]  (112 foto lama selesai [ ], 33 foto baru [ ])
- [ ] 2.5 `CEK=1` -> SEMUA SYARAT TERPENUHI
- [ ] 2.6 training selesai ("model final")
- [ ] 2.7 hasil dikirim ke Claude
- [ ] 3.2 sesi S terekam
- [ ] 3.3 uji Sled Bag: nomor 2 [ ] 16 [ ] 0 [ ]
- [ ] 3.4 sesi data `s1_far` [ ] `s2_mid` [ ], frame diambil

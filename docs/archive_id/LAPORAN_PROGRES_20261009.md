# Laporan progres — Jumat, 9 Oktober 2026

Dasar: keputusan meeting 5 Oktober. Konfirmasi akhir tetap di dokter, tanpa LLM, hanya dua kategori barang
(balon dan stent), antarmuka utama software TM. Demo program: **Senin, 12 Oktober**.

## Ringkasan

- Program demo permintaan teks sudah berjalan di laptop dan di server. **20 dari 20** uji skenario lulus,
  **205** tes otomatis lulus di kedua tempat.
- Model deteksi kotak untuk demo: **v4** (terpasang di server). **v6** sudah dilatih dan diuji: aman (**0 MATCH salah**)
  tetapi setara v4/v5, bukan lebih baik, sehingga tidak dipakai.
- Belum diuji: kamera atau stream asli, dan posisi rak/level asli dari tim (tabel sekarang masih contoh).

## 1. Demo permintaan teks

Input: merek + produk + ukuran. Output: status, kotak yang dipilih, posisi rak/level, dan pesan JSON untuk
software TM. Kotak dipilih dengan **FEFO ketat**: selalu kedaluwarsa paling awal. Teks saja tidak pernah
menghasilkan MATCH, dan setiap pesan membawa `needs_doctor_confirmation: true`.

Hasil di server, 8 Oktober:

| Permintaan | Status | Kotak | Keterangan |
| --- | --- | --- | --- |
| `ivascular angiolite 2.5 x 29` | SIAP | K15 | rak R1, level 1, kolom 15 |
| `angiolite 4x19` | SIAP | K10 | K10 dikembalikan ke rak 8 Okt sore; sebelumnya TIDAK ADA |
| `angiolite 3 x 20` | TIDAK ADA | – | tidak ada di inventaris |
| `angiolite 2.5` | ULANGI | – | panjang belum ditulis |
| `sled bag` | SIAP | K03 | 4 kotak; K03 kedaluwarsa paling awal (2029-03-10) |
| `terumo xperience 2.5 x 15` | ULANGI | – | xperience pro bukan produk Terumo |

Contoh pesan untuk TM (satu baris JSON per permintaan):

```json
{"status": "siap_verifikasi", "brand": "ivascular", "product": "angiolite", "diameter_mm": 2.5,
 "length_mm": 29.0, "box_id": "K15", "rack": "R1", "level": "1", "column": "15", "stock_count": 1,
 "needs_doctor_confirmation": true, "reason": "posisi dari tabel CONTOH (belum dari tim)"}
```

*Tangkapan layar dari gladi Remmina: [ditambahkan setelah gladi].*

## 2. Hasil uji

| Uji | Laptop | Server |
| --- | --- | --- |
| Tes otomatis (`pytest`) | 205 lulus | 205 lulus |
| 20 skenario teks | 20/20 lulus | 20/20 lulus |
| Uji kamera nomor 1–17 | belum | belum |

Rincian 20 skenario ada di `docs/CHECKLIST_DEMO.md`: sejak K10 kembali ke rak (8 Okt), **17 barang** ada di rak
(SIAP, termasuk #17 angiolite 4x19), 1 kombinasi tidak ada, dan 2 permintaan yang harus diulang.

## 3. Model deteksi kotak dan label ukuran

Angka di bawah berasal dari cross-validation 5 fold, yaitu skor pada foto yang tidak dilihat model saat
training. Skor pada foto training sendiri (sekitar 0,83) tidak dipakai karena terlalu optimistis.

| Model | Tanggal | Foto | mAP50-95 | Recall label ukuran | Status |
| --- | --- | --- | --- | --- | --- |
| v4 | 29 Sep | 144 | 0,696 ± 0,044 | 0,962 | dipakai untuk demo |
| v5 | 2 Okt | 152 | 0,697 ± 0,039 | 0,959 | setara v4 |
| v6 | 8 Okt | 156 | 0,713 ± 0,032 | 0,953 | selesai dilatih; setara v4/v5, tidak dipakai (lihat di bawah) |

Uji keselamatan dengan detektor (kotak dari prediksi model, bukan label manual; foto diprediksi oleh model yang
tidak pernah melihat foto itu), 123 kotak, 12 jenis barang:

| Model | Kotak terdeteksi | MATCH pada permintaan benar | Salah MATCH |
| --- | --- | --- | --- |
| v4 (terpasang)* | 123/123 | 13 (11%) | **0** dari 1.353 |
| v5 (out-of-fold) | 119/123 (96,7%) | 13 (11%) | **0** dari 1.309 |
| v6 (out-of-fold) | 117/123 (95,1%) | 13 (11%) | **0** dari 1.287 |

*v4 diuji pada foto yang ikut melatihnya, jadi 100% tidak mencerminkan foto baru; tidak sebanding dengan v5/v6.

Kesimpulan: ketiganya aman (0 salah MATCH). v6 tidak lebih baik dari v5 pada deteksi kotak (6 kotak terlewat vs 4) dan
skor CV-nya hanya sedikit di atas v4/v5 (di dalam rentang simpangan). **Demo Senin tetap memakai v4.**

Catatan:
- v5 ternyata sudah pernah dilatih. Bobotnya ditemukan di arsip server bersama v4 dan sudah dipulihkan.
- v6 memakai 156 foto yang sudah selesai direview. Tujuh foto yang masih "perlu dicek manual" sengaja
  tidak dimasukkan. Pembagian validasi sekarang menjaga foto yang diambil berdekatan (< 5 detik) tetap
  dalam satu kelompok, supaya skor tidak terlalu optimistis. Akibatnya angka v6 tidak bisa dibandingkan
  persis satu per satu dengan v4/v5.

**Verifikasi foto** (evaluasi 4 Oktober, 123 kotak, 12 jenis barang): **0 MATCH salah** dari 1.353
permintaan yang salah, termasuk 92 produk sama dengan ukuran berbeda. Namun dari 123 permintaan yang benar,
hanya 16 (13%) yang langsung MATCH; 44 menjadi kandidat dan 63 perlu konfirmasi. Sistem sudah aman
(tidak salah ambil), tetapi masih sering meminta konfirmasi dokter.

## 4. Perbaikan pipeline data

Audit 4 Oktober menemukan delapan masalah di pipeline label dan training. Semuanya sudah diperbaiki:
- Foto yang belum selesai direview tidak lagi ikut training.
- Pembagian train/val dan fold tidak lagi memisahkan foto yang hampir sama.
- Hasil training lama tidak dipakai ulang jika foto, ukuran gambar, atau jumlah epoch berubah.
- Impor label menolak koordinat rusak (NaN), dan pembuatan batch tidak lagi meninggalkan foto lama.
- Pemeriksaan geometri label sekarang menghentikan pipeline jika menemukan masalah yang belum direview.
- Bobot dan log training lama (v2–v5) dipulihkan, dan daftar checksum dibuat ulang.

## 5. Risiko dan yang belum selesai

| Hal | Dampak | Rencana |
| --- | --- | --- |
| Kamera/stream asli belum diuji | Demo kamera bisa gagal saat Senin | Uji nomor 1–16 sebelum Senin |
| Tabel rak/level masih contoh | Posisi yang dikirim ke TM belum benar | Menunggu format dari tim |
| 16 foto dengan peringatan geometri, 7 foto belum direview | Training berikutnya terhambat | Review visual |
| Pembacaan label masih konservatif (11% MATCH langsung) | Dokter sering diminta konfirmasi | Perbaiki pembacaan dari jarak dekat/barcode; bukan masalah model deteksi |
| GPU server dipakai bersama | Training v6 memakan 4 jam | Selesai 8 Okt; training berikutnya dijadwalkan malam hari |

## 6. Pertanyaan untuk tim

1. Format file posisi rak/level: kolom apa saja, dan per kotak atau per jenis barang?
2. Bentuk data yang diterima software TM: JSON, teks, atau socket/ROS? Sementara ini dipakai pesan JSON di atas.
3. Siapa yang memegang antarmuka konfirmasi suara (warna + "execute")?

## 7. Rencana sampai Senin

1. Gladi demo lewat remote desktop, dan rekam layar sebagai cadangan.
2. Uji kamera untuk 17 barang (termasuk #17 angiolite 4x19, K10, pasangan mirip dengan #1). Aturannya: tidak boleh ada MATCH pada kotak yang salah.
3. v6 sudah dibandingkan: setara, tidak lebih baik. Demo memakai v4.
4. Ganti tabel rak/level contoh jika data dari tim sudah ada.

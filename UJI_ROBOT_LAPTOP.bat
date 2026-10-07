@echo off
rem Uji pemilihan kotak di laptop dengan skenario terbaik:
rem model v4 (yolo11s-obb, OpenVINO iGPU 960), OCR 2 thread, bukti digabung antar bacaan, kunci AMBIL stabil,
rem inventaris rak (kalau data\rak\inventaris.csv ada), tanpa perekaman video. Setiap bacaan dicatat untuk analisis.
cd /d "%~dp0"
title Uji robot - pemilihan kotak (laptop)
echo.
echo  SEBELUM MULAI:
echo   1. Webcam w300 terpasang DIAM (tripod / dijepit) di depan rak
echo   2. Program lain yang memakai webcam sudah ditutup
echo.
set "REQ=angiolite,2.5,29"
set /p REQ=Kotak yang diminta [Enter = angiolite,2.5,29]:
set "N=0"
set /p N=Jumlah kotak di rak [Enter = tidak tahu]:
set "INV="
if exist data\rak\inventaris.csv set "INV=--inventaris data/rak/inventaris.csv"
if defined INV (echo  inventaris rak: data\rak\inventaris.csv dipakai) else (echo  inventaris rak: belum ada - cari tanpa inventaris)
echo.
echo  Jendela video muncul dalam 10-60 detik. Tombol di jendela video: r = minta lagi, b = pindai dekat, q = selesai
echo    a = pilih AREA RAK (tarik kotak di sekitar rak lalu Enter; disimpan dan dipakai lagi selama kamera tidak dipindah)
echo  Permintaan berikutnya: ketik di SINI (mis. accuforce,2.75,20) lalu Enter
echo    beberapa barang: angiolite 2.5 29; accuforce 2.75 20 jumlah 2   ^|  tambah: + essential pro 3 40   ^|  lewati
echo.
venv\Scripts\python.exe scripts\test_webcam_box.py --request "%REQ%" --expected %N% %INV% --debug-readings results\bacaan_terakhir.csv
echo.
echo  Selesai. Log: results\verify_log.csv   Foto bukti: results\screens\peristiwa   Bacaan: results\bacaan_terakhir.csv
pause

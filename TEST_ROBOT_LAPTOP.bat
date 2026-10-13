@echo off
rem Box selection test on the laptop with the best setup so far:
rem model v4 (yolo11s-obb, OpenVINO iGPU 960), OCR in 2 threads, evidence combined across readings, stable PICK lock,
rem shelf inventory (if data\rak\inventaris.csv exists), no video recording. Every reading is logged for analysis.
cd /d "%~dp0"
title Robot test - box selection (laptop)
echo.
echo  BEFORE YOU START:
echo   1. Webcam w300 mounted STILL (tripod / clamp) in front of the shelf
echo   2. Every other program that uses the webcam is closed
echo.
set "REQ=angiolite,2.5,29"
set /p REQ=Requested box [Enter = angiolite,2.5,29]:
set "N=0"
set /p N=Number of boxes on the shelf [Enter = unknown]:
set "INV="
if exist data\rak\inventaris.csv set "INV=--inventaris data/rak/inventaris.csv"
if defined INV (echo  shelf inventory: data\rak\inventaris.csv is used) else (echo  shelf inventory: none yet - searching without it)
echo.
echo  The video window appears in 10-60 s. Keys in the video window: r = request again, b = close scan, q = quit
echo    a = choose the SHELF AREA (drag a box around the shelf, then Enter; kept while the camera does not move)
echo  Next request: type it HERE (e.g. accuforce,2.75,20) and press Enter
echo    several items: angiolite 2.5 29; accuforce 2.75 20 jumlah 2   ^|  add: + essential pro 3 40   ^|  lewati = skip
echo.
venv\Scripts\python.exe scripts\test_webcam_box.py --request "%REQ%" --expected %N% %INV% --debug-readings results\bacaan_terakhir.csv
echo.
echo  Done. Log: results\verify_log.csv   Event pictures: results\screens\peristiwa   Readings: results\bacaan_terakhir.csv
pause

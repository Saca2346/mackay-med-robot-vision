@echo off
rem RealSense D455 test on the LAPTOP (one menu). All evidence goes to results\bukti_laporan_20261008.
rem Usage:  TEST_REALSENSE.bat   then choose a number. In the camera window: q = stop.
rem Recorded tests also save an ANNOTATED video (*_vis.mp4: boxes, labels, status) for reports.
rem Recordings of tests 4, 7, 8, 9 get a time stamp in the name (nothing is overwritten).
rem Tests 4, 7, 8, 9 use a fresh COPY of the inventory (inventaris_uji.csv): taking boxes during a test never
rem changes data\rak\inventaris.csv, so no "kembali" is needed afterwards.
cd /d "%~dp0"
set B=results\bukti_laporan_20261008
set C=%B%\config_realsense.yaml
set P=venv\Scripts\python.exe
set I=%B%\inventaris_uji.csv
:menu
for /f %%i in ('powershell -nologo -noprofile -command "Get-Date -Format yyyyMMdd_HHmmss"') do set T=%%i
echo.
echo ===== REALSENSE TEST =====
echo  1  Check USB (must be 3.x, Ctrl+C to stop)
echo  2  Detection only (adjust the camera; box width and sharpness are in docs\DATA_RAK_PROTOCOL.md)
echo  3  Record test 2 min: angiolite 2.5 x 29  (rs_uji1)
echo  4  Record test: another box, type the request (rs_uji2)
echo  5  Replay rs_uji1.mp4 with the 960 model (default)
echo  6  Replay rs_uji1.mp4 with the 800x1280 model
echo  7  ALL BOXES IN TURN, boxes stay on the shelf: found (PICK held 2 s) = next box; max 90 s per box
echo  8  MANUAL: type each request yourself; after TAKEN type the next one (list: %B%\daftar_permintaan_acak.txt)
echo  9  LIST, TAKE EACH BOX: PICK -^> take the box -^> empty slot verified (TAKEN) -^> next box of the list; max 180 s
echo  D  SHELF DATA: camera calibration first, then record a named session (docs\DATA_RAK_PROTOCOL.md)
echo  0  Exit
set /p PILIH=Choose a number: 
if "%PILIH%"=="4" copy /Y data\rak\inventaris.csv %I% >nul
if "%PILIH%"=="7" copy /Y data\rak\inventaris.csv %I% >nul
if "%PILIH%"=="8" copy /Y data\rak\inventaris.csv %I% >nul
if "%PILIH%"=="9" copy /Y data\rak\inventaris.csv %I% >nul
if "%PILIH%"=="1" %P% scripts\cek_usb_realsense.py
if "%PILIH%"=="2" %P% scripts\test_webcam_box.py --config %C% --expected 20
if "%PILIH%"=="3" %P% scripts\test_webcam_box.py --config %C% --request "angiolite,2.5,29" --expected 20 --record %B%\rs_uji1.mp4 --record-vis %B%\rs_uji1_vis.mp4 --debug-readings %B%\bacaan_uji1
if "%PILIH%"=="4" goto lain
if "%PILIH%"=="5" %P% scripts\test_webcam_box.py --config %C% --weights models\box_v4s_960_openvino_model --imgsz 544,960 --source %B%\rs_uji1.mp4 --request "angiolite,2.5,29" --expected 20 --record-vis %B%\replay_960_vis.mp4 --debug-readings %B%\replay_960
if "%PILIH%"=="6" %P% scripts\test_webcam_box.py --config %C% --weights models\box_v4s_800x1280_openvino_model --imgsz 800,1280 --source %B%\rs_uji1.mp4 --request "angiolite,2.5,29" --expected 20 --debug-readings %B%\replay_1280
if "%PILIH%"=="7" %P% scripts\test_webcam_box.py --config %C% --rencana %B%\rencana_20_kotak.txt --waktu-per-barang 90 --inventaris %I% --expected 20 --log-stress %B%\hasil_bergilir_%T%.csv --record %B%\bergilir_%T%.mp4 --record-vis %B%\bergilir_%T%_vis.mp4 --debug-readings %B%\bacaan_bergilir_%T%
if "%PILIH%"=="8" %P% scripts\test_webcam_box.py --config %C% --request "angiolite,2.5,29" --inventaris %I% --expected 20 --record %B%\manual_%T%.mp4 --record-vis %B%\manual_%T%_vis.mp4 --debug-readings %B%\bacaan_manual_%T%
if "%PILIH%"=="9" %P% scripts\test_webcam_box.py --config %C% --rencana %B%\rencana_20_kotak.txt --until-taken --waktu-per-barang 180 --inventaris %I% --expected 20 --log-stress %B%\hasil_ambil_%T%.csv --record %B%\ambil_%T%.mp4 --record-vis %B%\ambil_%T%_vis.mp4 --debug-readings %B%\bacaan_ambil_%T%
if /I "%PILIH%"=="D" goto data
if "%PILIH%"=="0" exit /b 0
goto menu
:lain
set /p REQ=Request, e.g. accuforce,2.5,12 : 
%P% scripts\test_webcam_box.py --config %C% --request "%REQ%" --inventaris %I% --expected 20 --record %B%\rs_uji2_%T%.mp4 --record-vis %B%\rs_uji2_%T%_vis.mp4 --debug-readings %B%\bacaan_uji2_%T%
goto menu
:data
set /p SES=Session name, no spaces (e.g. s1_far, s2_mid, s3_close): 
set /p REF=Reference session to compare the camera position with (Enter = none): 
if not exist results\data_rak mkdir results\data_rak
if "%REF%"=="" goto cal_plain
%P% scripts\kalibrasi_kamera.py --sesi %SES% --config %C% --ref results\data_rak\%REF%_kalibrasi.json --same-position
goto cal_done
:cal_plain
%P% scripts\kalibrasi_kamera.py --sesi %SES% --config %C%
:cal_done
if errorlevel 1 goto cal_fail
goto record
:cal_fail
echo.
echo Calibration FAILED or was cancelled. See the advice above and results\data_rak\%SES%_kalibrasi.jpg
set /p GO=Record anyway? (y/n): 
if /I not "%GO%"=="y" goto menu
:record
%P% scripts\test_webcam_box.py --config %C% --expected 20 --record results\data_rak\%SES%.mp4 --record-vis results\data_rak\%SES%_vis.mp4
goto menu

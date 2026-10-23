@echo off
rem RealSense D455 test on the LAPTOP (one menu). New recordings and logs go to results\uji_realsense (O);
rem results\bukti_laporan_20261008 (B) is evidence and is only READ (config, replay of rs_uji1.mp4).
rem Usage:  TEST_REALSENSE.bat   then choose a number. In the camera window: q = stop.
rem Recorded tests also save an ANNOTATED video (*_vis.mp4: boxes, labels, status) for reports.
rem Recordings of tests 4, 7, 8, 9 get a time stamp in the name (nothing is overwritten).
rem Tests 4, 7, 8, 9 use a fresh COPY of the inventory (inventaris_uji.csv): taking boxes during a test never
rem changes data\rak\inventaris.csv, so no "kembali" is needed afterwards.
cd /d "%~dp0"
set B=results\bukti_laporan_20261008
set P=venv\Scripts\python.exe
set O=results\uji_realsense
if not exist %O% mkdir %O%
rem own config copy: event log, screens and pictures go to %O%, never into the evidence folder %B%
%P% scripts\sesi_config.py config\box_pipeline_config_realsense.yaml %O% >nul
set C=%O%\config_sesi.yaml
set I=%O%\inventaris_uji.csv
:menu
for /f %%i in ('powershell -nologo -noprofile -command "Get-Date -Format yyyyMMdd_HHmmss"') do set T=%%i
echo.
echo ===== REALSENSE TEST =====
echo  1  Check USB (must be 3.x, Ctrl+C to stop)
echo  2  Detection only (adjust the camera; box width and sharpness are in docs\DATA_RAK_PROTOCOL.md)
echo  3  Record test 2 min: angiolite 2.5 x 29  (saved as uji1_DATE; the evidence rs_uji1 is not touched)
echo  4  Record test: another box, type the request (rs_uji2)
echo  5  Replay rs_uji1.mp4 with the 960 model (default)
echo  6  Replay rs_uji1.mp4 with the 800x1280 model
echo  7  ALL 20 BOXES IN TURN, boxes stay on the shelf: found (PICK held 2 s) = next box; max 90 s per box
echo  8  MANUAL: the numbered box list is shown; type a NUMBER (or the text) per request; ? = list again
echo  9  ALL 20 BOXES, TAKE EACH: PICK -^> take the box -^> empty slot verified (TAKEN) -^> next box; max 180 s
echo  G  GRASP (3D): like 8, plus D455 depth -^> grasp data per PICK in %O%\grasp.jsonl (for the arm team)
echo  P  SHELF MAP for the target lock (do this first after boxes were moved): camera still in front of the whole shelf, 1-2 min, q = save
echo  D  SHELF DATA: camera calibration first, then record a named session (docs\DATA_RAK_PROTOCOL.md)
echo  0  Exit
set /p PILIH=Choose a number: 
if "%PILIH%"=="4" copy /Y data\rak\inventaris.csv %I% >nul
if "%PILIH%"=="7" copy /Y data\rak\inventaris.csv %I% >nul
if "%PILIH%"=="8" copy /Y data\rak\inventaris.csv %I% >nul
if /I "%PILIH%"=="G" copy /Y data\rak\inventaris.csv %I% >nul
if "%PILIH%"=="9" copy /Y data\rak\inventaris.csv %I% >nul
if "%PILIH%"=="1" %P% scripts\cek_usb_realsense.py
if "%PILIH%"=="2" %P% scripts\test_webcam_box.py --config %C% --expected 20
if "%PILIH%"=="3" %P% scripts\test_webcam_box.py --config %C% --request "angiolite,2.5,29" --expected 20 --record %O%\uji1_%T%.mp4 --record-vis %O%\uji1_%T%_vis.mp4 --debug-readings %O%\bacaan_uji1_%T%
if "%PILIH%"=="4" goto lain
if "%PILIH%"=="5" %P% scripts\test_webcam_box.py --config %C% --weights models\box_v4s_960_openvino_model --imgsz 544,960 --source %B%\rs_uji1.mp4 --request "angiolite,2.5,29" --expected 20 --record-vis %O%\replay_960_%T%_vis.mp4 --debug-readings %O%\replay_960_%T%
if "%PILIH%"=="6" %P% scripts\test_webcam_box.py --config %C% --weights models\box_v4s_800x1280_openvino_model --imgsz 800,1280 --source %B%\rs_uji1.mp4 --request "angiolite,2.5,29" --expected 20 --debug-readings %O%\replay_1280_%T%
if "%PILIH%"=="7" %P% scripts\demo_daftar_kotak.py --inventaris %I% --rencana-out %O%\rencana_20_%T%.txt
if "%PILIH%"=="7" %P% scripts\test_webcam_box.py --config %C% --rencana %O%\rencana_20_%T%.txt --waktu-per-barang 90 --inventaris %I% --expected 20 --log-stress %O%\hasil_bergilir_%T%.csv --record %O%\bergilir_%T%.mp4 --record-vis %O%\bergilir_%T%_vis.mp4 --debug-readings %O%\bacaan_bergilir_%T%
if "%PILIH%"=="8" goto manual
if "%PILIH%"=="9" %P% scripts\demo_daftar_kotak.py --inventaris %I% --rencana-out %O%\rencana_20_%T%.txt
if "%PILIH%"=="9" %P% scripts\test_webcam_box.py --config %C% --rencana %O%\rencana_20_%T%.txt --until-taken --waktu-per-barang 180 --inventaris %I% --expected 20 --log-stress %O%\hasil_ambil_%T%.csv --record %O%\ambil_%T%.mp4 --record-vis %O%\ambil_%T%_vis.mp4 --debug-readings %O%\bacaan_ambil_%T%
if /I "%PILIH%"=="P" %P% scripts\rak_inventaris.py peta --inventaris data\rak\inventaris.csv --config %C%
if /I "%PILIH%"=="D" goto data
if /I "%PILIH%"=="G" goto grasp
if "%PILIH%"=="0" exit /b 0
goto menu
:lain
set /p REQ=Request, e.g. accuforce,2.5,12 : 
%P% scripts\test_webcam_box.py --config %C% --request "%REQ%" --inventaris %I% --expected 20 --record %O%\uji2_%T%.mp4 --record-vis %O%\uji2_%T%_vis.mp4 --debug-readings %O%\bacaan_uji2_%T%
goto menu
:manual
%P% scripts\demo_daftar_kotak.py --inventaris %I%
set /p REQ=First request: type a NUMBER from the list (or the text): 
for /f "delims=" %%r in ('%P% scripts\demo_daftar_kotak.py --inventaris %I% --resolve "%REQ%"') do set REQT=%%r
echo Request: %REQT%   (next ones: type a number in this window while the camera runs; ? = list)
%P% scripts\test_webcam_box.py --config %C% --request "%REQT%" --inventaris %I% --expected 20 --record %O%\manual_%T%.mp4 --record-vis %O%\manual_%T%_vis.mp4 --debug-readings %O%\bacaan_manual_%T%
goto menu
:grasp
%P% scripts\demo_daftar_kotak.py --inventaris %I%
set /p REQ=First request: type a NUMBER from the list (or the text): 
for /f "delims=" %%r in ('%P% scripts\demo_daftar_kotak.py --inventaris %I% --resolve "%REQ%"') do set REQT=%%r
echo Request: %REQT%   (depth on: each PICK prints a [grasp] line with camera xyz and spine size in mm)
%P% scripts\test_webcam_box.py --config %C% --depth --request "%REQT%" --inventaris %I% --expected 20 --record %O%\grasp_%T%.mp4 --record-vis %O%\grasp_%T%_vis.mp4 --debug-readings %O%\bacaan_grasp_%T%
goto menu

:data
echo Two shelves: record shelf 1, then shelf 2, then both together. Name = shelf + session, e.g. r1_s1_far, r2_s2_mid, r12_s1_far
set /p SES=Session name, no spaces (e.g. r1_s1_far, r2_s2_mid, r12_s1_far): 
set NBOX=20
set /p NBOX=Boxes in view (Enter = 20): 
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
%P% scripts\test_webcam_box.py --config %C% --expected %NBOX% --record results\data_rak\%SES%.mp4 --record-vis results\data_rak\%SES%_vis.mp4
goto menu

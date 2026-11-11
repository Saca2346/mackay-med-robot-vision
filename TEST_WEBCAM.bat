@echo off
rem WEBCAM test on the LAPTOP (USB webcam instead of the RealSense D455). Same menu numbers as TEST_REALSENSE.bat;
rem the RealSense-only items (USB 3 check, rs_uji1 replays, depth / grasp 3D, shelf data for training) are left out.
rem Config: config\box_pipeline_config_webcam.yaml (model v4 1280, 1920x1080, camera "auto" = largest resolution).
rem New recordings and logs go to results\uji_webcam (O). In the camera window: q = stop, v = all boxes / target only,
rem a = drag a box around the shelf (boxes on a screen or wall behind the shelf are then ignored).
rem Tests 4, 7, 8, 9 use a fresh COPY of the inventory: taking boxes during a test never changes data\rak\inventaris.csv.
rem Unplug the RealSense while using the webcam: "auto" could otherwise open one of the RealSense streams.
cd /d "%~dp0"
set P=venv\Scripts\python.exe
set O=results\uji_webcam
if not exist %O% mkdir %O%
rem own config copy: event log, screens and pictures go to %O%
%P% scripts\sesi_config.py config\box_pipeline_config_webcam.yaml %O% >nul
set C=%O%\config_sesi.yaml
set I=%O%\inventaris_uji.csv
echo Looking for the webcam ...
set CAMW=0
for /f %%w in ('%P% -c "from src.box_camera import list_cameras as l; print(max([c[1] for c in l()] or [0]))" 2^>nul') do set CAMW=%%w
if %CAMW% LSS 1280 echo  WARNING: no USB webcam found - the largest camera is %CAMW% px wide, the laptop camera at best.
if %CAMW% LSS 1280 echo  Plug in the webcam, wait 5 s, then start again. Text on the boxes cannot be read at that size.
if %CAMW% LSS 1280 pause
:menu
for /f %%i in ('powershell -nologo -noprofile -command "Get-Date -Format yyyyMMdd_HHmmss"') do set T=%%i
echo.
echo ===== WEBCAM TEST =====
echo  1  List the cameras (number, resolution) - the largest one is used
echo  2  Detection only (adjust the camera: sharpness in the top line should be high)
echo  4  LOCK TARGET and record: type the request, e.g. accuforce,2.5,12
echo  8  LOCK TARGET, MANUAL: the numbered box list is shown; type a NUMBER (or the text) per request; ? = list again
echo  P  SHELF MAP for the target lock (do this first after boxes were moved): camera still in front of the whole shelf, 1-2 min, q = save
echo  0  Exit
set "PILIH="
set /p PILIH=Choose a number:
if "%PILIH%"=="4" copy /Y data\rak\inventaris.csv %I% >nul
if "%PILIH%"=="8" copy /Y data\rak\inventaris.csv %I% >nul
if "%PILIH%"=="1" %P% -c "from src.box_camera import list_cameras; print('cameras (number, width, height):', list_cameras())"
if "%PILIH%"=="2" %P% scripts\test_webcam_box.py --config %C% --expected 20
if "%PILIH%"=="4" goto lain
if "%PILIH%"=="8" goto manual
if /I "%PILIH%"=="P" %P% scripts\rak_inventaris.py peta --inventaris data\rak\inventaris.csv --config %C%
if "%PILIH%"=="0" exit /b 0
goto menu
:lain
set "REQ="
set /p REQ=Request, e.g. accuforce,2.5,12 :
%P% scripts\test_webcam_box.py --config %C% --lock-only --request "%REQ%" --inventaris %I% --expected 20 --record %O%\uji_%T%.mp4 --record-vis %O%\uji_%T%_vis.mp4 --debug-readings %O%\bacaan_uji_%T%
goto menu
:manual
set "REQ="
set "REQT="
%P% scripts\demo_daftar_kotak.py --inventaris %I%
set /p REQ=First request: type a NUMBER from the list (or the text):
for /f "delims=" %%r in ('%P% scripts\demo_daftar_kotak.py --inventaris %I% --resolve "%REQ%"') do set REQT=%%r
if "%REQT%"=="" goto menu
echo Request: %REQT%   (next ones: type a number in this window while the camera runs; ? = list)
%P% scripts\test_webcam_box.py --config %C% --lock-only --request "%REQT%" --inventaris %I% --expected 20 --record %O%\manual_%T%.mp4 --record-vis %O%\manual_%T%_vis.mp4 --debug-readings %O%\bacaan_manual_%T%
goto menu

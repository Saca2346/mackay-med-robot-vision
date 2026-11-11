@echo off
setlocal
cd /d "%~dp0"
rem Recorded live test with camera calibration: TEST_LIVE_CALIBRATED.bat webcam   or   TEST_LIVE_CALIBRATED.bat realsense
rem Add a second word "auto" (TEST_LIVE_CALIBRATED.bat webcam auto) for a hands-off run: no key presses, calibration measured automatically.
rem Everything on the screen, from this window to the final result, is recorded to results\live_<camera>_<time>\00_layar_penuh.mp4
set "CAM=%~1"
set "AUTO=%~2"
if "%CAM%"=="" set /p CAM=Camera (webcam or realsense):
if /I not "%CAM%"=="webcam" if /I not "%CAM%"=="realsense" goto bad_camera
set "P=venv\Scripts\python.exe"
set "CFG=config\box_pipeline_config_%CAM%_eleven.yaml"
for /f %%i in ('powershell -nologo -noprofile -command "Get-Date -Format yyyyMMdd_HHmmss"') do set T=%%i
set "SD=results\live_%CAM%_%T%"
mkdir "%SD%"
start "SCREEN RECORDER - do not close, it stops by itself" /min %P% scripts\rekam_layar.py --out "%SD%\00_layar_penuh.mp4" --stop-file "%SD%\STOP_REC"
echo.
echo ===== RECORDED LIVE TEST %CAM% %T% =====
echo Screen recording: %SD%\00_layar_penuh.mp4
echo.
echo [0/3] CHARGER: must say Online (battery makes timing 3 to 8 times slower)
powershell -nologo -noprofile -command "Add-Type -A System.Windows.Forms; 'Power line: ' + [System.Windows.Forms.SystemInformation]::PowerStatus.PowerLineStatus"
if /I not "%AUTO%"=="auto" pause
if /I "%AUTO%"=="auto" timeout /t 5 >nul
echo.
echo [1/3] CALIBRATION: move the camera until there is no FAIL, press Enter or q, then do not touch anything
if /I not "%AUTO%"=="auto" %P% scripts\kalibrasi_kamera.py --sesi live_%CAM%_%T% --config %CFG% --out "%SD%" --min-boxes 11
if /I "%AUTO%"=="auto" %P% scripts\kalibrasi_kamera.py --sesi live_%CAM%_%T% --config %CFG% --out "%SD%" --min-boxes 11 --no-window
echo Calibration exit code %errorlevel% (0 = pass or warn, 1 = fail)
if /I not "%AUTO%"=="auto" pause
if /I "%AUTO%"=="auto" timeout /t 3 >nul
echo.
echo [2/3] LIVE TEST: 11 requests, one after the other, 60 frames each; calibration is measured again and must pass
echo       The runner reads data\rak\inventaris.csv read-only and writes nothing to it.
%P% scripts\validate_box_targets.py --live --camera %CAM% --config %CFG% --calibrate-first --no-window --max-frames 60 --output "%SD%\batch"
set "LIVE_RC=%errorlevel%"
echo Live test exit code %LIVE_RC% (0 = finished, 1 = calibration failed or error)
echo.
if not "%LIVE_RC%"=="0" echo [3/3] LIVE TEST DID NOT FINISH: read the message above. No results table.
if "%LIVE_RC%"=="0" echo [3/3] DONE. Results table and review page:
if "%LIVE_RC%"=="0" echo   %SD%\batch\REPORT.md
if "%LIVE_RC%"=="0" echo   %SD%\batch\REVIEW.html
if /I not "%AUTO%"=="auto" echo Press any key to stop the screen recording.
if /I not "%AUTO%"=="auto" pause >nul
if /I "%AUTO%"=="auto" timeout /t 8 >nul
echo stop> "%SD%\STOP_REC"
timeout /t 8 >nul
echo Video: %SD%\00_layar_penuh.mp4
if /I not "%AUTO%"=="auto" pause
if /I "%AUTO%"=="auto" timeout /t 5 >nul
exit /b 0
:bad_camera
echo Camera must be webcam or realsense.
pause
exit /b 1

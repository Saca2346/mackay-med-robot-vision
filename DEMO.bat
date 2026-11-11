@echo off
rem DEMO launcher for an external USB webcam. Run it from cmd: DEMO.bat
rem Order for the demo:  1 (pre-flight)  ->  2 (text scenes)  ->  3 or 4 (camera)  ->  5 (only if the camera fails).
rem Each launch saves to results\demo_TIMESTAMP. Camera takes use a COPY of the inventory (inventaris_demo.csv):
rem taking a box in the demo never changes data\rak\inventaris.csv. Menu R resets the copy.
rem In the camera window: q = stop.
cd /d "%~dp0"
set "BASE=config\box_pipeline_config_webcam.yaml"
rem Menu 5 asks for a webcam recording; no historical evidence file is used by default.
set "P=venv\Scripts\python.exe"
for /f %%i in ('powershell -nologo -noprofile -command "Get-Date -Format yyyyMMdd_HHmmss_fff"') do set "T=%%i"
set "D=results\demo_%T%"
set "C=%D%\config_sesi.yaml"
set "I=%D%\inventaris_demo.csv"
%P% scripts\sesi_config.py "%BASE%" "%D%"
if not "%errorlevel%"=="0" goto startup_failed
copy /Y data\rak\inventaris.csv "%I%" >nul
if not "%errorlevel%"=="0" goto startup_failed
echo Demo outputs: %D%
echo Camera: external USB webcam, requested 1920x1080; highest-resolution camera selected automatically.
echo Close other camera apps and unplug unused cameras. Grasp output: pixels only.
:menu
echo.
echo ===== DEMO =====
echo  S  RECORDED VERIFIED TEST SESSION (screen recorded from now to the end): hardware check, verification, calibration,
echo     hardware test 1, random requests from the list, report. One folder per session. (No stress test.)
echo  1  Pre-flight check (model, inventory, RAM, USB webcam frame and resolution)  - run this first
echo  2  TEXT scenes, no camera: 7 requests with the answer sent to the TM software
echo  3  LOCK TARGET: choose a NUMBER or request; one full-box outline, no PICK/TAKEN wait
echo  4  LOCK TARGET, fixed request: ivascular angiolite 2.5 x 29
echo  5  Replay a recorded webcam video (choose the file; this is not a live test)
echo  V  VERIFIED PICK mode: two-source verification and physical take workflow
echo  L  LIST of the boxes: numbers, exact text to type, sizes, expiry, expected result
echo  R  Reset the demo inventory (all boxes back on the shelf)
echo  F  FOCUSED WEBCAM: YOLO26, two-source VERIFIED + crops, no stress test
echo  0  Exit
set "PILIH="
set /p PILIH=Choose a number:
if /I "%PILIH%"=="S" goto session
if /I "%PILIH%"=="V" goto verified
if "%PILIH%"=="1" %P% scripts\demo_preflight.py --config "%C%" --inventaris "%I%"
if "%PILIH%"=="2" goto scenes
if "%PILIH%"=="3" goto cam
if "%PILIH%"=="4" goto fixed
if "%PILIH%"=="5" goto replay
if /I "%PILIH%"=="L" %P% scripts\demo_daftar_kotak.py --inventaris %I%
if /I "%PILIH%"=="R" goto reset
if /I "%PILIH%"=="F" call FOCUS_WEBCAM.bat
if "%PILIH%"=="0" exit /b 0
goto menu

:scenes
echo.
echo --- Scene 1 of 7: exact size, one box (expect READY, K15) ---
%P% scripts\demo_request.py --inventaris %I% --log %D%\demo_log.jsonl --sekali "ivascular angiolite 2.5 x 29"
pause
echo --- Scene 2 of 7: same product, other size, must NOT be confused (expect READY, K10) ---
%P% scripts\demo_request.py --inventaris %I% --log %D%\demo_log.jsonl --sekali "ivascular angiolite 4 x 19"
pause
echo --- Scene 3 of 7: four identical boxes, FEFO picks the earliest expiry (expect K03, 2029-03-10) ---
%P% scripts\demo_request.py --inventaris %I% --log %D%\demo_log.jsonl --sekali "boston scientific permanent sled bag"
pause
echo --- Scene 4 of 7: not in stock (expect NOT FOUND) ---
%P% scripts\demo_request.py --inventaris %I% --log %D%\demo_log.jsonl --sekali "terumo accuforce 2.75 x 15"
pause
echo --- Scene 5 of 7: known product, size not in stock (expect NOT FOUND) ---
%P% scripts\demo_request.py --inventaris %I% --log %D%\demo_log.jsonl --sekali "xperience pro 3.0 x 18"
pause
echo --- Scene 6 of 7: incomplete request (expect REPEAT, asks for diameter AND length) ---
%P% scripts\demo_request.py --inventaris %I% --log %D%\demo_log.jsonl --sekali "essential pro 3"
pause
echo --- Scene 7 of 7: unknown product (expect REPEAT, lists the known products) ---
%P% scripts\demo_request.py --inventaris %I% --log %D%\demo_log.jsonl --sekali "medtronic resolute 3.0 x 18"
echo.
echo Log of every answer: %D%\demo_log.jsonl
goto menu

:cam
set "TARGETMODE=--lock-only"
goto cam_list
:verified
set "TARGETMODE="
:cam_list
%P% scripts\demo_daftar_kotak.py --inventaris %I%
:cam_ask
echo.
echo Type a NUMBER from the list (or brand + product + size).  ? = show the list again,  Enter = back to the menu
set REQ=
set /p REQ=Request:
if "%REQ%"=="" goto menu
if "%REQ%"=="?" goto cam_list
set R2=
for /f "usebackq delims=" %%r in (`%P% scripts\demo_daftar_kotak.py --inventaris %I% --resolve "%REQ%"`) do set R2=%%r
if "%R2%"=="" goto cam_ask
echo Request: %R2%
for /f %%i in ('powershell -nologo -noprofile -command "Get-Date -Format yyyyMMdd_HHmmss"') do set T=%%i
%P% scripts\demo_request.py --inventaris %I% --log %D%\demo_log.jsonl --sekali "%R2%" --kamera --kamera-arg %TARGETMODE% --config %C% --expected 20 --record %D%\cam_%T%.mp4 --record-vis %D%\cam_%T%_vis.mp4 --debug-readings %D%\bacaan_%T%
goto cam_ask

:fixed
rem The fixed scene starts from a fresh inventory copy, including after a verified test.
copy /Y data\rak\inventaris.csv %I% >nul
if not "%errorlevel%"=="0" goto inventory_failed
for /f %%i in ('powershell -nologo -noprofile -command "Get-Date -Format yyyyMMdd_HHmmss"') do set T=%%i
%P% scripts\demo_request.py --inventaris %I% --log %D%\demo_log.jsonl --sekali "ivascular angiolite 2.5 x 29" --kamera --kamera-arg --lock-only --config %C% --expected 20 --record %D%\scene_%T%.mp4 --record-vis %D%\scene_%T%_vis.mp4 --debug-readings %D%\bacaan_scene_%T%
goto menu

:replay
set "F="
set /p F=Webcam recording path (no surrounding quotes, Enter = back):
if "%F%"=="" goto menu
if not exist "%F%" goto replay_missing
for /f %%i in ('powershell -nologo -noprofile -command "Get-Date -Format yyyyMMdd_HHmmss"') do set T=%%i
echo Replaying the selected webcam recording (not live). Say so in the demo.
%P% scripts\test_webcam_box.py --config "%C%" --lock-only --inventaris "%I%" --source "%F%" --request "angiolite,2.5,29" --expected 20 --record-vis %D%\replay_%T%_vis.mp4 --debug-readings %D%\bacaan_replay_%T%
goto menu

:replay_missing
echo Replay unavailable: %F%
echo Choose an existing webcam recording, such as a cam_*.mp4 from a previous demo session.
echo Keep its .mp4.times.csv file beside it for recorded timing. No replay was started.
goto menu

:allboxes
rem every box of the plan in turn; a box that stays blue/yellow for 60 s: type  lewati  in THIS window; gives up after 180 s
copy /Y data\rak\inventaris.csv %I% >nul
if not "%errorlevel%"=="0" goto inventory_failed
%P% scripts\demo_daftar_kotak.py --inventaris "%I%" --rencana-out "%D%\rencana_20.txt"
if not "%errorlevel%"=="0" goto plan_failed
for /f %%i in ('powershell -nologo -noprofile -command "Get-Date -Format yyyyMMdd_HHmmss"') do set T=%%i
%P% scripts\test_webcam_box.py --config %C% --rencana "%D%\rencana_20.txt" --until-taken --waktu-per-barang 180 --inventaris %I% --expected 20 --log-stress %D%\hasil_semua_%T%.csv --record %D%\semua_%T%.mp4 --record-vis %D%\semua_%T%_vis.mp4 --debug-readings %D%\bacaan_semua_%T%
echo Results table: %D%\hasil_semua_%T%.csv   Video: %D%\semua_%T%_vis.mp4
echo Put every taken box back on the shelf as in your photo, then press R in this menu.
goto menu

:session
rem One folder per session; every program writes its output there (own config copy, own inventory copy).
for /f %%i in ('powershell -nologo -noprofile -command "Get-Date -Format yyyyMMdd_HHmmss_fff"') do set T=%%i
set "SD=%D%\sesi_%T%"
%P% scripts\sesi_config.py "%BASE%" "%SD%"
if not "%errorlevel%"=="0" goto session_failed
set "CS=%SD%\config_sesi.yaml"
set "IS=%SD%\inventaris_sesi.csv"
copy /Y data\rak\inventaris.csv %IS% >nul
if not "%errorlevel%"=="0" goto inventory_failed
set N=0
start "SCREEN RECORDER - do not close, it stops by itself" /min %P% scripts\rekam_layar.py --out %SD%\00_layar_penuh.mp4 --stop-file %SD%\STOP_REC
echo %date% %time% session %T% started, screen recording on>> %SD%\langkah.txt
echo.
echo ================= RECORDED TEST SESSION %T% =================
echo Everything on the screen is recorded to %SD%\00_layar_penuh.mp4
echo Steps: 1 hardware check, 2 verification, 3 calibration, 4 hardware test 1, 5 random requests
echo.
echo [1/5] HARDWARE CHECK: model, inventory, free RAM, USB webcam frame and resolution
%P% scripts\demo_preflight.py --config "%CS%" --inventaris "%IS%"
if not "%errorlevel%"=="0" goto s_hwfail
echo %time% 1 hardware check: READY>> %SD%\langkah.txt
goto s_verify
:s_hwfail
echo %time% 1 hardware check: NOT READY>> %SD%\langkah.txt
echo Hardware check FAILED (see the FAIL lines above). Fix it first, or continue anyway.
set GO=
set /p GO=Continue anyway? (y/n):
if /I not "%GO%"=="y" goto s_end
:s_verify
pause
echo.
echo [2/5] SOFTWARE VERIFICATION: automated tests (about 30 s)
%P% -m pytest tests -q --basetemp "%SD%\pytest_tmp" -p no:cacheprovider
set "CHECK_RC=%errorlevel%"
if not "%CHECK_RC%"=="0" echo %time% 2 verification: tests FAILED>> %SD%\langkah.txt
if "%CHECK_RC%"=="0" echo %time% 2 verification: all tests passed>> %SD%\langkah.txt
pause
echo.
echo [3/5] CAMERA CALIBRATION: move the camera until there is no FAIL, press q, then hold still 3 s
%P% scripts\kalibrasi_kamera.py --sesi sesi_%T% --config %CS% --out %SD%
set "CHECK_RC=%errorlevel%"
if not "%CHECK_RC%"=="0" echo %time% 3 calibration: FAIL or cancelled>> %SD%\langkah.txt
if "%CHECK_RC%"=="0" echo %time% 3 calibration: done>> %SD%\langkah.txt
pause
echo.
echo [4/5] HARDWARE TEST 1: detection only. The HUD must show boxes 20 and sharpness 300 or more.
echo       Hold the camera still about 10 s, then press q in the camera window.
%P% scripts\test_webcam_box.py --config %CS% --expected 20 --record-vis %SD%\04_hw_test1_vis.mp4
echo %time% 4 hardware test 1: done (video 04_hw_test1_vis.mp4)>> %SD%\langkah.txt
pause
echo.
rem 18-10: the stress test is no longer part of the session (owner); menu 6 still runs it.
echo.
echo [5/5] RANDOM REQUESTS: type a NUMBER from the list (or text). Enter on an empty line = end the session.
:s_list
%P% scripts\demo_daftar_kotak.py --inventaris %IS%
:s_rand
set REQ=
set /p REQ=Request (number or text, ? = list, Enter = finish):
if "%REQ%"=="" goto s_end
if "%REQ%"=="?" goto s_list
set R2=
for /f "usebackq delims=" %%r in (`%P% scripts\demo_daftar_kotak.py --inventaris %IS% --resolve "%REQ%"`) do set R2=%%r
if "%R2%"=="" goto s_rand
set /a N+=1
echo Request %N%: %R2%
%P% scripts\demo_request.py --inventaris %IS% --log %SD%\demo_log.jsonl --sekali "%R2%" --kamera --kamera-arg --config %CS% --expected 20 --record %SD%\06_acak_%N%.mp4 --record-vis %SD%\06_acak_%N%_vis.mp4 --debug-readings %SD%\06_acak_%N%_bacaan.csv
echo %time% 5 random request %N%: %R2%>> %SD%\langkah.txt
goto s_rand
:s_end
echo %time% session end>> %SD%\langkah.txt
type nul > %SD%\STOP_REC
echo Closing the screen recording ...
timeout /t 6 /nobreak >nul
%P% scripts\ringkas_sesi.py %SD%
echo.
echo SESSION DONE. Folder: %SD%   Report: %SD%\LAPORAN_SESI.md   Screen video: %SD%\00_layar_penuh.mp4
goto menu

:reset
copy /Y data\rak\inventaris.csv %I% >nul
if not "%errorlevel%"=="0" goto inventory_failed
echo Demo inventory reset from data\rak\inventaris.csv
goto menu

:s_plan_failed
echo Could not create the session request plan. Ending the session.
goto s_end

:s_inventory_failed
echo Could not reset the session inventory copy. Ending the session.
goto s_end

:plan_failed
echo Could not create the request plan. Camera was not started.
goto menu

:inventory_failed
echo Could not create or reset the inventory copy. Camera was not started.
goto menu

:session_failed
echo Could not prepare the session config. Session was not started.
goto menu

:startup_failed
echo Could not prepare the demo config or inventory copy. Demo was not started.
exit /b 1

@echo off
rem Task 1-3 demo (meeting 15 Oct 2026) in ONE cmd window, from start to end - good for a screen recording.
rem Input = medicine data (text), output = ONE bounding box on the target box.
rem   Task 1  "Braun, DCB, 2.75mm, 30mm"  photo of 13 Oct (results\task_20261015\input\task1_braun.jpg)
rem   Task 2  "Chroma, 2.75 , 33"         photo of 13 Oct (results\task_20261015\input\task2_chroma.jpg)
rem   Task 3  "Accuforce, 2.5, 12"        LIVE shelf: choose RealSense D455, webcam (USB) or the shelf recording
rem Photo window: v = all boxes / target only, q = next task. Camera window: q = stop.
rem Every run writes into its own folder results\task_live_DATE; the evidence folders are only read.
cd /d "%~dp0"
set P=venv\Scripts\python.exe
set IN=results\task_20261015\input
set B=results\bukti_laporan_20261008
set REC=results\task_20261015\task3\shelf_accuforce_k07.mp4
rem Task 1-2: model v4 fine-tuned 15-10 with the two task photos, their TARGET boxes hidden from training
rem (results\task_20261015\training_data). The live demo (task 3) stays on v4 until the full training check.
set NB=--weights models\box_v4s_nb1015_960_openvino_model --imgsz 544,960
for /f %%i in ('powershell -nologo -noprofile -command "Get-Date -Format yyyyMMdd_HHmmss"') do set T=%%i
set O=results\task_live_%T%
mkdir %O%
mkdir %O%\task3
cls
echo ==================================================================
echo  VISION DEMO - medicine box robot
echo  Input: medicine data (text)   Output: ONE bounding box on the target
echo  Model: YOLO11s-OBB v4 + OCR + REF / barcode check
echo  Green = verified (2 sources, the robot may pick)   Blue = candidate (1 source)
echo  Results: %O%
echo ==================================================================
set BAT=
for /f %%b in ('powershell -nologo -noprofile -command "(Get-CimInstance Win32_Battery).BatteryStatus"') do set BAT=%%b
if not "%BAT%"=="1" goto intro_done
echo.
echo  WARNING: the laptop runs on BATTERY. Plug in the charger first:
echo  on battery the same task was 8x slower (94 s instead of 11 s).
echo.
:intro_done
pause

:task1
cls
echo ==================================================================
echo  TASK 1   input: "Braun, DCB, 2.75mm, 30mm"
echo           photo: B. Braun SeQuent Please NEO, 8 boxes (new brand)
echo  Window:  v = all boxes / target only     q = next task
echo ==================================================================
%P% scripts\demo_foto_target.py %IN%\task1_braun.jpg --request "Braun, DCB, 2.75mm, 30mm" --out %O%\task1 --show --view-width 560 %NB%
echo.
echo  TASK 1 done. Result picture: %O%\task1\screens
pause

:task2
cls
echo ==================================================================
echo  TASK 2   input: "Chroma, 2.75 , 33"
echo           photo: Biosensors CHROMA shelf, 31 boxes (new brand)
echo  Window:  v = all boxes / target only     q = next task
echo ==================================================================
%P% scripts\demo_foto_target.py %IN%\task2_chroma.jpg --request "Chroma, 2.75 , 33" --out %O%\task2 --show --view-width 900 %NB%
echo.
echo  TASK 2 done. Result picture: %O%\task2\screens
pause

:task3
cls
echo ==================================================================
echo  TASK 3   input: "Accuforce, 2.5, 12"   real shelf, 20 boxes
echo ==================================================================
copy /Y data\rak\inventaris.csv %O%\task3\inventaris_copy.csv >nul
set CAM=0
for /f %%c in ('%P% -c "import pyrealsense2 as rs; print(len(rs.context().query_devices()))" 2^>nul') do set CAM=%%c
set CAMSEL=2
rem Current default: external USB webcam, even when a RealSense is also connected.
echo  Camera:  1 = RealSense D455    2 = webcam (USB)    3 = shelf recording, no camera
if not "%CAM%"=="0" echo  A RealSense is connected.
set /p CAMSEL=Choose 1, 2 or 3 (Enter = %CAMSEL%):
if "%CAMSEL%"=="1" goto task3_rs
if "%CAMSEL%"=="2" goto task3_webcam
goto task3_replay

:task3_rs
if "%CAM%"=="0" echo  No RealSense found - check the USB 3 cable. Using the shelf recording instead.
if "%CAM%"=="0" goto task3_replay
%P% scripts\sesi_config.py config\box_pipeline_config_realsense.yaml %O%\task3 >nul
set CAMNAME=realsense
goto task3_live

:task3_webcam
%P% scripts\sesi_config.py config\box_pipeline_config_webcam.yaml %O%\task3 >nul
set CAMNAME=webcam
if not "%CAM%"=="0" echo  NOTE: a RealSense is also connected. Unplug it, otherwise the webcam search may open a RealSense stream.
echo  Webcam: the camera with the largest resolution is used (1920x1080 when the webcam allows it).
set CAMW=0
for /f %%w in ('%P% -c "from src.box_camera import list_cameras as l; print(max([c[1] for c in l()] or [0]))" 2^>nul') do set CAMW=%%w
if %CAMW% LSS 1280 echo  WARNING: no USB webcam found - the largest camera is %CAMW% px wide, the laptop camera at best.
if %CAMW% LSS 1280 echo  Plug in the webcam, wait 5 s, then start again. Text on the boxes cannot be read at that size.
if %CAMW% LSS 1280 pause

:task3_live
echo  Visual target lock: read the requested box, then track it with OCR paused. Timing depends on the view.
set MAP=n
set /p MAP=Map the shelf now (camera still on the whole shelf, about 1-2 min, q = save)? (y/n, Enter = n): 
if /I "%MAP%"=="y" %P% scripts\rak_inventaris.py peta --inventaris data\rak\inventaris.csv --config %O%\task3\config_sesi.yaml
if /I "%MAP%"=="y" copy /Y data\rak\inventaris.csv %O%\task3\inventaris_copy.csv >nul
echo  LIVE test with the %CAMNAME%. One target outline; CHECKING until its text is read consistently.
echo  If boxes are found outside the shelf (screen, wall): press a and drag a box around the shelf.
set REQ3=Accuforce, 2.5, 12
set /p REQ3=Request (Enter = Accuforce, 2.5, 12):
echo  Request: %REQ3%   - press q to close the visual target lock (no PICK authorization)
%P% scripts\test_webcam_box.py --config %O%\task3\config_sesi.yaml --lock-only --request "%REQ3%" --inventaris %O%\task3\inventaris_copy.csv --expected 20 --record %O%\task3\task3_%CAMNAME%.mp4 --record-vis %O%\task3\task3_%CAMNAME%_vis.mp4
goto summary

:task3_replay
%P% scripts\sesi_config.py config\box_pipeline_config_realsense.yaml %O%\task3 >nul
echo  Task 3 runs on the D455 shelf recording of 15 Oct
echo  (same program, same rules, the video plays like a live camera).
echo  Press q in the camera window when the box is green (PICK); the video ends by itself after 2 min.
%P% scripts\test_webcam_box.py --config %O%\task3\config_sesi.yaml --source %REC% --request "Accuforce, 2.5, 12" --inventaris %O%\task3\inventaris_copy.csv --expected 20 --record-vis %O%\task3\task3_replay_vis.mp4

:summary
echo.
echo ==================================================================
echo  DEMO DONE. Results in %O%
echo    task1\screens, task2\screens : result pictures
echo    task3 live : target_view.jsonl (visual lock, no PICK), *_vis.mp4
echo    task3 replay : verify_log.csv, grasp.jsonl, *_vis.mp4 (verified workflow)
echo ==================================================================
start "" explorer %O%
pause

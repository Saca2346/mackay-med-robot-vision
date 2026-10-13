@echo off
rem Live test of one box on the LAPTOP (RealSense directly, no server / tunnel). Evidence goes to results\bukti_laporan_20261008.
rem Usage:  TEST_LIVE_LOCAL.bat 5      (full shelf, 20 boxes)
rem         TEST_LIVE_LOCAL.bat 5 0    (not on the shelf / few boxes: the box count is not checked)
rem         test numbers 1-16 in docs\CHECKLIST_DEMO.md; 17 only while K10 is on the shelf; q = stop
rem Saves: uji_live_NN.mp4 (raw camera), uji_live_NN_vis.mp4 (ANNOTATED: boxes, labels, status), bacaan_live_NN.csv
cd /d "%~dp0"
if "%1"=="" (
  echo usage: TEST_LIVE_LOCAL.bat NUMBER [BOXES]   ^(NUMBER = 1 to 17^)
  exit /b 1
)
set N=%1
if %N% LSS 10 set N=0%N%
set E=%2
if "%E%"=="" set E=20
set B=results\bukti_laporan_20261008
venv\Scripts\python.exe scripts\uji_checklist.py --kamera %1 --kamera-arg --config %B%\config_uji.yaml --expected %E% --record %B%\uji_live_%N%.mp4 --record-vis %B%\uji_live_%N%_vis.mp4 --debug-readings %B%\bacaan_live_%N%.csv

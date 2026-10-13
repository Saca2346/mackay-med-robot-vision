@echo off
rem Laptop: send the webcam to the GPU server for the 16-box camera test (run it in the code folder).
rem Opens 2 windows: (1) webcam stream, (2) SSH tunnel (asks the server password). Do NOT close them during the test.
cd /d "%~dp0"
title Camera test - send to server
echo.
echo  BEFORE YOU START:
echo   1. Webcam mounted STILL (tripod / clamp) facing the shelf, boxes on the shelf
echo   2. Close EVERY program that uses the webcam (only one program can use the camera)
echo   3. Open a terminal on the server (VS Code Remote-SSH, see docs\REMOTE_SERVER_CLAUDE_CODEX.md), then follow
echo      docs\PANDUAN_UJI_KAMERA.md step 4
echo.
pause
if not exist venv\Scripts\python.exe goto novenv
start "1 - WEBCAM STREAM (do not close)" cmd /k venv\Scripts\python.exe scripts\stream_webcam.py
echo  Window 1 is open. Wait until it says the stream is active / shows an fps number, then...
pause
start "2 - SSH TUNNEL (type the server password, do not close)" cmd /k ssh -N -R 18090:127.0.0.1:8090 Zen@140.113.149.94
echo.
echo  Window 2 is open: type the server password, then leave it (no text = normal).
echo  Now run in a server terminal:
echo      cd ~/Downloads/mackay-med-robot-vision ^&^& bash scripts/jalankan_uji_kamera_server.sh
echo.
pause
exit /b 0
:novenv
echo FAILED: venv\Scripts\python.exe not found. Run this from the code folder.
pause
exit /b 1

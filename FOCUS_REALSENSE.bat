@echo off
setlocal
cd /d "%~dp0"
venv\Scripts\python.exe scripts\run_target.py --camera realsense --interactive %*
exit /b %errorlevel%

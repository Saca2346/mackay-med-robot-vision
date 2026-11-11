@echo off
setlocal
cd /d "%~dp0"
venv\Scripts\python.exe scripts\run_target.py --camera webcam --interactive %*
exit /b %errorlevel%

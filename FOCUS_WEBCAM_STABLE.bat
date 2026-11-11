@echo off
setlocal
cd /d "%~dp0"
venv\Scripts\python.exe scripts\run_target.py --camera webcam --config config\box_pipeline_config_focus_stable.yaml --interactive %*
exit /b %errorlevel%

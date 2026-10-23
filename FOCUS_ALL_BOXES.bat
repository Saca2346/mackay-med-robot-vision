@echo off
setlocal
cd /d "%~dp0"
venv\Scripts\python.exe scripts\run_target.py --camera webcam --config config\box_pipeline_config_all_targets.yaml --interactive %*
exit /b %errorlevel%

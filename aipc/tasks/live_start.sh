#!/bin/bash
# 10 Nov 2026: slide-1 recording. Runs in an xterm on a virtual screen of the AI PC (Xvfb), recorded with ffmpeg
# x11grab at real speed. Each command is printed after a prompt line, then run for real.
say() { printf '\n\033[1;32mZen@%s\033[0m:\033[1;34m%s\033[0m$ \033[1m%s\033[0m\n' "$(hostname)" "$(pwd | sed "s|$HOME|~|")" "$*"; sleep 1.2; }
note() { printf '\n\033[1;33m# %s\033[0m\n' "$*"; sleep 1.2; }

note "AI PC: machine, GPU and CPU"
say hostname; hostname
say nvidia-smi --query-gpu=name,memory.total,utilization.gpu --format=csv
nvidia-smi --query-gpu=name,memory.total,utilization.gpu --format=csv
say 'lscpu | grep "Model name"'; lscpu | grep "Model name"
note "The program folder"
say cd ~/Downloads/claude_offline_20261101/aipc_20261104/tasks
cd ~/Downloads/claude_offline_20261101/aipc_20261104/tasks
say ls aipc_main.py aipc_tasks.py live_window.py ../../models/box_y26_maxB.pt
ls aipc_main.py aipc_tasks.py live_window.py ../../models/box_y26_maxB.pt
note "Start: the model runs live in the window on the right (final flags of 10 Nov)"
say export OCR_ALGO=HEURISTIC
export OCR_ALGO=HEURISTIC
CMD='PYTHONPATH=../pylib ~/trse_venv/bin/python live_window.py 2>&1 | tee live1110.log | grep --line-buffered -av "\[INFO\]\|Qt\|QStandardPaths\|QFont"'
say "$CMD"
eval "$CMD"
note "Done"
sleep 2

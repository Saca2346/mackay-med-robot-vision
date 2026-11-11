#!/bin/bash
# Leo's Drive product photos (38 + demo samples), one box per photo (aipc_tasks.py drive), final reader (6 Nov).
# Runs after the live replays (beside the OCR comparison, whose server variants read on the CPU); the GPU is shared with the detector training, so the PICK times here are under load.
cd "$(dirname "$0")"
P=~/trse_venv/bin/python
export PYTHONPATH=../pylib
while pgrep -f "[l]ive_runs.sh" > /dev/null; do sleep 60; done
F="--imgsz 544x960 --workers 16 --procs 8 --proc-threads 2 --cls-cpu 1 --order skim --snap 1 --refine 1 --refine-device cuda --variants 1 --lazy 1 --ordered-prefetch 1 --priority 1"
$P aipc_main.py drive $F --tag drive > drive_last.log 2>&1
grep -a "^{" drive_last.log | cut -c1-400; grep -a "Error" drive_last.log | tail -1 | cut -c1-200
echo DRIVE_DONE

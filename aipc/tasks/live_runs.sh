#!/bin/bash
# The live webcam runs of 30 Oct (20 boxes in view) and 1 Nov (12 in view, 10 asked; normal and close camera) replayed
# on the AI PC with the final reader (6 Nov), first camera still of each take (as Task 3). Accuracy run after the OCR
# comparison (shared GPU, training running: these times are not the reader's; the quiet timing is in final_quiet.sh).
cd "$(dirname "$0")"
P=~/trse_venv/bin/python
export PYTHONPATH=../pylib
while pgrep -f "[o]cr_compare.sh" > /dev/null; do sleep 60; done
F="--imgsz 544x960 --workers 16 --procs 8 --proc-threads 2 --cls-cpu 1 --order skim --snap 1 --refine 1 --refine-device cuda --variants 1 --lazy 1 --ordered-prefetch 1 --priority 1"
$P aipc_main.py live ../livesets/box20 ../livesets/box12 ../livesets/box10close $F --tag live_acc > live_last.log 2>&1
grep -a "^{" live_last.log | cut -c1-500; grep -a "Error" live_last.log | tail -1 | cut -c1-200
echo LIVE_DONE

#!/bin/bash
# --lazy 1 must give the same cell for every pair as --lazy 0 (same detector, same OCR device): A/B on the task photos
cd "$(dirname "$0")"
P=~/trse_venv/bin/python
export PYTHONPATH=../pylib
C="--imgsz 544x960 --workers 27 --procs 20 --proc-device cpu --proc-threads 1 --variants 1"
for L in 0 1; do
  $P aipc_main.py score t1_drive t2_drive t1_orig $C --lazy $L --tag equiv_lazy$L 2>&1 | grep -a "^{" | cut -c1-300
  $P aipc_main.py task3 $C --lazy $L --claude-slots 1 --tag equiv_lazy$L 2>&1 | grep -a "^{" | cut -c1-300
done
echo EQUIV_DONE

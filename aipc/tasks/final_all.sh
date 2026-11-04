#!/bin/bash
# The final runs in one sequence (4 Nov 2026, late): nothing else of this work runs at the same time.
# 1. wait until the shared GPU is quiet (utilisation < 30% for 1 min)
# 2. timing of the final reader (four outlines, lazy, edge-corrected outline), events logged for the evidence videos
# 3. Task 3 with 24 OCR processes and with 8 reader threads; 4. one outline per box, for the cost of the variants
# 5. experiment: barcode decoding on (accuracy only, OCR on CPU processes)
# GPU load logged for the whole run (final_all_gpu.log). Training queue released at the end (det/PAUSE removed).
cd "$(dirname "$0")"
P=~/trse_venv/bin/python
export PYTHONPATH=../pylib
quiet=0
while [ $quiet -lt 6 ]; do
  u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d " ")
  if [ "$u" -lt 30 ]; then quiet=$((quiet+1)); else quiet=0; fi
  sleep 10
done
echo "GPU quiet $(date)"
(while true; do echo $(date +%T) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader) $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .); sleep 5; done) > final_all_gpu.log 2>&1 &
L=$!
PROCS="--workers 27 --procs 24 --cls-cpu 1 --proc-threads 1 --order skim"
SAME="--imgsz 544x960 --variants 1 --lazy 1 --snap 1"
run() { echo "== $(date +%T) $*"; $P aipc_main.py "$@" > final_last.log 2>&1; grep -a "^{" final_last.log | cut -c1-600; grep -a "Error" final_last.log | tail -1 | cut -c1-200; }
run time t1_drive t2_drive t1_orig --runs 5 $SAME $PROCS --tag fin
run task3 $SAME $PROCS --claude-slots 1 --tag fin_t3procs
run task3 $SAME --workers 8 --claude-slots 1 --tag fin_t3threads
run time t1_drive t2_drive t1_orig --runs 5 --imgsz 544x960 --variants 0 --snap 1 $PROCS --tag fin_novar
kill $L
CPU="--imgsz 544x960 --variants 1 --lazy 1 --workers 27 --procs 20 --proc-device cpu --proc-threads 1"
run score t1_drive t2_drive t1_orig $CPU --barcode 1 --tag barcode1
run task3 $CPU --barcode 1 --claude-slots 1 --tag barcode1
rm -f ../det/PAUSE
echo "FINAL_ALL_DONE $(date)"

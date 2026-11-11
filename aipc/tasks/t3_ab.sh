#!/bin/bash
# Task 3 speed A/B (5 Nov 2026, 01:00, GPU shared with other users, load logged): all four outlines prefetched for
# boxes whose skim shows both requested numbers (2, as before) or at least one (1). Order only; decisions compared.
cd "$(dirname "$0")"
P=~/trse_venv/bin/python
export PYTHONPATH=../pylib
(while true; do echo $(date +%T) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader) $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .); sleep 2; done) > t3ab_gpu.log 2>&1 &
L=$!
C="--imgsz 544x960 --workers 27 --cls-cpu 1 --proc-threads 1 --order skim --snap 1 --procs 20 --variants 1 --lazy 1 --ordered-prefetch 1 --claude-slots 1"
for r in 1 2; do
  for e in 2 1; do
    echo "== $(date +%T) eager$e r$r"
    $P aipc_main.py task3 $C --eager-hits $e --tag t3ab_e${e}_r$r > t3ab_last.log 2>&1
    grep -a "Error" t3ab_last.log | tail -1 | cut -c1-120
  done
done
kill $L
echo T3AB_DONE

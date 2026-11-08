#!/bin/bash
# The learned edge refiner in the final reader (6 Nov 2026): timing with --refine 1 (the PICK outline is placed by the
# refiner, inside the PICK time; decisions unchanged, the reads use the detector outlines), then deck pictures with
# zooms of the PICKed box (red detector, blue old snap, magenta refiner). Quiet-GPU gate and load log as stages.sh.
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
(while true; do echo $(date +%T) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader) $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .); sleep 5; done) > refine_gpu.log 2>&1 &
L=$!
F="--imgsz 544x960 --workers 27 --cls-cpu 1 --proc-threads 1 --order skim --snap 1 --refine 1 --variants 1 --lazy 1 --ordered-prefetch 1 --priority 1"
run() { echo "== $(date +%T) $*"; $P aipc_main.py "$@" > refine_last.log 2>&1; grep -a "^{\|demo.jpg" refine_last.log | cut -c1-300; grep -a "Error" refine_last.log | tail -1 | cut -c1-200; }
run time t1_drive t2_drive t1_orig --runs 5 $F --procs 24 --tag refine
run task3 $F --procs 20 --claude-slots 1 --tag refine_t3
kill $L
run demo t1_drive t2_drive task3:K01 task3:K04 task3:K10 task3:K21 $F --procs 20 --tag refine_demo
echo "REFINE_RUNS_DONE $(date)"

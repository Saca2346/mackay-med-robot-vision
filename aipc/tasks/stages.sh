#!/bin/bash
# Stage times of the final reader (6 Nov 2026): photo read, detection, crops (first outline; the other 3 lazily),
# skim, each box's verification, PICK, edge-corrected outline, all boxes read. Same flags as the final reader; waits
# for a quiet shared GPU (utilisation < 30% for 1 min) and logs its load. Output: out/stages*/.
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
(while true; do echo $(date +%T) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader) $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .); sleep 5; done) > stages_gpu.log 2>&1 &
L=$!
F="--imgsz 544x960 --workers 27 --cls-cpu 1 --proc-threads 1 --order skim --snap 1 --variants 1 --lazy 1 --ordered-prefetch 1 --priority 1"
run() { echo "== $(date +%T) $*"; $P aipc_main.py "$@" > stages_last.log 2>&1; grep -a "^{" stages_last.log | cut -c1-300; grep -a "Error" stages_last.log | tail -1 | cut -c1-200; }
run time t1_drive t2_drive t1_orig --runs 5 $F --procs 24 --tag stages
run task3 $F --procs 20 --claude-slots 1 --tag stages_t3
kill $L
echo "STAGES_DONE $(date)"

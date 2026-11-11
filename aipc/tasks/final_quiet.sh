#!/bin/bash
# Final timing of 6 Nov, after the detector training (queue5.sh) and the OCR comparison (ocr_compare.sh) are done and
# the shared GPU is quiet (utilisation < 30% for 1 min), GPU load logged. Final reader: four outlines, lazy, ordered
# prefetch, priority queue, the PICKed box drawn from a 1280 px detection and the learned refiner. Task 1 (Drive photo
# and the WhatsApp copy), Task 2, Task 3 (41 requests), the live runs with 20 / 12 / 10 asked boxes, Leo's Drive photos,
# deck pictures (green box only).
cd "$(dirname "$0")"
P=~/trse_venv/bin/python
export PYTHONPATH=../pylib
while pgrep -f "[q]ueue5.sh|[o]cr_compare.sh" > /dev/null; do sleep 120; done
touch ../det/PAUSE
quiet=0
while [ $quiet -lt 6 ]; do
  u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d " ")
  if [ "$u" -lt 30 ]; then quiet=$((quiet+1)); else quiet=0; fi
  sleep 10
done
echo "GPU quiet $(date)"
(while true; do echo $(date +%T) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader) $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .); sleep 5; done) > final_quiet_gpu.log 2>&1 &
L=$!
F="--imgsz 544x960 --workers 27 --cls-cpu 1 --proc-threads 1 --order skim --snap 1 --refine 1 --refine-device cuda --draw-imgsz 1280 --variants 1 --lazy 1 --ordered-prefetch 1 --priority 1"
run() { echo "== $(date +%T) $*"; $P aipc_main.py "$@" > final_quiet_last.log 2>&1; grep -a "^{" final_quiet_last.log | cut -c1-400; grep -a "Error" final_quiet_last.log | tail -1 | cut -c1-200; }
run time t1_drive t1_orig t2_drive --runs 5 $F --procs 24 --tag fq
run task3 $F --procs 20 --claude-slots 1 --tag fq_t3
run live ../livesets/box20 ../livesets/box12 ../livesets/box10close $F --procs 20 --tag fq_live
run drive $F --procs 20 --tag fq_drive
kill $L
run demo t1_drive t1_orig t2_drive task3:K01 task3:K04 task3:K10 task3:K21 $F --procs 20 --tag fq_demo
rm -f ../det/PAUSE
echo "FINAL_QUIET_DONE $(date)"

#!/bin/bash
# Cost of the edge refiner inside the PICK time (6 Nov 2026): the final reader with the old snap vs with the refiner
# on the GPU, interleaved in 2 rounds, quiet-GPU gate, load logged. Decisions are the same by construction (drawing only).
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
(while true; do echo $(date +%T) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader) $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .); sleep 5; done) > refine_ab_gpu.log 2>&1 &
L=$!
F="--imgsz 544x960 --workers 27 --cls-cpu 1 --proc-threads 1 --order skim --snap 1 --variants 1 --lazy 1 --ordered-prefetch 1 --priority 1 --procs 24"
for r in 1 2; do
  $P aipc_main.py time t1_drive t2_drive --runs 5 $F --tag rab_snap_r$r > refine_ab_last.log 2>&1; grep -a Error refine_ab_last.log | tail -1
  $P aipc_main.py time t1_drive t2_drive --runs 5 $F --refine 1 --refine-device cuda --tag rab_refgpu_r$r > refine_ab_last.log 2>&1; grep -a Error refine_ab_last.log | tail -1
done
kill $L
echo "REFINE_AB_DONE $(date)"

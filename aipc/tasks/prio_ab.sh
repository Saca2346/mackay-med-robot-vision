#!/bin/bash
# Priority dispatcher A/B (5 Nov 2026, ~01:10): GPU shared with other users as it is (option 2), load logged every 2 s,
# configurations interleaved. Final reader: four outlines, lazy, ordered prefetch; with and without --priority 1.
cd "$(dirname "$0")"
P=~/trse_venv/bin/python
export PYTHONPATH=../pylib
(while true; do echo $(date +%T) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader) $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .); sleep 2; done) > prio_gpu.log 2>&1 &
L=$!
C="--imgsz 544x960 --workers 27 --cls-cpu 1 --proc-threads 1 --order skim --snap 1 --variants 1 --lazy 1 --ordered-prefetch 1"
for r in 1 2; do
  for p in 0 1; do
    echo "== $(date +%T) time prio$p r$r"
    $P aipc_main.py time t1_drive t2_drive --runs 3 $C --procs 24 --priority $p --tag pab_p${p}_r$r > pab_last.log 2>&1
    grep -a "Error" pab_last.log | tail -1 | cut -c1-120
    echo "== $(date +%T) task3 prio$p r$r"
    $P aipc_main.py task3 $C --procs 20 --priority $p --claude-slots 1 --tag pab3_p${p}_r$r > pab_last.log 2>&1
    grep -a "Error" pab_last.log | tail -1 | cut -c1-120
  done
done
kill $L
echo PRIO_DONE

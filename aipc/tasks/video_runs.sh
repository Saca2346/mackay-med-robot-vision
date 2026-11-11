#!/bin/bash
# Runs for the evidence videos: final configuration, events logged (aipc_tasks.py stores when each box's read ended),
# GPU load logged alongside.   bash video_runs.sh TAG PROCS
cd "$(dirname "$0")"
P=~/trse_venv/bin/python
export PYTHONPATH=../pylib
TAG=${1:-video_busy}; N=${2:-24}
C="--imgsz 544x960 --variants 1 --lazy 1 --snap 1 --workers 27 --procs $N --cls-cpu 1 --proc-threads 1 --order skim"
(while true; do echo $(date +%T) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader); sleep 5; done) > ${TAG}_gpu.log 2>&1 &
L=$!
$P aipc_main.py time t1_orig t1_drive t2_drive --runs 5 $C --tag $TAG > ${TAG}_time.log 2>&1
$P aipc_main.py task3 $C --claude-slots 1 --tag $TAG > ${TAG}_t3.log 2>&1
kill $L
echo VIDEO_RUNS_DONE

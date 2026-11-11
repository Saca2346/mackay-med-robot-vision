#!/bin/bash
# Final run (4 Nov 2026), GPU free of training (det/PAUSE): one configuration for every number of the deck.
#   bash final.sh [WEIGHTS] [IMGSZ] [TAG]        default: model B at 544x960, tag final_B
# --lazy 1: identical cells to --lazy 0 on all 814 task-photo pairs and the 41 Task 3 requests (equiv.sh), only faster
# 1. time: the mentor's requests, photo on disk -> PICK, 5 runs after a warm-up (Drive photos + WhatsApp original)
# 2. confusion matrices: equiv.sh (same detector, outlines and rule; identical on CPU and GPU OCR)
# 3. task3: the 41 requests of 29 Oct, OCR in 24 GPU processes (A) and in 8 threads (B), the faster one is reported
#    with both shown
# 4. the same timing with one outline per box (no variants), for the cost of the variants
cd "$(dirname "$0")"
P=~/trse_venv/bin/python
export PYTHONPATH=../pylib
W=${1:-}; SZ=${2:-544x960}; TAG=${3:-final_B}
DET=""; [ -n "$W" ] && DET="--det $W"
PROCS="--workers 27 --procs 24 --cls-cpu 1 --proc-threads 1 --order skim"
SAME="$DET --imgsz $SZ --variants 1 --lazy 1 --snap 1"
run() { echo "== $*"; $P aipc_main.py "$@" > final_last.log 2>&1; grep -a "^{" final_last.log | cut -c1-900; grep -a "Error" final_last.log | tail -2; }
run time t1_drive t2_drive t1_orig --runs 5 $SAME $PROCS --tag $TAG
run task3 $SAME $PROCS --claude-slots 1 --tag ${TAG}_t3procs
run task3 $SAME --workers 8 --claude-slots 1 --tag ${TAG}_t3threads
run time t1_drive t2_drive t1_orig --runs 5 $DET --imgsz $SZ --variants 0 --snap 1 $PROCS --tag ${TAG}_novar
echo FINAL_DONE

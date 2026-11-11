#!/bin/bash
# Closing runs (5 Nov 2026): the final reader on the WhatsApp copy (timing, events logged), then the equivalence of
# tonight's scheduling changes (lazy crops, ordered prefetch, priority queue) on every task-photo pair, OCR on CPU.
cd "$(dirname "$0")"
P=~/trse_venv/bin/python
export PYTHONPATH=../pylib
F="--imgsz 544x960 --workers 27 --cls-cpu 1 --proc-threads 1 --order skim --snap 1 --variants 1 --lazy 1 --ordered-prefetch 1 --priority 1"
$P aipc_main.py time t1_orig --runs 3 $F --procs 24 --tag final_orig > close_last.log 2>&1; grep -a Error close_last.log | tail -1
$P aipc_main.py score t1_drive t2_drive t1_orig --imgsz 544x960 --workers 27 --procs 20 --proc-device cpu --proc-threads 1 --variants 1 --lazy 1 --ordered-prefetch 1 --priority 1 --tag equiv_final > close_last.log 2>&1; grep -a Error close_last.log | tail -1
echo CLOSE_DONE

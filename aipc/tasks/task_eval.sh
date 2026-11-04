#!/bin/bash
# Each detector candidate on the TASK photos (accuracy only; OCR on CPU processes while the GPU trains): Tasks 1-2
# confusion (Drive photos + WhatsApp original) and the 41 Task 3 requests, outline variants on, full slot truth.
cd "$(dirname "$0")"
P=~/trse_venv/bin/python
export PYTHONPATH=../pylib
D=../det/runs
for spec in "$@"; do
  set -- $(echo $spec | tr ":" " ")
  name=$1; w=$2; sz=$3
  [ -f "out/task_$name/task3.json" ] && continue
  $P aipc_main.py score t1_drive t2_drive t1_orig --det $w --imgsz $sz --workers 27 --procs 20 --proc-device cpu --proc-threads 1 --variants 1 --tag task_$name 2>&1 | grep -a "^{" | cut -c1-400
  $P aipc_main.py task3 --det $w --imgsz $sz --workers 27 --procs 20 --proc-device cpu --proc-threads 1 --variants 1 --claude-slots 1 --tag task_$name 2>&1 | grep -a "^{" | cut -c1-600
done
echo TASK_EVAL_DONE

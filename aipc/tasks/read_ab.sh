#!/bin/bash
# 7 Nov 2026: reading options on every task-photo pair and the 41 Task 3 requests (accuracy, cells incl. FP):
# baseline vs --split-sizes 1 (joined sizes "3.533") vs + --spine-size 1 (printed size on a spine without a size label).
cd "$(dirname "$0")"
P=~/trse_venv/bin/python
export PYTHONPATH=../pylib
F="--imgsz 544x960 --workers 27 --cls-cpu 1 --proc-threads 1 --order skim --variants 1 --lazy 1 --ordered-prefetch 1 --priority 1 --procs 20"
run() { echo "== $(date +%T) $*"; $P aipc_main.py "$@" > read_ab_last.log 2>&1; grep -a "^{" read_ab_last.log | cut -c1-300; grep -a "Error" read_ab_last.log | tail -1 | cut -c1-200; }
for v in "base:" "split:--split-sizes 1" "spine:--split-sizes 1 --spine-size 1"; do
  t=${v%%:*}; o=${v#*:}
  run score t1_drive t2_drive t1_orig $F $o --tag rd_$t
  run task3 $F $o --claude-slots 1 --tag rd_${t}_t3
done
echo READ_AB_DONE

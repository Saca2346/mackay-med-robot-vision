#!/bin/bash
# 7 Nov 2026: Leo's Drive photos, every box read (size read whatever the brand; identity for catalogue products), the
# final reader without and with the reading options. Accuracy run (other runs may share the GPU; times not reported).
cd "$(dirname "$0")"
P=~/trse_venv/bin/python
export PYTHONPATH=../pylib
F="--imgsz 544x960 --workers 27 --cls-cpu 1 --proc-threads 1 --order skim --variants 1 --lazy 1 --ordered-prefetch 1 --priority 1 --procs 16"
run() { echo "== $(date +%T) $*"; $P aipc_main.py "$@" > drive_ab_last_$T.log 2>&1; grep -a "^{" drive_ab_last_$T.log | cut -c1-300; grep -a "Error" drive_ab_last_$T.log | tail -1 | cut -c1-200; }
T=base; run drive $F --tag dr_base
T=read; run drive $F --split-sizes 1 --spine-size 1 --tag dr_read
echo DRIVE_AB_DONE

#!/bin/bash
# 7 Nov 2026: the laptop's recordings replayed from the first 5 camera stills of each take (read until a PICK), the
# reader as in the final run, without and with the reading options (--split-sizes 1 --spine-size 1).
cd "$(dirname "$0")"
P=~/trse_venv/bin/python
export PYTHONPATH=../pylib
while ! grep -q DRIVE_AB_DONE drive_ab.log 2>/dev/null; do sleep 60; done
F="--imgsz 544x960 --workers 27 --cls-cpu 1 --proc-threads 1 --order skim --variants 1 --lazy 1 --ordered-prefetch 1 --priority 1 --procs 20"
run() { echo "== $(date +%T) $*"; $P aipc_main.py "$@" > live_multi_last.log 2>&1; grep -a "^{" live_multi_last.log | cut -c1-500; grep -a "Error" live_multi_last.log | tail -1 | cut -c1-200; }
S="../livesets/box20_multi ../livesets/box12_multi ../livesets/box10close_multi"
run live $S $F --tag lm_base
run live $S $F --split-sizes 1 --spine-size 1 --tag lm_read
echo LIVE_MULTI_DONE

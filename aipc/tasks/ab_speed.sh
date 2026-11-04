#!/bin/bash
# Speed A/B (4 Nov 2026, night), configurations interleaved in rounds so a changing load from other users falls on all
# of them alike; GPU load logged. Decisions are compared too (picked box of every run).
#   arena:  OCR_ARENA=default (grow to the next power of two, as before 23:00) | same (grow by what is asked)
#   crops:  AIPC_EAGER_CROPS=1 (all 4 outlines' crops made before the skim, as before) | lazy (made when needed)
cd "$(dirname "$0")"
P=~/trse_venv/bin/python
export PYTHONPATH=../pylib
(while true; do echo $(date +%T) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader) $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .); sleep 3; done) > ab_gpu.log 2>&1 &
L=$!
C="--imgsz 544x960 --workers 27 --cls-cpu 1 --proc-threads 1 --order skim --snap 1"
one() {  # tag arena eager procs variants-args
  echo "== $(date +%T) $1"
  OCR_ARENA=$2 AIPC_EAGER_CROPS=$3 $P aipc_main.py time t1_drive t2_drive --runs 3 $C --procs $4 $5 --tag ab_$1_r$r > ab_last.log 2>&1
  grep -a "^{" ab_last.log | $P -c "
import sys,json
for l in sys.stdin:
    d=json.loads(l); print(' ', d['set'], d['picked_box'], d['pick_s_median'], round(sum(r['skim_s'] or 0 for r in d['raw'])/len(d['raw']),3))"
  grep -a "Error" ab_last.log | tail -1 | cut -c1-120
}
for r in 1 2; do
  one novar_default default 0 24 "--variants 0"
  one novar_same same 0 24 "--variants 0"
  one var4_eager_same same 1 24 "--variants 1 --lazy 1"
  one var4_lazy_same same 0 24 "--variants 1 --lazy 1"
  one var4_lazy_default default 0 24 "--variants 1 --lazy 1"
  one var4_lazy_same16 same 0 16 "--variants 1 --lazy 1"
done
kill $L
echo AB_DONE

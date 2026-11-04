#!/bin/bash
# Quiet-GPU timing, round 2 (after quiet_ab.sh): ordered prefetch against the baseline, interleaved; then the final
# reader on the WhatsApp copy and the 41 Task 3 requests. Same idle gate and GPU log as quiet_ab.sh.
cd "$(dirname "$0")"
P=~/trse_venv/bin/python
export PYTHONPATH=../pylib
while pgrep -f "bash quiet_ab.sh" > /dev/null; do sleep 10; done
(while true; do echo $(date +%T) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader) $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .); sleep 2; done) > quiet2_gpu.log 2>&1 &
L=$!
idle() { q=0; while [ $q -lt 10 ]; do u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d " "); if [ "$u" -lt 10 ]; then q=$((q+1)); else q=0; fi; sleep 3; done; }
C="--imgsz 544x960 --workers 27 --cls-cpu 1 --proc-threads 1 --order skim --snap 1 --procs 24"
one() {  # tag arena sets args
  idle; echo "== $(date +%T) $1"
  OCR_ARENA=$2 $P aipc_main.py time $3 --runs 3 $C $4 --tag q2_$1_r$r > q2_last.log 2>&1
  grep -a "Error" q2_last.log | tail -1 | cut -c1-120
}
for r in 1 2; do
  one novar_unordered default "t1_drive t2_drive" "--variants 0"
  one novar_ordered default "t1_drive t2_drive" "--variants 0 --ordered-prefetch 1"
  one var4_ordered same "t1_drive t2_drive" "--variants 1 --lazy 1 --ordered-prefetch 1"
done
r=0; one final_orig same "t1_orig" "--variants 1 --lazy 1 --ordered-prefetch 1"
idle; echo "== $(date +%T) task3"
$P aipc_main.py task3 $C --variants 1 --lazy 1 --ordered-prefetch 1 --procs 20 --claude-slots 1 --tag q2_final_t3 > q2_last.log 2>&1
grep -a "Error" q2_last.log | tail -1 | cut -c1-120
kill $L
echo QUIET2_DONE

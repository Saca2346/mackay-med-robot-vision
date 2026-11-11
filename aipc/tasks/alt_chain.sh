#!/bin/bash
# 8 Nov 2026: after step 2 of acc_chain.sh (views on the task photos), its Drive and recording runs are stopped and this
# runs first: the other view's outline of the same box read as an extra outline (--alt-views 1; rd_va vs rd_vb showed
# the 544 and the 1280 outline of one box reading different size labels), then the honest input-size probe, then the
# baselines of the Drive photos and the recordings. One run at a time; each stopped after its time limit.
cd "$(dirname "$0")"
P=~/trse_venv/bin/python
while ! grep -q "== .* dr_base " acc_chain.log 2>/dev/null; do sleep 10; done
for p in $(pgrep -f "bash acc_chain.sh"); do kill $p; done
sleep 1
for p in $(pgrep -f -- "--tag dr_base"); do kill $p; done
sleep 5
echo "acc_chain stopped before its Drive runs: $(date +%T)"
F="--workers 27 --cls-cpu 1 --proc-threads 1 --order skim --variants 1 --lazy 1 --ordered-prefetch 1 --priority 1"
R="--split-sizes 1 --spine-size 1"
VA="B:544x960:0+B:1280:0"
VB="B:1280:0+B:544x960:0"
run() {
  T=$1; L=$2; shift 2
  echo "== $(date +%T) $T $*"
  PYTHONPATH=../pylib timeout -k 60 $L $P aipc_main.py "$@" --tag $T > chain_last_$T.log 2>&1
  echo "exit $? engine_errors $(grep -ac OCR_ENGINE_ERROR chain_last_$T.log)"
  grep -a "^{" chain_last_$T.log | cut -c1-400
  grep -a "Error" chain_last_$T.log | tail -1 | cut -c1-200
}
for v in "alta:--views $VA" "altb:--views $VB"; do
  t=${v%%:*}; o=${v#*:}
  run rd_$t 900 score t1_drive t2_drive t1_orig --imgsz 544x960 $F $o --alt-views 1 $R --procs 20
  run rd_${t}_t3 1500 task3 --imgsz 544x960 $F $o --alt-views 1 $R --procs 20 --claude-slots 1
done
( cd ../det
  for c in s26h_544 s12h_544; do
    [ -f probe/${c}_test.json ] || PYTHONPATH= timeout 1800 $P det_probe.py heldout $c --tilt 0,20,35 --photos test 2>&1 | grep -v Warning
  done )
run dr_base 5400 drive --imgsz 544x960 $F --procs 16
S="../livesets/box20_multi ../livesets/box12_multi ../livesets/box10close_multi"
run lm_base 3600 live $S --imgsz 544x960 $F --procs 20
echo ALT_CHAIN_DONE

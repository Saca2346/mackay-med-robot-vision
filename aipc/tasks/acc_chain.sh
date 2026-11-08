#!/bin/bash
# 8 Nov 2026: accuracy runs one after another, so no two runs share the GPU. On 7 Nov a Drive run started while the
# detector probe ran: an OCR worker could not get GPU memory (CUBLAS_STATUS_ALLOC_FAILED), the worker pool broke and
# the run waited 18 h. Now each run is stopped after its time limit. Accuracy only (cells incl. FP); no times reported.
#   detector views: VA = 544x960 first (the boxes it finds keep their reads), 1280 adds the boxes it missed
#                   VB = 1280 first, 544x960 adds;  v1280 = the reader's detector at 1280 alone
cd "$(dirname "$0")"
P=~/trse_venv/bin/python
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
# 1. detector only, 1280 first: held-out 28-Sep photos (tilt 0, 20) and box counts on Leo's Drive photos
D=../code/results/drive_photos_20261029/src_photos/main
( cd ../det
  for c in o_B1280_544 o_B1280_rot o_B1280_s12; do
    [ -f probe/${c}_test.json ] || PYTHONPATH= timeout 1800 $P det_probe.py heldout $c --tilt 0,20 --photos test 2>&1 | grep -v Warning
  done
  for c in B1280 o_B1280_544 o_B1280_s12; do
    [ -f probe/drive_$c/count.json ] || PYTHONPATH= timeout 1800 $P det_probe.py count $c probe/drive_$c $D/*.jpg > /dev/null 2>&1
    echo "drive $c $(python3 -c "import json;d=json.load(open('probe/drive_$c/count.json'))['boxes'];print(sum(d.values()))")"
  done )
# 2. task photos and Task 3 with the reading options: views VA, VB and 1280 alone (compare with rd_spine, rd_spine_t3)
for v in "va:--imgsz 544x960 --views $VA" "vb:--imgsz 544x960 --views $VB" "v1280:--imgsz 1280"; do
  t=${v%%:*}; o=${v#*:}
  run rd_$t 900 score t1_drive t2_drive t1_orig $F $o $R --procs 20
  run rd_${t}_t3 1500 task3 $F $o $R --procs 20 --claude-slots 1
done
# 3. Leo's Drive photos (every box) without and with the reading options
run dr_base 5400 drive --imgsz 544x960 $F --procs 16
run dr_read 5400 drive --imgsz 544x960 $F $R --procs 16
# 4. recordings, the first 5 camera stills of each take, without and with the reading options
S="../livesets/box20_multi ../livesets/box12_multi ../livesets/box10close_multi"
run lm_base 3600 live $S --imgsz 544x960 $F --procs 20
run lm_read 3600 live $S --imgsz 544x960 $F $R --procs 20
echo ACC_CHAIN_DONE

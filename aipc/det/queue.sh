#!/bin/bash
# Detector candidates, one after the other on the shared A5000 (honest split: 28-Sep photos held out).
cd "$(dirname "$0")"
P=~/trse_venv/bin/python
for job in "yolo26s-obb.pt 1280 16" "yolo26m-obb.pt 1280 8" "yolo26l-obb.pt 1280 6" "yolo26x-obb.pt 1280 4" \
           "yolo11x-obb.pt 1280 4"; do
  set -- $job
  n="${1%.pt}_$2_hold"
  [ -f "runs/$n/done" ] && continue
  echo "START $n $(date)"
  $P train_det.py $1 $2 $3 100 hold > "log_$n.txt" 2>&1
  echo "END $n $(date) $(tail -1 log_$n.txt)"
done
echo QUEUE_DONE

#!/bin/bash
# Round 2 (4 Nov evening): s and m showed the same held-out result (size is not the limit), so: photo-focused recipe at
# 1600 px, another architecture family (YOLO11m), each evaluated on the held-out 28-Sep photos right after training.
cd "$(dirname "$0")"
P=~/trse_venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for job in "yolo26m-obb.pt 1600 4 100 hold 6 6" "yolo26s-obb.pt 1600 8 100 hold 6 6" "yolo11m-obb.pt 1280 6 100 hold 2 3"; do
  set -- $job
  tag=""; [ "$6 $7" != "2 3" ] && tag="_p$6f$7"
  n="${1%.pt}_$2_$5$tag"
  if [ ! -f "runs/$n/done" ]; then
    echo "START $n $(date)"
    $P train_det.py $1 $2 $3 $4 $5 $6 $7 > "log_$n.txt" 2>&1
    echo "END $n $(date) $(tail -1 log_$n.txt)"
  fi
  [ -f "eval/$n.json" ] || $P eval_det.py "runs/$n/weights/last.pt" $2 "$n" 2>&1 | tail -1
done
echo QUEUE3_DONE

#!/bin/bash
# Round 3 (5 Nov): YOLO26s at 1600 px with the photo-focused recipe. Its queue3 slot was stopped in epoch 1 (no weights),
# so it is trained here once queue3 (YOLO11m) has finished, one job at a time on the shared GPU.
cd "$(dirname "$0")"
P=~/trse_venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "[q]ueue3.sh" > /dev/null; do sleep 60; done
for job in "yolo26s-obb.pt 1600 8 100 hold 6 6"; do
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
echo QUEUE4_DONE

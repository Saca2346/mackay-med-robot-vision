#!/bin/bash
# Detector candidates on the shared A5000, then each one's held-out evaluation on the GPU right after its training
# (honest split: 28-Sep photos held out). Smaller batches than queue.sh so other users keep some GPU memory.
cd "$(dirname "$0")"
P=~/trse_venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "train_det.py yolo26s-obb.pt" > /dev/null; do sleep 60; done      # the running yolo26s job
for job in "yolo26s-obb.pt 1280 16" "yolo26m-obb.pt 1280 6" "yolo26l-obb.pt 1280 4" "yolo26x-obb.pt 1280 3" \
           "yolo11x-obb.pt 1280 3"; do
  set -- $job
  n="${1%.pt}_$2_hold"
  if [ ! -f "runs/$n/done" ]; then
    echo "START $n $(date)"
    $P train_det.py $1 $2 $3 100 hold > "log_$n.txt" 2>&1
    echo "END $n $(date) $(tail -1 log_$n.txt)"
  fi
  [ -f "eval/$n.json" ] || $P eval_det.py "runs/$n/weights/last.pt" $2 "$n" 2>&1 | tail -1
done
echo QUEUE_DONE

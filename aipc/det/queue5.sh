#!/bin/bash
# Round 4 (6 Nov): two more families asked by the owner, YOLOv8 and YOLO12, s and m, the round-1 recipe at 1280 px
# without the 28-Sep session; each evaluated on the held-out 28-Sep photos right after training. YOLO12 has no
# published OBB weights: built from its OBB yaml with the official detection weights (train_det.py).
cd "$(dirname "$0")"
P=~/trse_venv/bin/python
export PYTHONPATH=
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for job in "yolov8s-obb.pt 1280 8" "yolo12s-obb.yaml 1280 8" "yolov8m-obb.pt 1280 6" "yolo12m-obb.yaml 1280 6"; do
  set -- $job
  n="${1%.*}_$2_hold"
  if [ ! -f "runs/$n/done" ]; then
    echo "START $n $(date)"
    $P train_det.py $1 $2 $3 100 hold > "log_$n.txt" 2>&1
    echo "END $n $(date) $(tail -1 log_$n.txt)"
  fi
  [ -f "eval/$n.json" ] || $P eval_det.py "runs/$n/weights/last.pt" $2 "$n" 2>&1 | tail -1
done
echo QUEUE5_DONE

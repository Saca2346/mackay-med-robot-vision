#!/bin/bash
# 7 Nov 2026: the YOLOv8 and YOLO12 candidates (trained without the 28 Sep session, queue5.sh) on the task photos,
# exactly like the YOLO11m run (task_y11m1280): 1280 px, the four outline variants, 20 processes.
cd "$(dirname "$0")"
P=~/trse_venv/bin/python
export PYTHONPATH=../pylib
for n in yolov8s yolo12s yolov8m yolo12m; do
  t=task_${n}1280
  [ -f out/$t/task3.json ] && continue
  W=../det/runs/${n}-obb_1280_hold/weights/last.pt
  echo "== $(date +%T) $n"
  $P aipc_main.py score t1_drive t2_drive t1_orig --det $W --imgsz 1280 --workers 27 --procs 20 --tag $t > det_last.log 2>&1
  grep -a "^{" det_last.log | cut -c1-260; grep -a "Error" det_last.log | tail -1 | cut -c1-200
  $P aipc_main.py task3 --det $W --imgsz 1280 --workers 27 --procs 20 --claude-slots 1 --tag $t > det_last.log 2>&1
  grep -a "^{" det_last.log | cut -c1-400; grep -a "Error" det_last.log | tail -1 | cut -c1-200
done
echo DET_TASKS_DONE

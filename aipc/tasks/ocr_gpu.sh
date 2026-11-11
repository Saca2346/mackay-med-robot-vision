#!/bin/bash
# Remaining OCR engines (7 Nov 2026), after the detector training freed the GPU: accuracy (Task 1, Task 2, WhatsApp
# copy and the 41 Task 3 requests) for PP-OCRv5 server + English reader, EasyOCR and Tesseract with the same flags as
# ocr_compare.sh, then the time of every engine on Task 1 and Task 2 with the final reader (3 runs each).
cd "$(dirname "$0")"
P=~/trse_venv/bin/python
export PYTHONPATH=../pylib
dev_for() {
  case $1 in
    tess) echo "--proc-device cpu --procs 24 --proc-threads 1";;
    v5det|v5srv|v5en) echo "--proc-device cuda --procs 8 --proc-threads 2";;
    easy) echo "--proc-device cuda --procs 4 --proc-threads 2";;
    *) echo "--proc-device cuda --procs 24 --proc-threads 1";;
  esac
}
F="--imgsz 544x960 --workers 16 --variants 1 --lazy 1 --ordered-prefetch 1 --priority 1"
for v in v5en easy tess; do
  [ -f out/ocr_$v/task3.json ] && continue
  echo "== $(date +%T) $v"
  OCR_VARIANT=$v $P aipc_main.py score t1_drive t2_drive t1_orig $F $(dev_for $v) --tag ocr_$v > ocr_last.log 2>&1
  grep -a "^{" ocr_last.log | cut -c1-260; grep -a "Error" ocr_last.log | tail -1 | cut -c1-200
  OCR_VARIANT=$v $P aipc_main.py task3 $F $(dev_for $v) --claude-slots 1 --tag ocr_$v > ocr_last.log 2>&1
  grep -a "^{" ocr_last.log | cut -c1-400; grep -a "Error" ocr_last.log | tail -1 | cut -c1-200
done
T="--imgsz 544x960 --workers 27 --cls-cpu 1 --order skim --snap 1 --refine 1 --refine-device cuda --draw-imgsz 1280 --variants 1 --lazy 1 --ordered-prefetch 1 --priority 1"
for v in v6s v5det v5srv v5en easy tess; do
  echo "== $(date +%T) time $v"
  OCR_VARIANT=$v $P aipc_main.py time t1_drive t2_drive --runs 3 $T $(dev_for $v) --tag ocrt_$v > ocr_last.log 2>&1
  grep -a "^{" ocr_last.log | cut -c1-400; grep -a "Error" ocr_last.log | tail -1 | cut -c1-200
done
bash det_tasks.sh
echo OCR_GPU_DONE

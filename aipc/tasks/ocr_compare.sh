#!/bin/bash
# OCR engines compared (6 Nov 2026), accuracy only: the final reader (four outlines, lazy, the laptop's two-source
# rule) with each OCR_VARIANT (ocr_worker.py) on every task-photo pair and the 41 Task 3 requests. 8 OCR processes
# (the GPU is shared with the detector training of queue5.sh), so the times in these files are not the reader's.
cd "$(dirname "$0")"
P=~/trse_venv/bin/python
export PYTHONPATH=../pylib
# The PP-OCRv5 server detector ran out of the shared card's memory beside the training (558 MB buffers, 6 Nov 18:06),
# so the server variants and Tesseract read on the CPU (8 processes x 4 threads) and EasyOCR on 4 GPU processes.
F="--imgsz 544x960 --workers 16 --variants 1 --lazy 1 --ordered-prefetch 1 --priority 1"
for v in v6s v5det v5srv v5en easy tess; do
  dev="--proc-device cpu --procs 8 --proc-threads 4"
  [ $v = v6s ] && dev="--proc-device cuda --procs 8 --proc-threads 2"
  [ $v = easy ] && dev="--proc-device cuda --procs 4 --proc-threads 2"
  [ -f out/ocr_$v/task3.json ] && continue
  echo "== $(date +%T) $v"
  OCR_VARIANT=$v $P aipc_main.py score t1_drive t2_drive t1_orig $F $dev --tag ocr_$v > ocr_last.log 2>&1
  grep -a "^{" ocr_last.log | cut -c1-260; grep -a "Error" ocr_last.log | tail -1 | cut -c1-200
  OCR_VARIANT=$v $P aipc_main.py task3 $F $dev --claude-slots 1 --tag ocr_$v > ocr_last.log 2>&1
  grep -a "^{" ocr_last.log | cut -c1-400; grep -a "Error" ocr_last.log | tail -1 | cut -c1-200
done
echo OCR_COMPARE_DONE

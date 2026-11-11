#!/bin/bash
# 7 Nov 2026: let the PP-OCRv5 server Task 3 run (PID given) finish, then the final timing on the quiet GPU, then the
# remaining OCR engines on the GPU.
cd "$(dirname "$0")"
while kill -0 $1 2>/dev/null; do sleep 30; done
echo "v5srv done $(date)"; grep -a "^{" ocr_last.log | cut -c1-400
bash final_quiet.sh > final_quiet.log 2>&1
bash ocr_gpu.sh > ocr_gpu.log 2>&1
echo "CHAIN_DONE $(date)"

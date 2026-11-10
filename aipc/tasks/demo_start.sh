#!/bin/bash
# 10 Nov 2026: documentation recording of starting the program on the AI PC (recorded with asciinema rec, real time).
# Each command is printed after a prompt line, then run for real. One Task 1 run (Drive photo) with the final flags of
# job 103 (fqzg), tag demo1110 (scratch output, not a result folder).
say() { printf '\n\033[1;32mZen@%s\033[0m:\033[1;34m%s\033[0m$ \033[1m%s\033[0m\n' "$(hostname)" "$(pwd | sed "s|$HOME|~|")" "$*"; sleep 1.5; }
note() { printf '\n\033[1;33m# %s\033[0m\n' "$*"; sleep 1.5; }

note "AI PC: which machine, GPU and CPU"
say hostname; hostname
say date; date
say nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
say 'lscpu | grep -E "Model name|^CPU\(s\)"'; lscpu | grep -E "Model name|^CPU\(s\)"

note "The program folder and its files"
say cd ~/Downloads/claude_offline_20261101/aipc_20261104/tasks
cd ~/Downloads/claude_offline_20261101/aipc_20261104/tasks
say ls -l aipc_main.py aipc_tasks.py pick_parts.py ../../models/box_y26_maxB.pt
ls -l aipc_main.py aipc_tasks.py pick_parts.py ../../models/box_y26_maxB.pt
say ~/trse_venv/bin/python --version; ~/trse_venv/bin/python --version
say head -2 aipc_main.py; head -2 aipc_main.py

note "Start: Task 1 (Drive photo), 1 run, final flags (same as job 103). [INFO] lines of the OCR library are hidden on screen, the full output goes to demo1110.log"
say export OCR_ALGO=HEURISTIC
export OCR_ALGO=HEURISTIC
CMD='PYTHONPATH=../pylib ~/trse_venv/bin/python aipc_main.py time t1_drive --runs 1 --imgsz 544x960 --workers 27 --cls-cpu 1 --proc-threads 1 --order skim --snap 1 --refine 1 --refine-device cuda --draw-imgsz 1280 --variants 1 --lazy 1 --ordered-prefetch 1 --priority 1 --refine-webcam 0 --webcam-ends 1 --views B:1280:0+B:544x960:0 --alt-views 1 --alt-lazy 1 --skim-name 1 --split-sizes 1 --spine-size 1 --par-variants 1 --pick-crops 1 --procs 24 --tag demo1110 2>&1 | tee demo1110.log | grep -av "\[INFO\]" | cut -c1-230'
say "time ( $CMD )"
time ( eval "$CMD" )

note "Result of this run (from the result line in demo1110.log)"
SUM='grep -a "^{" demo1110.log | ~/trse_venv/bin/python -c "import json,sys; d=json.loads(sys.stdin.readline()); [print(k, \"=\", d[k]) for k in (\"set\",\"request\",\"picked_box\",\"matches_each_run\",\"pick_s_median\",\"all_read_s_median\")]"'
say "$SUM"; eval "$SUM"
say ls out/demo1110/crops/; ls out/demo1110/crops/
CR='~/trse_venv/bin/python -c "import json,glob; d=json.load(open(glob.glob(\"out/demo1110/crops/*/crops.json\")[0])); [print(k, \"=\", d.get(k)) for k in (\"product_name\",\"brand_name\",\"brand_from\",\"detail_text\",\"barcode\")]"'
say "$CR"; eval "$CR"
note "Done"
sleep 3

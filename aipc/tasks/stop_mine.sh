#!/bin/bash
# Stop this work's runners and their OCR processes (not the training queue, not other users).
me=$$
for p in $(pgrep -u "$USER" -f "when_free.sh|final.sh|video_runs.sh|aipc_main.py|multiprocessing.spawn|multiprocessing.resource_tracker"); do
  [ "$p" = "$me" ] && continue
  cmd=$(ps -o cmd= -p $p)
  case "$cmd" in *train_det*|*stop_mine*) continue;; esac
  kill $p 2>/dev/null
done
sleep 3
pgrep -u "$USER" -af "aipc_main|multiprocessing|when_free|final.sh|video_runs" | grep -v stop_mine | cut -c1-100
echo STOPPED

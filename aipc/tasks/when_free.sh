#!/bin/bash
# Wait until the shared GPU is quiet (no other compute user above 1 GB, utilisation < 30% for 1 min), then the final
# timing run (final.sh, tag final_B_free) with its GPU log; afterwards release the training queue (det/PAUSE).
cd "$(dirname "$0")"
quiet=0
while [ $quiet -lt 6 ]; do
  u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d " ")
  big=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | awk "\$1>1000" | wc -l)
  if [ "$u" -lt 30 ] && [ "$big" -eq 0 ]; then quiet=$((quiet+1)); else quiet=0; fi
  sleep 10
done
echo "GPU quiet $(date)"
(while true; do echo $(date +%T) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader) $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .); sleep 5; done) > final_free_gpu.log 2>&1 &
LOGGER=$!
bash final.sh "" 544x960 final_B_free
kill $LOGGER
rm -f ../det/PAUSE
echo "WHEN_FREE_DONE $(date)"

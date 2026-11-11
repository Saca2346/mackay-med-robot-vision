#!/bin/bash
# 8 Nov 2026: GPU runs one at a time from a folder of job files (queue/NN_name.job, run in name order). Each job is a
# few lines of bash that call run TAG LIMIT_S MODE ARGS... (aipc_main.py, stopped after LIMIT_S) or any other command.
# Starts after the run named in $1 (a log line marker, e.g. ALT_CHAIN_DONE in alt_chain.log). Stops when queue/STOP
# exists and no job is left. Log: queue_runner.log; each run's full output: chain_last_TAG.log.
cd "$(dirname "$0")"
P=~/trse_venv/bin/python
F="--workers 27 --cls-cpu 1 --proc-threads 1 --order skim --variants 1 --lazy 1 --ordered-prefetch 1 --priority 1"
R="--split-sizes 1 --spine-size 1"
VA="B:544x960:0+B:1280:0"
VB="B:1280:0+B:544x960:0"
S="../livesets/box20_multi ../livesets/box12_multi ../livesets/box10close_multi"
run() {
  T=$1; L=$2; shift 2
  echo "== $(date +%T) $T $*"
  PYTHONPATH=../pylib timeout -k 60 $L $P aipc_main.py "$@" --tag $T > chain_last_$T.log 2>&1
  echo "exit $? engine_errors $(grep -ac OCR_ENGINE_ERROR chain_last_$T.log)"
  grep -a "^{" chain_last_$T.log | cut -c1-500
  grep -a "Error" chain_last_$T.log | tail -1 | cut -c1-200
}
mkdir -p queue/done
[ -n "$1" ] && while ! grep -q "$1" ${2:-alt_chain.log} 2>/dev/null; do sleep 15; done
echo "queue start $(date +%T)"
while true; do
  j=$(ls queue/*.job 2>/dev/null | sort | head -1)
  if [ -z "$j" ]; then
    [ -f queue/STOP ] && break
    sleep 20
    continue
  fi
  echo "## job $(basename $j) $(date +%T)"
  mv "$j" queue/done/
  . "queue/done/$(basename $j)"
done
echo QUEUE_DONE

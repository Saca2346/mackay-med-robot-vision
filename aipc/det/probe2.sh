#!/bin/bash
# 7 Nov 2026: ordered merges (first view kept, later views add missed boxes) on the held-out 28-Sep photos (tilt 0, 20)
# and box counts on Leo's Drive photos. After probe_run.sh.
cd "$(dirname "$0")"
P=~/trse_venv/bin/python
export PYTHONPATH=
D=../code/results/drive_photos_20261029/src_photos/main
while ! grep -q PROBE_DONE probe_run.log 2>/dev/null; do sleep 60; done
for c in o_s26_s12 o_s26_m26_s12 o_s26_rot o_s26_all o_B o_B_s12 o_B_all; do
  [ -f probe/${c}_test.json ] || $P det_probe.py heldout $c --tilt 0,20 --photos test 2>&1 | grep -v Warning
done
for c in o_B o_B_s12 o_B_all; do
  [ -f probe/drive_$c/count.json ] || $P det_probe.py count $c probe/drive_$c $D/*.jpg > /dev/null 2>&1
  echo "drive $c $(python3 -c "import json;d=json.load(open('probe/drive_$c/count.json'))['boxes'];print(sum(d.values()))")"
done
echo PROBE2_DONE

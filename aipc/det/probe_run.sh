#!/bin/bash
# 7 Nov 2026: tilted / missed boxes. Every config on the held-out 28-Sep photos (also turned by 20 and 35 degrees),
# then box counts on Leo's Drive photos. Model B configs saw the 28-Sep photos (upper bound); *h configs did not.
cd "$(dirname "$0")"
P=~/trse_venv/bin/python
export PYTHONPATH=
D=../code/results/drive_photos_20261029/src_photos/main
for c in s26h s26h_rot s12h s12h_rot ens_h ens_h_rot ens_h_rot_s2 B544 B1280 B_ms B_rot B_ms_rot B1280_c20 B_ms_rot_c25; do
  [ -f probe/${c}_test.json ] || $P det_probe.py heldout $c --tilt 0,20,35 --photos test 2>&1 | grep -v Warning
done
for c in B544 B_ms_rot B_ms_rot_c25 ens_h_rot; do
  [ -f probe/drive_$c/count.json ] || $P det_probe.py count $c probe/drive_$c $D/*.jpg > /dev/null 2>&1
  echo "drive $c $(python3 -c "import json;d=json.load(open('probe/drive_$c/count.json'))['boxes'];print(sum(d.values()))")"
done
echo PROBE_DONE

#!/bin/bash
# Product-look model on the owner-reviewed labels: same recipe as 2 Nov (EfficientNet-B0, 30 epochs, 3 seeds), no
# automatic label filter (--k 1000) because the labels are now reviewed by the owner.
cd "$(dirname "$0")"
P=~/trse_venv/bin/python
for s in 0 1 2; do
  $P train_id.py --split photo --arch efficientnet_b0 --epochs 30 --seed $s --k 1000 --out runs/owner_photo_b0_s$s
  $P train_id.py --split shelf --arch efficientnet_b0 --epochs 30 --seed $s --k 1000 --out runs/owner_shelf_b0_s$s
done
echo LOOK_DONE

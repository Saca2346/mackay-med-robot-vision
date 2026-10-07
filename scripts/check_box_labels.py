#!/usr/bin/env python3
"""
Cari label "longgar": kotak box yang saling tumpang tindih terlalu banyak.

Pada foto dengan kemasan miring, label kotak lurus (tidak diputar) ikut menutupi
kemasan di sebelahnya -> IoU antar label box yang bertetangga jadi besar. Label
yang benar (diputar mengikuti kemasan) hampir tidak saling tumpang tindih.

Yang dilaporkan per foto:
  max_iou      : IoU terbesar antara dua label box di foto itu
  pairs_over   : jumlah pasangan box dengan IoU > --iou-thr
  straight     : jumlah box yang lurus (miring < 1 derajat)
  orphan_label : size_label yang pusatnya tidak berada di dalam box mana pun

Contoh:
  python scripts/check_box_labels.py --labels data/box_dataset/labels --images data/box_dataset/images
  python scripts/check_box_labels.py --labels data/box_dataset/labels --iou-thr 0.3
"""
from __future__ import annotations

import argparse
import csv
import itertools
import pathlib
import sys

import cv2

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.box_geometry import axis_deviation_deg, center, draw_poly, point_in_poly, poly_iou, put_label, read_label_file
from src.config import resolve

IMG_EXT = [".jpg", ".jpeg", ".png", ".bmp"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--images", default=None, help="kalau diisi: simpan gambar foto yang bermasalah")
    ap.add_argument("--iou-thr", type=float, default=0.3)
    ap.add_argument("--box-class", type=int, default=0)
    ap.add_argument("--label-class", type=int, default=1)
    ap.add_argument("--out", default="results/label_check")
    args = ap.parse_args()

    lab_dir = pathlib.Path(resolve(args.labels))
    img_dir = pathlib.Path(resolve(args.images)) if args.images else None
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for lf in sorted(lab_dir.glob("*.txt")):
        img, w, h = None, 1000, 1000   # tanpa foto: koordinat 0..1 diskalakan ke 1000x1000 (IoU tidak berubah)
        if img_dir:
            for ext in IMG_EXT:
                p = img_dir / (lf.stem + ext)
                if p.exists():
                    img = cv2.imread(str(p))
                    h, w = img.shape[:2]
                    break
        labels = read_label_file(lf, w, h)
        boxes = [p for c, p in labels if c == args.box_class]
        sizes = [p for c, p in labels if c == args.label_class]
        bad_pairs = []
        max_iou = 0.0
        for (i, a), (j, b) in itertools.combinations(enumerate(boxes), 2):
            iou = poly_iou(a, b)
            max_iou = max(max_iou, iou)
            if iou > args.iou_thr:
                bad_pairs.append((i, j, iou))
        straight = sum(1 for b in boxes if axis_deviation_deg(b) < 1.0)
        orphans = [s for s in sizes if not any(point_in_poly(center(s), b) for b in boxes)]
        rows.append({"file": lf.name, "boxes": len(boxes), "size_labels": len(sizes),
                     "max_iou": round(max_iou, 3), "pairs_over": len(bad_pairs),
                     "straight": straight, "orphan_label": len(orphans),
                     "redraw": "YES" if bad_pairs or orphans else ""})
        if img is not None and (bad_pairs or orphans):
            vis = img.copy()
            flagged = {i for i, _, _ in bad_pairs} | {j for _, j, _ in bad_pairs}
            for k, b in enumerate(boxes):
                draw_poly(vis, b, (0, 0, 255) if k in flagged else (0, 200, 0), 3 if k in flagged else 1)
            for i, j, iou in bad_pairs:
                put_label(vis, f"IoU {iou:.2f}", center(boxes[i]), (0, 0, 255), 0.6)
            for s in orphans:
                draw_poly(vis, s, (255, 0, 255), 2)
            cv2.imwrite(str(out / f"{lf.stem}_check.jpg"), vis)

    rows.sort(key=lambda r: (-r["pairs_over"], -r["max_iou"]))
    with open(out / "label_check.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["file"])
        wr.writeheader()
        wr.writerows(rows)
    flagged = [r for r in rows if r["redraw"]]
    print(f"{len(rows)} file label dicek, {len(flagged)} perlu digambar ulang (IoU antar box > {args.iou_thr} "
          f"atau size_label di luar box).")
    for r in flagged[:25]:
        print(f"  {r['file']:40s} max_iou {r['max_iou']:.2f}  pasangan>{args.iou_thr}: {r['pairs_over']}  "
              f"lurus: {r['straight']}/{r['boxes']}  size_label yatim: {r['orphan_label']}")
    print(f"\nTabel lengkap: {out / 'label_check.csv'}")


if __name__ == "__main__":
    main()

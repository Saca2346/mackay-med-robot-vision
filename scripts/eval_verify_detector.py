#!/usr/bin/env python3
"""
Evaluasi verifikasi DENGAN detektor: kotak dan size_label berasal dari prediksi model, bukan dari label
ground truth seperti scripts/eval_verify.py.

1. Tiap foto di --gt diprediksi oleh model. Dengan --fold-weights/--fold-lists dipakai model fold yang TIDAK
   melihat foto itu saat training (out-of-fold), supaya hasilnya tidak terlalu optimistis.
2. Tiap kotak GT dicocokkan dengan kotak prediksi (IoU poligon >= --iou-match, satu-satu, IoU terbesar dulu).
   Kotak GT tanpa pasangan = terlewat oleh detektor; verifier tidak pernah melihatnya (aman, tapi tidak MATCH).
3. scripts/eval_verify.py dijalankan pada label prediksi dengan GT yang nomor kotaknya sudah dipetakan ulang.

Syarat keselamatan sama: SALAH MATCH = 0.

Contoh (v6, out-of-fold):
  python scripts/eval_verify_detector.py --images ~/Downloads/prelabel_size_label_full/raw \
     --gt-labels ~/Downloads/prelabel_size_label_full/labels_final_obb \
     --fold-weights "~/Downloads/prelabel_size_label_full/runs/cv/yolo11s-obb@180_fold{k}_i1280_e200/weights/best.pt" \
     --fold-lists "~/Downloads/prelabel_size_label_full/export/cv/fold{k}/labels/val" \
     --out-dir results/eval_detector_v6 --today 2026-10-04
"""
from __future__ import annotations

import argparse
import csv
import os
import pathlib
import runpy
import sys

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.box_geometry import read_label_file  # noqa: E402


def poly_iou(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    inter, _ = cv2.intersectConvexConvex(cv2.convexHull(a), cv2.convexHull(b))
    union = cv2.contourArea(a) + cv2.contourArea(b) - inter
    return inter / union if union > 0 else 0.0


def names_in(path):
    p = pathlib.Path(os.path.expanduser(path))
    if p.is_dir():
        return {f.stem for f in p.glob("*.txt")}
    return {ln.strip() for ln in open(p, encoding="utf-8") if ln.strip()}


def model_for_photos(args, photos):
    """{photo: weights path}: the out-of-fold model of each photo, or the single --weights for all."""
    if args.weights:
        return {p: os.path.expanduser(args.weights) for p in photos}
    out = {}
    for k in range(args.folds):
        val = names_in(args.fold_lists.format(k=k))
        for p in photos & val:
            out[p] = os.path.expanduser(args.fold_weights.format(k=k))
    missing = photos - set(out)
    if missing and args.fallback_weights:
        # foto yang tidak ikut training sama sekali (mis. masih "perlu dicek manual"): model final pun belum pernah melihatnya
        print(f"{len(missing)} foto tidak ada di fold mana pun -> model --fallback-weights (tidak pernah dilatih dengan foto itu)")
        out.update({p: os.path.expanduser(args.fallback_weights) for p in missing})
        missing = set()
    if missing:
        raise SystemExit(f"foto tidak ada di fold mana pun (bukan out-of-fold): {sorted(missing)[:5]}")
    return out


def predict(weights_for, images, args, label_dir):
    from ultralytics import YOLO
    models, conf = {}, {0: args.conf_box, 1: args.conf_sl}
    for photo, w in sorted(weights_for.items(), key=lambda x: x[1]):
        model = models.setdefault(w, YOLO(w))
        r = model.predict(str(images / f"{photo}.jpg"), imgsz=args.imgsz, conf=min(conf.values()), iou=args.nms_iou,
                          device=args.device, verbose=False)[0]
        lines = []
        for xy, c, s in zip(r.obb.xyxyxyxyn.cpu().numpy(), r.obb.cls.cpu().numpy(), r.obb.conf.cpu().numpy()):
            if s >= conf[int(c)]:
                lines.append(f"{int(c)} " + " ".join(f"{v:.6f}" for v in xy.reshape(-1)))
        (label_dir / f"{photo}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def match_boxes(gt_rows, images, gt_labels, label_dir, iou_min):
    """Per GT row: index (1-based) of the matched predicted box, or None. Greedy by IoU within each photo."""
    result = {}
    for photo in sorted({r["photo"] for r in gt_rows}):
        h, w = cv2.imread(str(images / f"{photo}.jpg")).shape[:2]
        gt_boxes = [p for c, p in read_label_file(gt_labels / f"{photo}.txt", w, h) if c == 0]
        pr_boxes = [p for c, p in read_label_file(label_dir / f"{photo}.txt", w, h) if c == 0]
        wanted = [int(r["box"]) for r in gt_rows if r["photo"] == photo]
        pairs = sorted(((poly_iou(gt_boxes[g - 1], pr_boxes[j]), g, j) for g in wanted for j in range(len(pr_boxes))),
                       reverse=True)
        used_g, used_p = set(), set()
        for iou, g, j in pairs:
            if iou < iou_min or g in used_g or j in used_p:
                continue
            used_g.add(g)
            used_p.add(j)
            result[(photo, g)] = (j + 1, iou)
        for g in wanted:
            result.setdefault((photo, g), (None, max((x for x, gg, _ in pairs if gg == g), default=0.0)))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", default="data/verify_eval_gt.csv")
    ap.add_argument("--images", required=True)
    ap.add_argument("--gt-labels", required=True, help="label YOLO-OBB yang nomor kotaknya dipakai di --gt")
    ap.add_argument("--weights", help="satu model untuk semua foto (BUKAN out-of-fold: skor optimistis)")
    ap.add_argument("--fold-weights", help="template bobot per fold, mis. .../fold{k}/weights/best.pt")
    ap.add_argument("--fold-lists", help="template daftar foto val per fold: folder *.txt atau file nama")
    ap.add_argument("--fallback-weights", help="model untuk foto yang tidak ada di fold mana pun; HANYA benar bila foto itu "
                                               "tidak termasuk data training model tersebut")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--conf-box", type=float, default=0.35)
    ap.add_argument("--conf-sl", type=float, default=0.45)
    ap.add_argument("--nms-iou", type=float, default=0.5)
    ap.add_argument("--iou-match", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="0")
    ap.add_argument("--config", default="config/box_pipeline_config_server.yaml")
    ap.add_argument("--today", default=None)
    ap.add_argument("--ocr-threads", type=int, default=4)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    if not args.weights and not (args.fold_weights and args.fold_lists):
        ap.error("pakai --weights, atau --fold-weights dan --fold-lists")

    images = pathlib.Path(os.path.expanduser(args.images))
    gt_labels = pathlib.Path(os.path.expanduser(args.gt_labels))
    out = (ROOT / args.out_dir) if not os.path.isabs(args.out_dir) else pathlib.Path(args.out_dir)
    label_dir = out / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)

    with open(ROOT / args.gt, newline="", encoding="utf-8") as fh:
        gt_rows = [r for r in csv.DictReader(fh) if (r.get("photo") or "").strip() and not r["photo"].startswith("#")]
    for r in gt_rows:
        r["photo"] = r["photo"].strip()
    photos = {r["photo"] for r in gt_rows}

    predict(model_for_photos(args, photos), images, args, label_dir)
    matched = match_boxes(gt_rows, images, gt_labels, label_dir, args.iou_match)

    fields = list(gt_rows[0].keys())
    with open(out / "gt_terdeteksi.csv", "w", newline="", encoding="utf-8") as fh, \
            open(out / "deteksi.csv", "w", newline="", encoding="utf-8") as fd:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        d = csv.writer(fd)
        d.writerow(["photo", "box_gt", "box_prediksi", "iou", "product", "diameter", "length"])
        for r in gt_rows:
            j, iou = matched[(r["photo"], int(r["box"]))]
            d.writerow([r["photo"], r["box"], j or "", f"{iou:.3f}", r["product"], r["diameter"], r["length"]])
            if j is not None:
                w.writerow(dict(r, box=j))
    found = sum(1 for j, _ in matched.values() if j is not None)
    print(f"Deteksi kotak GT (IoU >= {args.iou_match}): {found}/{len(gt_rows)} "
          f"({found / len(gt_rows):.1%}); terlewat {len(gt_rows) - found} -> {out / 'deteksi.csv'}", flush=True)

    import src.box_ocr as ocr
    base = ocr.SizeTextReader

    class Reader(base):
        def __init__(self, *a, **kw):
            kw["threads"] = args.ocr_threads
            super().__init__(*a, **kw)

    ocr.SizeTextReader = Reader
    sys.argv = ["eval_verify.py", "--config", args.config, "--gt", str(out / "gt_terdeteksi.csv"),
                "--images", str(images), "--labels", str(label_dir), "--out", str(out / "eval_verify.csv")]
    if args.today:
        sys.argv += ["--today", args.today]
    print(f"Verifikasi pada {found} kotak yang terdeteksi (dari {len(gt_rows)} kotak GT):", flush=True)
    runpy.run_path(str(ROOT / "scripts" / "eval_verify.py"), run_name="__main__")


if __name__ == "__main__":
    main()

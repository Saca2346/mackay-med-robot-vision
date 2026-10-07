#!/usr/bin/env python3
"""
Evaluasi pada foto uji yang TIDAK pernah dipakai training: hitung TP, FP, FN per foto,
lalu precision dan recall. Hasilnya tabel yang sama dengan format mentor:
  Foto uji | Kotak asli | Kotak prediksi | TP | FP | FN | Kenapa gagal (diisi manual)

Aturan pencocokan (sama dengan cara menghitung manual):
  - prediksi diurutkan dari confidence tertinggi
  - tiap prediksi dicocokkan ke label asli (kelas sama) yang belum terpakai dengan IoU terbesar
  - IoU >= 0.5 -> TP ; kalau tidak ada -> FP ; label asli yang tersisa -> FN

Script juga menyimpan gambar cek: hijau = TP, merah = FP, kuning = FN (label asli yang
terlewat), putih tipis = label asli. Lihat gambarnya dan koreksi angka kalau ada yang janggal --
itu bagian "dihitung manual" yang diminta mentor.

Contoh:
  python scripts/eval_box_manual.py --images data/uji/images --labels data/uji/labels
  python scripts/eval_box_manual.py --images data/uji/images --labels data/uji/labels --iou 0.5 --conf 0.35
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys
from datetime import datetime

import cv2

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.box_detector import BoxDetector
from src.box_geometry import center, draw_poly, poly_iou, put_label, read_label_file
from src.config import load_config, resolve

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def match_image(preds, gts, iou_thr):
    """preds: [(conf, poly)], gts: [poly] (satu kelas) -> tp_pairs, fp_list, fn_list."""
    used = set()
    tp, fp = [], []
    for conf, p in sorted(preds, key=lambda x: -x[0]):
        best, best_iou = None, 0.0
        for j, g in enumerate(gts):
            if j in used:
                continue
            iou = poly_iou(p, g)
            if iou > best_iou:
                best, best_iou = j, iou
        if best is not None and best_iou >= iou_thr:
            used.add(best)
            tp.append((conf, p, gts[best], best_iou))
        else:
            fp.append((conf, p, best_iou))
    fn = [g for j, g in enumerate(gts) if j not in used]
    return tp, fp, fn


def pr(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/box_pipeline_config.yaml")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True, help="folder .txt label YOLO (OBB 9 angka atau biasa 5 angka)")
    ap.add_argument("--classes", default="data/box_classes.txt")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--conf", type=float, default=None, help="override confidence semua kelas")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(resolve(args.config))
    conf = {k: args.conf for k in cfg["model"]["conf"]} if args.conf is not None else cfg["model"]["conf"]
    weights = resolve(args.weights or cfg["model"]["weights"])
    det = BoxDetector(str(weights), cfg["model"]["task"],
                      cfg["model"]["imgsz"], cfg["model"]["device"], conf, cfg["model"]["iou"], cfg["classes"])
    class_names = [c.strip() for c in resolve(args.classes).read_text().splitlines() if c.strip()]
    out = resolve(args.out or f"results/eval_{datetime.now():%Y%m%d_%H%M}")
    (out / "vis").mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in pathlib.Path(resolve(args.images)).iterdir() if p.suffix.lower() in IMG_EXT)
    if not images:
        sys.exit("Tidak ada foto di --images")
    rows, total = [], {c: [0, 0, 0] for c in class_names}
    for img_path in images:
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        gt_all = read_label_file(pathlib.Path(resolve(args.labels)) / (img_path.stem + ".txt"), w, h)
        dets, _ = det.detect(img)
        vis = img.copy()
        for cid, cname in enumerate(class_names):
            gts = [p for c, p in gt_all if c == cid]
            preds = [(d.conf, d.poly) for d in dets if d.cls == cname]
            tp, fp, fn = match_image(preds, gts, args.iou)
            total[cname][0] += len(tp)
            total[cname][1] += len(fp)
            total[cname][2] += len(fn)
            rows.append({"foto": img_path.name, "kelas": cname, "kotak_asli": len(gts), "kotak_prediksi": len(preds),
                         "TP": len(tp), "FP": len(fp), "FN": len(fn),
                         "IoU_rata2_TP": round(sum(t[3] for t in tp) / len(tp), 3) if tp else "",
                         "kenapa_gagal": ""})
            thick = 2 if cname == "box" else 1
            for g in gts:
                draw_poly(vis, g, (255, 255, 255), 1)
            for c, p, g, iou in tp:
                draw_poly(vis, p, (0, 200, 0), thick)
            for c, p, iou in fp:
                draw_poly(vis, p, (0, 0, 255), thick + 1)
                put_label(vis, f"FP {cname} {c:.2f} IoU{iou:.2f}", center(p), (0, 0, 255), 0.45)
            for g in fn:
                draw_poly(vis, g, (0, 220, 255), thick + 1)
                put_label(vis, f"FN {cname}", center(g), (0, 220, 255), 0.45)
        cv2.imwrite(str(out / "vis" / f"{img_path.stem}_eval.jpg"), vis)

    fields = list(rows[0].keys())
    for cname in class_names:
        sel = [r for r in rows if r["kelas"] == cname]
        with open(out / f"tabel_{cname}.csv", "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=fields)
            wr.writeheader()
            wr.writerows(sel)
            t = total[cname]
            wr.writerow({"foto": "Total", "kelas": cname, "kotak_asli": sum(r["kotak_asli"] for r in sel),
                         "kotak_prediksi": sum(r["kotak_prediksi"] for r in sel), "TP": t[0], "FP": t[1], "FN": t[2]})

    lines = [f"Evaluasi {len(images)} foto uji | IoU >= {args.iou} | model {weights.name}", ""]
    for cname in class_names:
        tp, fp, fn = total[cname]
        p, r, f1 = pr(tp, fp, fn)
        lines += [f"[{cname}]  TP={tp}  FP={fp}  FN={fn}",
                  f"  Precision = TP/(TP+FP) = {tp}/({tp}+{fp}) = {p:.3f}",
                  f"  Recall    = TP/(TP+FN) = {tp}/({tp}+{fn}) = {r:.3f}",
                  f"  F1        = 2PR/(P+R)  = {f1:.3f}", ""]
    tp, fp, fn = (sum(v[i] for v in total.values()) for i in range(3))
    p, r, f1 = pr(tp, fp, fn)
    lines += [f"[gabungan]  TP={tp}  FP={fp}  FN={fn}  Precision={p:.3f}  Recall={r:.3f}  F1={f1:.3f}"]
    (out / "ringkasan.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nTabel  : {out}/tabel_box.csv (isi kolom kenapa_gagal sendiri)\nGambar : {out}/vis/")


if __name__ == "__main__":
    main()

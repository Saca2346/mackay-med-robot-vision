#!/usr/bin/env python3
"""
Evaluasi verifikasi (bukan deteksi) pada foto yang identitas kotaknya sudah diketahui.

Untuk tiap kotak di --gt (poligon dari file label YOLO-OBB, size_label = label kelas 1 yang pusatnya di dalam
kotak itu): bukti dibaca SEKALI (barcode + OCR size_label + OCR seluruh punggung), lalu diputuskan untuk
  - permintaan yang BENAR (identitas kotak itu sendiri)  -> diharapkan MATCH (atau minimal KANDIDAT/CONFIRM)
  - semua identitas LAIN di --gt (produk sama ukuran beda = kasus paling berbahaya, dan produk lain)
                                                         -> TIDAK BOLEH MATCH
Syarat keselamatan: salah MATCH = 0.

Contoh:
  python scripts/eval_verify.py --gt data/verify_eval_gt.csv --images <folder foto> --labels <folder label>
  python scripts/eval_verify.py ... --trust-draft      # anggap katalog draf sudah diverifikasi (HANYA untuk uji)
"""
from __future__ import annotations

import argparse
import collections
import csv
import pathlib
import sys
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.box_catalog import Catalog, norm_product  # noqa: E402
from src.box_detector import BoxGroup, Det  # noqa: E402
from src.box_evidence import gather, gather_full_box, prepare_crops  # noqa: E402
from src.box_geometry import center, point_in_poly, read_label_file  # noqa: E402
from src.box_ocr import SizeTextReader  # noqa: E402
from src.box_verifier import CANDIDATE, CONFIRM, EXPIRED, IGNORED, MATCH, Request, decide_multi  # noqa: E402
from src.config import load_config, resolve  # noqa: E402


def _f(v):
    v = (v or "").strip()
    return float(v) if v else None


def load_gt(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if not (r.get("photo") or "").strip() or r["photo"].startswith("#"):
                continue
            rows.append(dict(photo=r["photo"].strip(), box=int(r["box"]), product=norm_product(r["product"]),
                             diameter=_f(r.get("diameter")), length=_f(r.get("length")), note=r.get("note", "")))
    return rows


def group_for(img, label_file, box_idx):
    h, w = img.shape[:2]
    labels = read_label_file(label_file, w, h)
    boxes = [p for c, p in labels if c == 0]
    box = boxes[box_idx - 1]
    sls = [p for c, p in labels if c == 1 if point_in_poly(center(p), box)]
    return BoxGroup(box=Det("box", 1.0, box), labels=[Det("size_label", 1.0, p) for p in sls])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/box_pipeline_config.yaml")
    ap.add_argument("--gt", default="data/verify_eval_gt.csv")
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--trust-draft", action="store_true", help="pakai entri katalog verified = 0 (uji saja)")
    ap.add_argument("--today", default=None, help="tanggal untuk cek kedaluwarsa (YYYY-MM-DD)")
    ap.add_argument("--out", default="results/eval_verify.csv")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    cfg = load_config(resolve(args.config))
    v = cfg.get("verify", {})
    catalog = Catalog(resolve(v.get("catalog_csv", "data/gtin_catalog.csv")))
    reader = SizeTextReader(cfg["ocr"]["engine"], cfg["ocr"]["retry_flip_below"])
    gt = load_gt(resolve(args.gt))
    idents = sorted({(g["product"], g["diameter"], g["length"]) for g in gt}, key=str)
    today = None
    if args.today:
        import datetime as dt
        today = dt.date.fromisoformat(args.today)
    base = dict(catalog=catalog, today=today, min_score=v.get("min_score", 2))
    modes = {"katalog saat ini (hanya entri terverifikasi)": dict(base, trust_unverified=False)}
    if args.trust_draft:
        modes["katalog draf dianggap terverifikasi (uji)"] = dict(base, trust_unverified=True)
    stats = {m: {"pos": collections.Counter(), "neg": collections.Counter(), "same": collections.Counter(),
                 "wrong": []} for m in modes}
    rows = []
    images = {}
    t0 = time.perf_counter()
    for gi, g in enumerate(gt, 1):
        name = g["photo"] if g["photo"].lower().endswith((".jpg", ".png")) else g["photo"] + ".jpg"
        if name not in images:
            images = {name: cv2.imread(str(pathlib.Path(args.images) / name))}
        img = images[name]
        group = group_for(img, pathlib.Path(args.labels) / (pathlib.Path(name).stem + ".txt"), g["box"])
        sl, box, too_far = prepare_crops(img, group, cfg)
        ev, _ = gather(sl, box, reader, cfg)
        if box is not None:
            gather_full_box(ev, box, reader, cfg)
        true = (g["product"], g["diameter"], g["length"])
        t = ev.text
        row = {"foto": name, "box": g["box"], "identitas": f"{true[0]} {true[1]} x {true[2]}", "catatan": g["note"],
               "teks": "" if t is None else f"{t.product}/{t.diameter}/{t.length} ok={int(t.product_ok)}{int(t.diameter_ok)}{int(t.length_ok)}",
               "ref": " ".join(f"{h.product}:{h.diameter:g}x{h.length:g}" for h in ev.refs),
               "gtin": " ".join(f"{c.gtin}({c.source})" for c in ev.codes)}
        for mi, (mode, kw) in enumerate(modes.items()):
            st = stats[mode]
            d_true, r_true, _ = decide_multi(ev, Request(*true), **kw)
            st["pos"][d_true] += 1
            for ident in idents:
                if ident == true:
                    continue
                d, r, _ = decide_multi(ev, Request(*ident), **kw)
                st["neg"][d] += 1
                if ident[0] == true[0]:
                    st["same"][d] += 1
                if d == MATCH:
                    st["wrong"].append((name, g["box"], true, ident, r))
            row[f"keputusan_{mi}"], row[f"alasan_{mi}"] = d_true, r_true
        rows.append(row)
        if args.verbose:
            print(f"[{gi}/{len(gt)}] {name} b{g['box']} [{true[0]} {true[1]} x {true[2]}] -> "
                  + " / ".join(row[f"keputusan_{i}"] for i in range(len(modes)))
                  + f" | teks {row['teks']} | REF {row['ref'] or '-'} | GTIN {row['gtin'] or '-'}", flush=True)

    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    n = len(gt)
    order = (MATCH, CANDIDATE, CONFIRM, IGNORED, EXPIRED)
    print(f"\nEvaluasi verifikasi: {n} kotak, {len(idents)} identitas | {time.perf_counter() - t0:.0f} s")
    for mi, mode in enumerate(modes):
        st = stats[mode]
        m = sum(st["neg"].values())
        s = sum(st["same"].values())
        print(f"\n[{mi}] {mode}")
        print(f"  permintaan BENAR ({n}): " + ", ".join(f"{k} {st['pos'][k]}" for k in order if st["pos"][k])
              + f"  -> MATCH rate {st['pos'][MATCH] / max(n, 1):.2f}")
        print(f"  permintaan SALAH ({m}): " + ", ".join(f"{k} {st['neg'][k]}" for k in order if st["neg"][k]))
        print(f"    produk sama ukuran beda ({s}): " + ", ".join(f"{k} {st['same'][k]}" for k in order if st["same"][k]))
        print(f"  SALAH MATCH = {len(st['wrong'])}" + ("  <- HARUS 0" if st["wrong"] else "  (syarat keselamatan terpenuhi)"))
        for x in st["wrong"]:
            print("   !!", x)
    print(f"\ntabel: {out}")


if __name__ == "__main__":
    main()

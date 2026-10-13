"""
Detection documentation for photos: one row per detected box (and per size label without a box).

  python scripts/dokumentasi_deteksi_foto.py FOLDER_OR_IMAGE --out results/demo_20261012/foto_baru

Writes in --out:
  deteksi_per_kotak.csv   one row per box: position, size in pixels, angle, confidence, size labels inside
  deteksi.json            the same plus run metadata (model, image size, thresholds, time)
  LAPORAN_DETEKSI.md      readable report (English): counts per photo, table per box, how to read it, limits

Detection only (stage 1): no OCR, no product identification. Nothing is written to the inventory.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time
from datetime import datetime

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.box_detector import BoxDetector, group_labels  # noqa: E402
from src.config import load_config, resolve  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def rect_info(poly) -> dict:
    """Centre, side lengths (long side = box length, short side = box width) and angle of the long side from vertical."""
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)[:4]
    (cx, cy), (a, b), _ = cv2.minAreaRect(pts)
    edges = [pts[(i + 1) % 4] - pts[i] for i in range(4)]
    v = max(edges, key=lambda e: float(np.hypot(e[0], e[1])))     # longest side of the outline
    if v[1] > 0:                                                  # point it upwards (image y grows downwards)
        v = -v
    tilt = float(np.degrees(np.arctan2(v[0], -v[1])))             # 0 = vertical, positive = top leans right
    return {"cx": round(cx, 1), "cy": round(cy, 1), "length_px": round(max(a, b), 1),
            "width_px": round(min(a, b), 1), "tilt_deg": round(tilt, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="image file or folder of images")
    ap.add_argument("--out", default="results/demo_20261012/foto_baru")
    ap.add_argument("--config", default="results/bukti_laporan_20261008/config_realsense.yaml")
    args = ap.parse_args()

    src = pathlib.Path(args.source)
    files = sorted(f for f in src.iterdir() if f.suffix.lower() in IMG_EXT) if src.is_dir() else [src]
    if not files:
        sys.exit(f"No images in {src}")
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_config(resolve(args.config))
    weights = resolve(cfg["model"]["weights"])
    det = BoxDetector(str(weights), cfg["model"]["task"], cfg["model"]["imgsz"], cfg["model"]["device"],
                      cfg["model"]["conf"], cfg["model"]["iou"], cfg["classes"])
    meta = {"time": datetime.now().isoformat(timespec="seconds"), "model": weights.name, "imgsz": list(det.imgsz),
            "device": det.device, "conf_threshold": cfg["model"]["conf"], "iou_threshold": cfg["model"]["iou"],
            "min_label_overlap": cfg["grouping"].get("min_label_overlap", 0.5)}

    rows, photos = [], []
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            print(f"skip (unreadable): {f}")
            continue
        det.detect(img)                                    # warm-up so the timing below is not the first call
        t0 = time.perf_counter()
        dets, _ = det.detect(img)
        ms = (time.perf_counter() - t0) * 1000
        groups, orphans = group_labels(dets, cfg["grouping"].get("min_label_overlap", 0.5))
        groups = sorted(groups, key=lambda g: rect_info(g.box.poly)["cx"])
        photo = {"image": f.name, "height": img.shape[0], "width": img.shape[1], "detect_ms": round(ms, 1),
                 "boxes": len(groups), "size_labels_in_boxes": sum(len(g.labels) for g in groups),
                 "size_labels_without_box": len(orphans), "boxes_without_label": sum(1 for g in groups if not g.labels)}
        photos.append(photo)
        for n, g in enumerate(groups, 1):
            r = rect_info(g.box.poly)
            rows.append({"image": f.name, "no": n, "kind": "box", "conf": round(float(g.box.conf), 3), **r,
                         "size_labels": len(g.labels),
                         "label_conf": ";".join(f"{float(l.conf):.2f}" for l in g.labels)})
        for o in orphans:
            r = rect_info(o.poly)
            rows.append({"image": f.name, "no": "", "kind": "size_label_without_box", "conf": round(float(o.conf), 3),
                         **r, "size_labels": "", "label_conf": ""})
        print(f"{f.name}: {photo['boxes']} boxes, {photo['size_labels_in_boxes']} size labels in boxes, "
              f"{len(orphans)} without a box, {ms:.0f} ms")

    cols = ["image", "no", "kind", "conf", "cx", "cy", "length_px", "width_px", "tilt_deg", "size_labels", "label_conf"]
    with open(out / "deteksi_per_kotak.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    (out / "deteksi.json").write_text(json.dumps({"meta": meta, "photos": photos, "detections": rows}, indent=1,
                                                 ensure_ascii=False), encoding="utf-8")

    md = ["# Detection report (photos)", "",
          f"Generated {meta['time']} | model `{meta['model']}` | input {meta['imgsz'][0]}x{meta['imgsz'][1]} | "
          f"device {meta['device']} | confidence threshold {meta['conf_threshold']}", "",
          "Stage 1 only: finds boxes and size labels. It does **not** read text and does **not** identify products.", ""]
    for p in photos:
        mine = [r for r in rows if r["image"] == p["image"] and r["kind"] == "box"]
        confs = [r["conf"] for r in mine]
        md += [f"## {p['image']} ({p['width']}x{p['height']})", "",
               "| Item | Value |", "| --- | --- |",
               f"| Boxes detected | {p['boxes']} |",
               f"| Size labels inside boxes | {p['size_labels_in_boxes']} |",
               f"| Size labels without a box | {p['size_labels_without_box']} |",
               f"| Boxes without any size label | {p['boxes_without_label']} |",
               f"| Box confidence min / median / max | {min(confs):.2f} / {float(np.median(confs)):.2f} / {max(confs):.2f} |"
               if confs else "| Box confidence | - |",
               f"| Detection time | {p['detect_ms']} ms |", "",
               "| No | Conf | Centre x,y (px) | Length x width (px) | Tilt (deg) | Size labels |",
               "| --- | --- | --- | --- | --- | --- |"]
        md += [f"| {r['no']} | {r['conf']:.2f} | {r['cx']:.0f}, {r['cy']:.0f} | {r['length_px']:.0f} x {r['width_px']:.0f} "
               f"| {r['tilt_deg']:+.1f} | {r['size_labels']} |" for r in mine]
        md.append("")
    md += ["## How to read this", "",
           "- Numbers are ordered left to right by the box centre; they are **not** inventory IDs.",
           "- Tilt 0 = the long side is vertical; positive = leaning to the right.",
           "- A box count can include neighbouring cartons that are not part of the stack of interest; "
           "compare with a hand count before quoting it as an accuracy.",
           "- Low confidence (< 0.5) boxes at the picture border are usually other products or partial boxes.",
           "- This run is generic detection. Product recognition needs the brand in the catalog and its REF rules."]
    (out / "LAPORAN_DETEKSI.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"written: {out / 'deteksi_per_kotak.csv'}, deteksi.json, LAPORAN_DETEKSI.md")


if __name__ == "__main__":
    main()

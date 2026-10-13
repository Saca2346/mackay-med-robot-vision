"""
PROPOSED labels (not reviewed) for the two new photos, in the same format as the labeling project:
YOLO-OBB text files, one object per line: "cls x1 y1 x2 y2 x3 y3 x4 y4" (corners normalised 0-1), classes
0 = box, 1 = size_label (2 = expiry_label is NOT proposed here).

  python scripts/usulan_label_foto.py FOLDER_WITH_PHOTOS --out results/demo_20261012/foto_baru/label_usulan

How the proposal is made (every label says where it came from in REVIEW.csv):
  model        the box / size_label found by the detector (same model as the demo)
  propagated   a size_label the detector missed on a box of the same design: the position of the labels the detector DID
               find (median, in box-local coordinates) is applied to that box
  dropped      detector hit that is clearly wrong (size_label on a logo) or a box that is another product / only a fragment
Boxes of other products and fragments at the photo border are NOT in the label file; they are listed in REVIEW.csv so a
person decides. Nothing here goes to labels_final_obb, the training export or the inventory.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import shutil
import sys

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.box_detector import BoxDetector, group_labels  # noqa: E402
from src.config import load_config, resolve  # noqa: E402

CLS_BOX, CLS_SIZE = 0, 1
IMG_EXT = {".jpg", ".jpeg", ".png"}


def frame_of(poly):
    """Centre, unit long axis (top -> bottom, or left -> right when the box lies flat), unit short axis, length, width."""
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)[:4]
    edges = [pts[(i + 1) % 4] - pts[i] for i in range(4)]
    long_e = max(edges, key=lambda e: float(np.hypot(*e)))
    a = long_e / np.hypot(*long_e)
    if abs(a[0]) > abs(a[1]):                      # lying flat: point to the right
        a = a if a[0] > 0 else -a
    else:                                          # standing: point downwards
        a = a if a[1] > 0 else -a
    b = np.array([-a[1], a[0]], dtype=np.float32)
    c = pts.mean(axis=0)
    rel = pts - c
    return c, a, b, float(np.ptp(rel @ a)), float(np.ptp(rel @ b))


def to_local(poly, fr):
    c, a, b, L, W = fr
    rel = np.asarray(poly, dtype=np.float32).reshape(-1, 2)[:4] - c
    s, t = rel @ a / L + 0.5, rel @ b / W
    return float(s.min()), float(s.max()), float(t.min()), float(t.max())


def from_local(rect, fr):
    c, a, b, L, W = fr
    s0, s1, t0, t1 = rect
    return np.array([c + (s - 0.5) * L * a + t * W * b for s, t in ((s0, t0), (s1, t0), (s1, t1), (s0, t1))],
                    dtype=np.float32)


def line(cls, poly, w, h):
    p = np.asarray(poly, dtype=np.float32).reshape(-1, 2)[:4]
    return f"{cls} " + " ".join(f"{x / w:.6f} {y / h:.6f}" for x, y in p)


def label_kind(r) -> str:
    """Where a detector size_label lies along its box: top logo, name+size cluster, or bottom logo block."""
    s0, s1 = r[0], r[1]
    if s1 < 0.2:
        return "logo at the top (brand only)"
    if s0 > 0.7:
        return "logo block at the bottom (optional, like the TERUMO block)"
    if s1 >= 0.25 and s0 < 0.3:
        return "cluster"
    return "other position"


def classify(name, g, fr, w, h):
    """Which kind of object is this detected box? Photo-specific thresholds (one-off, see the report)."""
    c, a, b, L, W = fr
    flat = abs(a[0]) > abs(a[1])
    if "chroma" in name:
        if flat and L > 0.45 * w:
            return "core", "stacked CHROMA"
        if not flat and L > 500 and 300 <= c[0] <= 1345 and c[1] > 500:
            return "core", "standing CHROMA"
        if L < 150:
            return "exclude", "fragment of a carton cut by the photo border"
        return "exclude", "box of another product (neighbour)"
    # B. Braun photo
    if L < 0.3 * h:
        return "exclude", "only a logo fragment: the box is cut by the photo border, draw it by hand"
    return "core", "B. Braun SeQuent box"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="results/bukti_laporan_20261008/config_realsense.yaml")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    for sub in ("images", "labels", "preview"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    src = pathlib.Path(args.source)
    files = sorted(f for f in src.iterdir() if f.suffix.lower() in IMG_EXT)
    cfg = load_config(resolve(args.config))
    det = BoxDetector(str(resolve(cfg["model"]["weights"])), cfg["model"]["task"], cfg["model"]["imgsz"],
                      cfg["model"]["device"], cfg["model"]["conf"], cfg["model"]["iou"], cfg["classes"])
    review = []
    for f in files:
        img = cv2.imread(str(f))
        h, w = img.shape[:2]
        dets, _ = det.detect(img)
        groups, orphans = group_labels(dets, cfg["grouping"].get("min_label_overlap", 0.5))
        info = []
        for g in groups:
            fr = frame_of(g.box.poly)
            kind, why = classify(f.name, g, fr, w, h)
            info.append({"g": g, "fr": fr, "kind": kind, "why": why})
        core = [i for i in info if i["kind"] == "core"]

        # size labels the detector found inside core boxes, in box-local coordinates (s: 0 = top end, 1 = bottom end).
        # Project rule: size_label = cluster of product name + size; a logo block alone is optional (like the TERUMO
        # block), REF / LOT / date / barcode are never size_label.
        for i in core:
            i["labels"] = [(l, to_local(l.poly, i["fr"])) for l in i["g"].labels]
            i["good"] = [(l, r) for l, r in i["labels"] if label_kind(r) == "cluster"]
            i["bad"] = [(l, r) for l, r in i["labels"] if label_kind(r) != "cluster"]
        groups_kind = {}
        for i in core:                                  # one local template per orientation (standing / flat)
            flat = abs(i["fr"][1][0]) > abs(i["fr"][1][1])
            groups_kind.setdefault(flat, []).append(i)
        template = {}
        for flat, lst in groups_kind.items():
            rects = [max(i["good"], key=lambda x: x[0].conf)[1] for i in lst if i["good"]]
            if rects:
                template[flat] = tuple(float(np.median([r[k] for r in rects])) for k in range(4))
            print(f"{f.name}: {'flat' if flat else 'standing'} boxes {len(lst)}, with a usable detector label "
                  f"{len(rects)}, template (s0,s1,t0,t1) = "
                  f"{tuple(round(v, 3) for v in template[flat]) if flat in template else None}")

        lines, n_prop = [], 0
        for n, i in enumerate(sorted(core, key=lambda i: i["fr"][0][0]), 1):
            lines.append(line(CLS_BOX, i["g"].box.poly, w, h))
            review.append({"image": f.name, "object": "box", "no": n, "origin": "model", "conf": f"{i['g'].box.conf:.2f}",
                           "note": i["why"], "decision": "proposed"})
            flat = abs(i["fr"][1][0]) > abs(i["fr"][1][1])
            if i["good"]:
                l, r = max(i["good"], key=lambda x: x[0].conf)
                lines.append(line(CLS_SIZE, l.poly, w, h))
                review.append({"image": f.name, "object": "size_label", "no": n, "origin": "model",
                               "conf": f"{l.conf:.2f}", "note": "", "decision": "proposed"})
            elif flat in template:
                poly = from_local(template[flat], i["fr"])
                lines.append(line(CLS_SIZE, poly, w, h))
                n_prop += 1
                review.append({"image": f.name, "object": "size_label", "no": n, "origin": "propagated", "conf": "",
                               "note": "detector missed it; position copied from the other boxes of the same design",
                               "decision": "proposed - check the position"})
            for l, r in i["bad"]:
                opt = label_kind(r).startswith("logo block")
                review.append({"image": f.name, "object": "size_label", "no": n, "origin": "model", "conf": f"{l.conf:.2f}",
                               "note": label_kind(r), "decision": "NOT in label file - optional, decide" if opt else "dropped"})
        for i in info:
            if i["kind"] == "exclude":
                c = i["fr"][0]
                review.append({"image": f.name, "object": "box", "no": "", "origin": "model",
                               "conf": f"{i['g'].box.conf:.2f}",
                               "note": f"{i['why']} (centre {c[0]:.0f},{c[1]:.0f})", "decision": "NOT in label file - decide"})
        for o in orphans:
            review.append({"image": f.name, "object": "size_label", "no": "", "origin": "model", "conf": f"{o.conf:.2f}",
                           "note": "size label without a box", "decision": "dropped"})
        stem = f.stem
        shutil.copy(f, out / "images" / f.name)
        (out / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

        vis = img.copy()
        for ln in lines:
            v = ln.split()
            pts = (np.array(v[1:], dtype=np.float32).reshape(4, 2) * [w, h]).astype(np.int32)
            col = (255, 140, 0) if v[0] == "0" else (0, 140, 255)       # box = blue, size_label = orange (BGR)
            cv2.polylines(vis, [pts], True, col, 3 if v[0] == "0" else 2, cv2.LINE_AA)
        cv2.putText(vis, f"PROPOSED labels, NOT reviewed: {sum(l.startswith('0') for l in lines)} box, "
                         f"{sum(l.startswith('1') for l in lines)} size_label ({n_prop} propagated)", (14, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, f"PROPOSED labels, NOT reviewed: {sum(l.startswith('0') for l in lines)} box, "
                         f"{sum(l.startswith('1') for l in lines)} size_label ({n_prop} propagated)", (14, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(out / "preview" / f"{stem}_label_usulan.png"), vis)
        print(f"{f.name}: {sum(l.startswith('0') for l in lines)} box + {sum(l.startswith('1') for l in lines)} size_label "
              f"({n_prop} propagated) -> labels/{stem}.txt")

    (out / "classes.txt").write_text("box\nsize_label\nexpiry_label\n", encoding="utf-8")
    cols = ["image", "object", "no", "origin", "conf", "note", "decision"]
    with open(out / "REVIEW.csv", "w", newline="", encoding="utf-8") as fh:
        w_ = csv.DictWriter(fh, fieldnames=cols)
        w_.writeheader()
        w_.writerows(review)
    print(f"written: {out}")


if __name__ == "__main__":
    main()

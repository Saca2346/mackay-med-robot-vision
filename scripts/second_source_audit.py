"""
Second-source audit: WHY a box gets no PICK from the shelf. For every mapped box of a shelf recording (inventory copy
with `pos` from `rak_inventaris.py peta`) and a sample of frames, the box is read exactly like the live pipeline
(src/box_evidence.read_evidence for the box's own request) and every source is counted: size text, REF, GTIN text,
barcode, printed dates. Contradictions (a source pointing at another product / size) are counted too.

An EXPERIMENTAL reader runs next to it and never takes part in a decision: the barcode block is cut out of the spine
and its human-readable digits (GS1 "(01) GTIN ...") are read enlarged (--hri-scale). It only answers whether a better
crop would make the GTIN text readable at shelf distance; a GTIN counts only with a valid check digit, and it is
compared with the verified GTIN catalog (right / other product).

  python scripts/second_source_audit.py --video results/lock_test_20261016/shelf_scan_clip.mp4 ^
      --inventory results/lock_test_20261016/inventaris_map.csv --frames 12 --out <scratch folder>
-> readings.csv (one row per box and frame) and summary.json / summary.md (per box).
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import pathlib
import sys

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.box_barcode import find_gtins_in_text  # noqa: E402
from src.box_catalog import Catalog  # noqa: E402
from src.box_crops import locate_barcode  # noqa: E402
from src.box_detector import BoxDetector, group_labels  # noqa: E402
from src.box_evidence import prepare_crops, read_evidence  # noqa: E402
from src.box_expiry import find_dates  # noqa: E402
from src.box_ocr import SizeTextReader  # noqa: E402
from src.box_verifier import MATCH, parse_request  # noqa: E402
from src.config import load_config, resolve  # noqa: E402


def request_text(r: dict) -> str:
    if r["diameter"] and r["length"]:
        return f"{r['brand']} {r['product']} {float(r['diameter']):g} x {float(r['length']):g}"
    return f"{r['brand']} {r['product']} {r['expiry']}".strip()


def assign(groups, rows, w: int, h: int) -> dict:
    """Mapped box id -> detected group: the box whose spine centre is nearest to the mapped place (x within half the
    mapped width, y within 0.10 of the frame). Tripod recording; one group per box at most."""
    cent = [np.asarray(g.box.poly, float).reshape(-1, 2).mean(0) / (w, h) for g in groups]
    out, used = {}, set()
    for r in rows:
        x, y, bw = (float(v) for v in r["pos"].split())
        best = None
        for i, c in enumerate(cent):
            if i in used or abs(c[0] - x) > bw / 2 or abs(c[1] - y) > 0.10:
                continue
            d = abs(c[0] - x) + 0.25 * abs(c[1] - y)
            if best is None or d < best[0]:
                best = (d, i)
        if best is not None:
            used.add(best[1])
            out[r["box_id"]] = groups[best[1]]
    return out


def barcode_strip(spine: np.ndarray):
    """Horizontal spine (prepare_crops) -> the part of the spine with its barcode block (src/box_crops.locate_barcode,
    the locator of the PICK panel) over the full spine height, so the digits printed beside the bars are included.
    None if no bar block is found."""
    upright = cv2.rotate(spine, cv2.ROTATE_90_COUNTERCLOCKWISE)          # locate_barcode wants the spine upright
    corners, _ = locate_barcode(upright)
    if corners is None:
        return None
    y0, y1 = int(corners[:, 1].min()), int(corners[:, 1].max())
    pad = int(0.15 * (y1 - y0))
    part = upright[max(0, y0 - pad):min(upright.shape[0], y1 + pad)]
    return cv2.rotate(part, cv2.ROTATE_90_CLOCKWISE)


def read_hri(spine: np.ndarray, reader, scale: float) -> tuple[list, str]:
    strip = barcode_strip(spine)
    if strip is None:
        return [], ""
    big = cv2.resize(strip, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    text = " ".join(t.text for t in reader.read_once(big).tokens)
    return find_gtins_in_text(text), text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--inventory", required=True, help="inventory COPY with pos (rak_inventaris.py peta)")
    ap.add_argument("--config", default="config/box_pipeline_config_realsense.yaml")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--hri-scale", type=float, default=3.0)
    ap.add_argument("--only", default="", help="comma-separated box ids (default: every mapped box)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_config(resolve(a.config))
    cfg_full = copy.deepcopy(cfg)
    cfg_full.setdefault("verify", {})["full_box_min_width_px"] = 0
    det = BoxDetector(str(resolve(a.weights or cfg["model"]["weights"])), cfg["model"]["task"], cfg["model"]["imgsz"],
                      cfg["model"]["device"], cfg["model"]["conf"], cfg["model"]["iou"], cfg["classes"])
    reader = SizeTextReader(cfg["ocr"]["engine"], cfg["ocr"]["retry_flip_below"],
                            backend=cfg["ocr"].get("backend", "onnxruntime"))
    catalog = Catalog(resolve(cfg.get("verify", {}).get("catalog_csv", "data/gtin_catalog.csv")))
    rows = [r for r in csv.DictReader(open(a.inventory, newline="", encoding="utf-8"))
            if r.get("pos") and r.get("status") == "terverifikasi"]
    if a.only:
        keep = set(a.only.split(","))
        rows = [r for r in rows if r["box_id"] in keep]
    reqs = {r["box_id"]: parse_request(request_text(r), cfg.get("catalog")) for r in rows}
    want_gtins = {r["box_id"]: set(catalog.gtins_for(reqs[r["box_id"]].product, reqs[r["box_id"]].diameter,
                                                    reqs[r["box_id"]].length)) for r in rows}

    cap = cv2.VideoCapture(a.video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log = open(out / "readings.csv", "w", newline="", encoding="utf-8")
    wr = csv.writer(log)
    wr.writerow(["frame", "box", "request", "decision", "atoms", "contra", "text_full", "ref_right", "ref_wrong",
                 "gtin_text_right", "gtin_text_other", "barcode_right", "barcode_other", "date_right", "dates",
                 "hri_right", "hri_other", "hri_text", "spine_text", "narrow", "full_ref_right", "full_ref_wrong",
                 "full_gtin_right", "full_match", "full_spine_text"])
    agg = {r["box_id"]: {"request": request_text(r), "catalog_gtin": sorted(want_gtins[r["box_id"]]), "seen": 0,
                         **{k: 0 for k in ("text_full", "ref_right", "ref_wrong", "gtin_text_right", "gtin_text_other",
                                           "barcode_right", "barcode_other", "date_right", "match_single_reading",
                                           "narrow", "full_ref_right", "full_ref_wrong", "full_gtin_right",
                                           "full_gtin_other", "full_match", "hri_right", "hri_other")}} for r in rows}
    for fi in np.linspace(30, n - 30, a.frames).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok:
            continue
        dets, _ = det.detect(frame)
        groups, _ = group_labels(dets, cfg["grouping"].get("min_label_overlap", 0.5))
        h, w = frame.shape[:2]
        for box_id, g in assign(groups, rows, w, h).items():
            req, r = reqs[box_id], next(x for x in rows if x["box_id"] == box_id)
            crops = prepare_crops(frame, g, cfg)
            ev, decision, _, _, _, _ = read_evidence(*crops, reader, cfg, req, catalog)
            atoms, contra = ev.support if ev.support else (frozenset(), False)
            want = want_gtins[box_id]
            text_codes = [c.gtin for c in ev.codes if c.source != "barcode"]
            bar_codes = [c.gtin for c in ev.codes if c.source == "barcode"]
            ref_ok = [x for x in ev.refs if x.product == req.product and (req.diameter is None or (
                abs(x.diameter - req.diameter) < 1e-6 and abs(x.length - req.length) < 1e-6))]
            dates = [d.isoformat() for d in find_dates(ev.spine or "")] if ev.spine else list(map(str, ev.dates))
            # spines narrower than verify.full_box_min_width_px never get the whole-spine OCR live: read them once
            # more WITH it (measurement only) to see whether REF / GTIN text would be there
            narrow = crops[1] is not None and min(crops[1].shape[:2]) < cfg["verify"].get("full_box_min_width_px", 40)
            ev_f, dec_f = (read_evidence(*crops, reader, cfg_full, req, catalog)[:2] if narrow else (ev, decision))
            f_ok = [x for x in ev_f.refs if x.product == req.product and (req.diameter is None or (
                abs(x.diameter - req.diameter) < 1e-6 and abs(x.length - req.length) < 1e-6))]
            f_codes = [c.gtin for c in ev_f.codes if c.source != "barcode"]
            hri, hri_text = read_hri(crops[1], reader, a.hri_scale) if crops[1] is not None else ([], "")
            hri_g = [c.gtin for c in hri]
            row = {"text_full": int({"tp", "td", "tl"} <= set(atoms)), "ref_right": int(bool(ref_ok)),
                   "ref_wrong": int(len(ev.refs) > len(ref_ok)),
                   "gtin_text_right": int(any(c in want for c in text_codes)),
                   "gtin_text_other": int(any(c not in want for c in text_codes)),
                   "barcode_right": int(any(c in want for c in bar_codes)),
                   "barcode_other": int(any(c not in want for c in bar_codes)),
                   "date_right": int(bool(r["expiry"]) and r["expiry"] in " ".join(map(str, ev.dates)) + " ".join(dates)),
                   "match_single_reading": int(decision == MATCH),
                   "narrow": int(narrow), "full_ref_right": int(bool(f_ok)), "full_ref_wrong": int(len(ev_f.refs) > len(f_ok)),
                   "full_gtin_right": int(any(c in want for c in f_codes)),
                   "full_gtin_other": int(any(c not in want for c in f_codes)), "full_match": int(dec_f == MATCH),
                   "hri_right": int(any(c in want for c in hri_g)), "hri_other": int(any(c not in want for c in hri_g))}
            agg[box_id]["seen"] += 1
            for k, v in row.items():
                agg[box_id][k] += v
            wr.writerow([fi, box_id, request_text(r), decision, " ".join(sorted(map(str, atoms))), int(contra),
                         row["text_full"], row["ref_right"], row["ref_wrong"], row["gtin_text_right"],
                         row["gtin_text_other"], row["barcode_right"], row["barcode_other"], row["date_right"],
                         " ".join(dates), row["hri_right"], row["hri_other"], hri_text[:120], (ev.spine or "")[:160],
                         row["narrow"], row["full_ref_right"], row["full_ref_wrong"], row["full_gtin_right"],
                         row["full_match"], (ev_f.spine or "")[:160]])
            log.flush()
        print(f"frame {fi}: {len(groups)} boxes", flush=True)
    log.close()
    json.dump(agg, open(out / "summary.json", "w", encoding="utf-8"), indent=1)
    cols = ["seen", "text_full", "ref_right", "ref_wrong", "gtin_text_right", "gtin_text_other", "barcode_right",
            "date_right", "match_single_reading", "narrow", "full_ref_right", "full_ref_wrong", "full_gtin_right",
            "full_gtin_other", "full_match", "hri_right", "hri_other"]
    lines = ["| box | request | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 2)]
    for b, s in agg.items():
        lines.append(f"| {b} | {s['request']} | " + " | ".join(str(s[c]) for c in cols) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

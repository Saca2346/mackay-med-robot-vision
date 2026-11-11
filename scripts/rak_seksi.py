"""
Shelf sections (meeting 29-10, Sub-project 1): how many fixed sections per shelf level, where the camera stops, and the
section / slot of every box in the position table. Geometry only, src/box_sections.py; nothing here reads a box.

  plan      shelf width + camera distance + FOV -> sections needed, their edges, camera stops, spine width in px
            python scripts/rak_seksi.py plan --shelf 120 --distance 40 --camera c930e --spine 3
  fov       measure the horizontal FOV: something W cm wide spans P px of the frame at D cm
            python scripts/rak_seksi.py fov --width 100 --width-px 1036 --distance 60 --save   (--save: plan uses it from then on)
  card      from a calibration card (scripts/kalibrasi_kamera.py): how much closer the camera has to be for each
            section to fill the frame, and the spine width in px there (barcode check, meeting 29-10)
            python scripts/rak_seksi.py card results/tasks_v2_20261029/task3/kalibrasi/task3_twenty_kalibrasi.json
  assign    write a COPY of the position table with the fixed `seksi` column (never overwrites data/rak/posisi.csv);
            the slot in the section is counted at request time among the boxes still there (src/box_demo.section_slot)
            python scripts/rak_seksi.py assign --sections 2 --columns 20 --out results/sections_20261030/posisi_2.csv

Positions choose where to look and which box to read first; PICK still needs two independent sources.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.box_sections import (CAMERA_HFOV, MARGIN, MAX_SECTIONS, assign_sections, distance_for,  # noqa: E402
                              measure_hfov, plan_sections)


FOV_FILE = ROOT / "results/sections_20261030/fov_measured.json"


def hfov_of(args) -> float:
    if args.hfov:
        return args.hfov
    if FOV_FILE.exists():
        m = json.loads(FOV_FILE.read_text(encoding="utf-8"))
        print(f"FOV {m['hfov_deg']:.1f} deg = measured {m['when']} ({m['note']}); --hfov overrides it")
        return m["hfov_deg"]
    print(f"FOV {CAMERA_HFOV[args.camera]} deg = {args.camera} data sheet, not measured (measure it: fov)")
    return CAMERA_HFOV[args.camera]


def cmd_plan(args):
    hfov = hfov_of(args)
    p = plan_sections(args.shelf, args.distance, hfov, args.frame_px, args.spine, args.margin)
    print(f"camera at {args.distance:g} cm, FOV {hfov:g} deg: sees {p.visible:.1f} cm across the frame "
          f"({p.px_per_cm:.1f} px/cm)")
    print(f"shelf level {args.shelf:g} cm -> {p.sections} section(s) with {args.margin:.0%} free each side"
          + ("" if p.fits else f"  [more than {MAX_SECTIONS}: move the camera back]"))
    for k, ((a, b), c) in enumerate(zip(p.edges, p.centres), 1):
        print(f"  section {k}: {a:g}-{b:g} cm from the left end, camera stop at {c:g} cm")
    print("distance at which each section fills the frame:")
    for n in range(1, MAX_SECTIONS + 1):
        d = distance_for(args.shelf / n, hfov, args.margin)
        q = plan_sections(args.shelf, d, hfov, args.frame_px, args.spine, args.margin, sections=n)
        print(f"  {n} section(s) of {args.shelf / n:.1f} cm: {d:.1f} cm"
              + ("" if q.spine_px is None else f", spine {args.spine:g} cm = {q.spine_px:.0f} px"))


def cmd_fov(args):
    h = measure_hfov(args.width, args.width_px, args.frame_px, args.distance)
    print(f"horizontal FOV {h:.1f} deg ({args.width:g} cm = {args.width_px:g} of {args.frame_px} px at "
          f"{args.distance:g} cm)")
    print(f"data sheet for comparison: {', '.join(f'{k} {v:g}' for k, v in sorted(CAMERA_HFOV.items()))} deg")
    if args.save:
        FOV_FILE.write_text(json.dumps({"hfov_deg": round(h, 2), "when": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
                                        "note": f"{args.width:g} cm = {args.width_px:g} px at {args.distance:g} cm"}, indent=1),
                            encoding="utf-8")
        print(f"saved to {FOV_FILE.relative_to(ROOT)}: plan uses it now (delete the file to go back to the data sheet)")


def cmd_card(args):
    card = json.loads(pathlib.Path(args.card).read_text(encoding="utf-8"))["card"]
    frame = card["frame"][0]
    span = card["edge_right_px"] - card["edge_left_px"]
    print(f"{args.card}: {card['boxes']} boxes over {span:.0f} of {frame} px, spine {card['width_px']:.1f} px "
          f"(narrowest {card['width_min_px']:.1f} px)")
    for n in range(1, MAX_SECTIONS + 1):
        s = (1.0 - 2.0 * args.margin) * frame / (span / n)
        print(f"  {n} section(s), ~{card['boxes'] / n:.0f} boxes each: camera at {1 / s:.2f} x today's distance, "
              f"spine {card['width_px'] * s:.0f} px (narrowest {card['width_min_px'] * s:.0f} px)")


def cmd_assign(args):
    src = pathlib.Path(args.positions)
    with open(src, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for lv in sorted({(r["rak"], r["level"]) for r in rows}):
        level = [r for r in rows if (r["rak"], r["level"]) == lv and r.get("kolom")]
        cols = {r["box_id"]: int(r["kolom"]) for r in level}
        n_cols = args.columns or max(cols.values())
        sec = assign_sections(cols, n_cols, args.sections)
        for r in level:
            r["seksi"], r["_slot"] = map(str, sec[r["box_id"]])
    out = pathlib.Path(args.out)
    if out.resolve() == src.resolve():
        sys.exit("refusing to overwrite the position table: write a copy and let the team check it")
    out.parent.mkdir(parents=True, exist_ok=True)
    names = [k for k in rows[0] if k not in ("seksi", "_slot")] + ["seksi"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=names, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"{out}: {len(rows)} rows, {args.sections} sections per level")
    for r in rows:
        print(f"  {r['box_id']}: rack {r['rak']} level {r['level']} column {r.get('kolom')} -> section "
              f"{r.get('seksi', '-')}, box {r.get('_slot', '-')} from its left end among the rows listed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--shelf", type=float, required=True, help="shelf level width, cm")
    p.add_argument("--distance", type=float, required=True, help="camera lens to box fronts, cm")
    p.add_argument("--camera", choices=sorted(CAMERA_HFOV), default="c930e")
    p.add_argument("--hfov", type=float, help="measured horizontal FOV, deg (overrides --camera)")
    p.add_argument("--spine", type=float, help="spine width of a box, cm")
    p.add_argument("--frame-px", type=int, default=1920)
    p.add_argument("--margin", type=float, default=MARGIN)
    p = sub.add_parser("fov")
    p.add_argument("--width", type=float, required=True, help="real width, cm")
    p.add_argument("--width-px", type=float, required=True, help="the same width in the photo, px")
    p.add_argument("--distance", type=float, required=True, help="lens to that object, cm")
    p.add_argument("--frame-px", type=int, default=1920)
    p.add_argument("--save", action="store_true", help="keep this FOV for plan")
    p = sub.add_parser("card")
    p.add_argument("card")
    p.add_argument("--margin", type=float, default=MARGIN)
    p = sub.add_parser("assign")
    p.add_argument("--positions", default=str(ROOT / "data/rak/posisi.csv"))
    p.add_argument("--sections", type=int, required=True, choices=range(1, MAX_SECTIONS + 1))
    p.add_argument("--columns", type=int, help="positions per level incl. empty ones (default: highest column)")
    p.add_argument("--out", required=True)
    args = ap.parse_args()
    {"plan": cmd_plan, "fov": cmd_fov, "card": cmd_card, "assign": cmd_assign}[args.cmd](args)


if __name__ == "__main__":
    main()

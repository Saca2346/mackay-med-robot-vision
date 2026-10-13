"""
Target card for a photo run of scripts/demo_foto_target.py: the pipeline picture on the left, a panel on the right
that says which box was requested, what the system decided and why.

  python scripts/kartu_target_foto.py RUN_DIR --request "chroma 4.0 x 19"

RUN_DIR is the --out folder of demo_foto_target.py (it holds screens/*_result.png and bacaan_*.csv).
The panel only repeats what the pipeline logged; nothing is decided here.
"""
from __future__ import annotations

import argparse
import csv
import pathlib

import cv2
import numpy as np

W_PANEL = 640
FONT = cv2.FONT_HERSHEY_SIMPLEX


def put(img, text, y, scale=0.62, col=(235, 235, 235), bold=1, x=20):
    cv2.putText(img, text, (x, y), FONT, scale, col, bold, cv2.LINE_AA)
    return y + int(34 * scale / 0.62)


def wrap(text, n):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > n:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    return lines + ([cur] if cur else [])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--request", required=True)
    ap.add_argument("--note", default="", help="one line of context shown at the bottom (e.g. 'close crop of the same photo')")
    args = ap.parse_args()
    run = pathlib.Path(args.run_dir)
    pic = sorted((run / "screens").glob("*_result.png"))[0]
    rows = list(csv.reader(open(next(run.glob("bacaan_*.csv")), encoding="utf-8")))[1:]
    # last decision per track
    last = {}
    for r in rows:
        last[r[1]] = r
    picks = [r for r in last.values() if r[4] == "match"]
    cands = [r for r in last.values() if r[4] == "candidate"]
    img = cv2.imread(str(pic))
    H = max(img.shape[0], 900)
    scale = H / img.shape[0]
    left = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    panel = np.full((left.shape[0], W_PANEL, 3), 28, np.uint8)
    y = put(panel, "REQUESTED", 44, 0.7, (150, 220, 255), 2)
    y = put(panel, args.request, y, 0.95, (255, 255, 255), 2)
    y += 8
    if picks:
        y = put(panel, "RESULT: PICK (2 independent sources)", y, 0.72, (120, 230, 120), 2)
    elif cands:
        y = put(panel, "RESULT: CANDIDATE only (1 source) - not a PICK", y, 0.72, (255, 220, 80), 2)
    else:
        y = put(panel, "RESULT: no candidate", y, 0.72, (120, 160, 255), 2)
    y += 10
    for r in (picks + cands):
        tid, kind, ident, text = r[1], r[4], r[5], r[6]
        f = (ident.split("|") + ["", "", "", "", ""])[:5]
        n_text = sum(bool(x) for x in (f[1], f[2]))          # diameter, length read from the text
        has_ref = bool(f[3])
        text_state = "read (diameter + length)" if n_text == 2 else "PARTLY read (one number only)" if n_text == 1 else "NOT parsed"
        y = put(panel, f"Target box #{tid}  ({kind})", y, 0.7, (255, 255, 255), 2)
        y = put(panel, "Source 1  text (name + size): " + text_state, y, 0.58,
                (120, 230, 120) if n_text == 2 else (255, 200, 90) if n_text == 1 else (120, 160, 255))
        y = put(panel, "Source 2  REF code: " + (f"read -> {f[3]}" if has_ref else "not read"), y, 0.58,
                (120, 230, 120) if has_ref else (120, 160, 255))
        y += 6
        y = put(panel, "Raw OCR of this box (shortened):", y, 0.52, (180, 180, 180))
        for ln in wrap(text.replace("||", " ").strip(), 62)[:5]:
            y = put(panel, ln, y, 0.5, (210, 210, 210))
        y += 12
    y = put(panel, "Rule: PICK (green) needs 2 independent sources.", y + 6, 0.55, (200, 200, 200))
    y = put(panel, "1 source = candidate (blue), the box is shown", y, 0.55, (200, 200, 200))
    y = put(panel, "but the robot would not pick it.", y, 0.55, (200, 200, 200))
    y += 10
    others = len([r for r in last.values() if r[4] == "ignored"])
    y = put(panel, f"{others} other boxes read as a different size (grey).", y, 0.55, (200, 200, 200))
    if args.note:
        yy = panel.shape[0] - 70
        for ln in wrap("Note: " + args.note, 58)[:3]:
            yy = put(panel, ln, yy, 0.5, (160, 190, 230))
    card = np.hstack([left, panel])
    out = run / f"KARTU_TARGET_{pic.stem.replace('_result', '')}.png"
    cv2.imwrite(str(out), card)
    print(out)


if __name__ == "__main__":
    main()

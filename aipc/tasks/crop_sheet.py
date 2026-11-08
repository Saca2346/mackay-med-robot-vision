"""One sheet of the crops every PICK of a run cut from its box (8 Nov 2026, night; aipc_tasks.py --pick-crops 1,
pick_parts.py), drawn on the laptop from the run logs: one row per PICK, the request and PICK time on the left, then
BRAND | DETAIL (size + REF) | BARCODE (located, not decoded), each with what was read. Nothing is cut or decided here.

  venv/Scripts/python.exe aipc/tasks/crop_sheet.py OUT.jpg [--title TEXT] [--cols N] RUN_JSON ...
      RUN_JSON: out/<tag>/<set>_time.json (its median run, as the videos), task3.json / live_<set>.json (every row
      with a PICK), drive.json (every photo with a PICK)
"""
import json
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import evidence_video as ev  # noqa: E402

FONT, AA = cv2.FONT_HERSHEY_SIMPLEX, cv2.LINE_AA
W, ROW, LEFT = 2000, 150, 520
INK, GREY, BG = (30, 30, 30), (110, 110, 110), (255, 255, 255)
NAMES = {"t1_drive": "Task 1, Drive photo", "t1_orig": "Task 1, mentor's photo", "t2_drive": "Task 2, Drive photo"}


def entries(path):
    """(label, request, pick text, base folder, crops info) of every PICK in a run log."""
    path = pathlib.Path(path)
    d = json.load(open(path, encoding="utf-8"))
    if "raw" in d:
        r = ev.median_run(d)
        if r.get("crops"):
            yield NAMES.get(d["set"], d["set"]), d["request"], f"PICK box {r['pick'][0]} in {r['pick'][1]:.2f} s", \
                path.parent, r["crops"]
        return
    s = d["summary"]
    for r in d["rows"]:
        c = r.get("chosen") if "chosen" in r else r
        if not c or not c.get("crops"):
            continue
        if "chosen" in r:
            yield f"Drive {r['photo'][:-4]}"[:34], c["request"], f"PICK in {c['pick_s']:.2f} s", path.parent, c["crops"]
        else:
            where = {"box20": "Live, 20 boxes", "box12": "Live, 12 boxes", "box10close": "Live, close"}.get(
                s.get("set", "").replace("_multi", ""), "Task 3")
            yield f"{where} {r['id']}", r["request"], f"PICK {r['pick_box']} in {r['pick_s']:.2f} s", path.parent, r["crops"]


def tile(base, info, key, w, h):
    out = np.full((h, w, 3), BG, np.uint8)
    f = info.get("files", {}).get(key)
    img = cv2.imread(str(base / info["dir"] / f)) if f else None
    if img is None or not img.size:
        cv2.putText(out, "not found" if key == "barcode" else "not read", (10, h // 2), FONT, 0.7, GREY, 2, AA)
        return out
    if key == "barcode" and img.shape[1] > 3 * img.shape[0]:
        img = cv2.resize(img, (img.shape[1], max(1, img.shape[1] // 3)), interpolation=cv2.INTER_LINEAR)
    s = min((w - 12) / img.shape[1], (h - 12) / img.shape[0])
    img = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))),
                     interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    y, x = (h - img.shape[0]) // 2, 6
    out[y:y + img.shape[0], x:x + img.shape[1]] = img
    return out


def row(label, request, pick, base, info):
    tw = (W - LEFT) // 3
    out = np.full((ROW, W, 3), BG, np.uint8)
    cv2.putText(out, label, (12, 34), FONT, 0.75, INK, 2, AA)
    cv2.putText(out, ev.plain(f"\"{request}\"")[:40], (12, 72), FONT, 0.65, INK, 1, AA)
    cv2.putText(out, pick, (12, 108), FONT, 0.65, ev.GREEN, 2, AA)
    texts = (info.get("brand_text") or "", info.get("detail_text") or "",
             f"located, {info.get('barcode_bars') or 0} bars" if info.get("files", {}).get("barcode") else "")
    for k, key in enumerate(("brand", "detail", "barcode")):
        x = LEFT + k * tw
        out[0:ROW - 30, x:x + tw] = tile(base, info, key, tw, ROW - 30)
        ev.put_fit(out, ev.plain(texts[k]), (x + 6, ROW - 9), tw - 12, 0.55, GREY, 1)
    cv2.line(out, (0, ROW - 1), (W, ROW - 1), (215, 215, 215), 1)
    return out


def main():
    args = sys.argv[1:]
    out = pathlib.Path(args.pop(0))
    title, cols = "", 1
    while args and args[0] in ("--title", "--cols"):
        k = args.pop(0)
        if k == "--title":
            title = args.pop(0)
        else:
            cols = int(args.pop(0))
    rows = [row(*e) for a in args for e in entries(a)]
    head = np.full((110, W, 3), BG, np.uint8)
    cv2.putText(head, title, (12, 40), FONT, 0.9, INK, 2, AA)
    tw = (W - LEFT) // 3
    for k, t in enumerate(("BRAND (as the OCR read it)", "DETAIL: size, REF, GTIN text", "BARCODE: bars only, located")):
        cv2.putText(head, t, (LEFT + k * tw + 6, 92), FONT, 0.7, GREY, 2, AA)
    per = -(-len(rows) // cols)
    blank = np.full((ROW, W, 3), BG, np.uint8)
    heads = [head] + [np.full_like(head, BG) for _ in range(cols - 1)]   # the title once, the column names on each
    for hd in heads[1:]:
        for k, t in enumerate(("BRAND (as the OCR read it)", "DETAIL: size, REF, GTIN text", "BARCODE: bars only, located")):
            cv2.putText(hd, t, (LEFT + k * tw + 6, 92), FONT, 0.7, GREY, 2, AA)
    columns = [np.vstack([heads[c]] + rows[c * per:(c + 1) * per] + [blank] * (per - len(rows[c * per:(c + 1) * per])))
               for c in range(cols)]
    gap = np.full((columns[0].shape[0], 40, 3), BG, np.uint8)
    sheet = columns[0]
    for c in columns[1:]:
        sheet = np.hstack([sheet, gap, c])
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(out, len(rows), "PICKs", flush=True)


if __name__ == "__main__":
    main()

"""Deck pictures of a run (8 Nov 2026), drawn on the laptop from the run's own log: the photo, the request, and ONE green
outline, the outline the run logged for its PICK ("pick_poly"; the box's detected outline when none was logged), with a
full-resolution zoom of that box on the right. Nothing is decided or moved here; same look as `aipc_tasks.py demo`.

  venv/Scripts/python.exe aipc/tasks/pick_still.py OUT_DIR TIME_JSON ... [--task3 TASK3_JSON ID ...]
      [--live LIVE_JSON SET_DIR ID ...]

TIME_JSON: the median-time run of the file (as evidence_video.py). --task3 / --live: the named requests; a live set with
several stills per request is drawn on the still the run read last (the one that gave its PICK).
8 Nov, night: a run made with --pick-crops 1 gets the crops it cut from the PICKed box under the picture (BRAND |
DETAIL | BARCODE, evidence_video.crops_strip), as logged; nothing is cut here.
"""
import glob
import json
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import evidence_video as ev  # noqa: E402

GREEN = (60, 200, 0)


def draw(photo, poly, label, zoom_h=None, crops=None):
    """The photo with the request banner and the green outline (width as `demo`), plus a zoom panel of the box;
    crops = (folder of the run log, its "crops" entry): the crops strip under both."""
    vis = photo.copy()
    q = np.rint(np.array(poly, np.float32)).astype(np.int32)
    cv2.polylines(vis, [q], True, GREEN, max(3, photo.shape[1] // 400), cv2.LINE_AA)
    s = max(1.0, photo.shape[1] / 1600)
    bar = int(56 * s)
    vis = cv2.copyMakeBorder(vis, bar, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    cv2.putText(vis, label, (int(14 * s), int(40 * s)), cv2.FONT_HERSHEY_SIMPLEX, 1.0 * s, (255, 255, 255),
                max(2, int(2 * s)), cv2.LINE_AA)
    # zoom: the box at the photo's own resolution, its outline 1-2 px wide, as tall as the photo part
    p = np.array(poly, np.float32)
    x0, y0 = np.maximum(np.floor(p.min(0)).astype(int) - 16, 0)
    x1, y1 = np.minimum(np.ceil(p.max(0)).astype(int) + 16, [photo.shape[1], photo.shape[0]])
    c = photo[y0:y1, x0:x1].copy()
    cv2.polylines(c, [np.rint(p - [x0, y0]).astype(np.int32)], True, GREEN, max(1, int(round(max(c.shape[:2]) / 700))),
                  cv2.LINE_AA)
    h = vis.shape[0]
    f = min(h / c.shape[0], 0.45 * photo.shape[1] / c.shape[1])
    z = cv2.resize(c, (max(1, int(c.shape[1] * f)), max(1, int(c.shape[0] * f))), interpolation=cv2.INTER_AREA)
    z = cv2.copyMakeBorder(z, 0, h - z.shape[0], 8, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    out = np.hstack([vis, z])
    if crops and crops[1]:
        portrait = photo.shape[0] > photo.shape[1]       # three tiles stacked under a portrait photo
        out = np.vstack([out, ev.crops_strip(crops[0], crops[1], out.shape[1], bg=(20, 20, 20), stacked=portrait)])
    return out


def row_still(out, r, photo, name, base=None):
    inv = {v: int(k) for k, v in r["ids"].items()}
    poly = r.get("pick_poly") or r["polys"][inv[r["pick_box"]]]
    still = f"  (still {r['stills_read']} of {r['stills']})" if r.get("stills", 1) > 1 else ""
    label = f"\"{r['request']}\"  ->  PICK {r['pick_box']} in {r['pick_s']:.2f} s{still}"
    cv2.imwrite(str(out / name), draw(photo, poly, label, crops=(base, r.get("crops"))), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(name, r["pick_box"], r["pick_s"], (r.get("crops") or {}).get("detail_text"), flush=True)


def main():
    args = sys.argv[1:]
    out = pathlib.Path(args.pop(0))
    out.mkdir(parents=True, exist_ok=True)
    while args:
        a = args.pop(0)
        if a == "--task3":
            jf = pathlib.Path(args.pop(0))
            d = json.load(open(jf, encoding="utf-8"))
            base = ev.ROOT / "results/tasks_v2_20261029/task3/live_20261029_155853/cases"
            while args and not args[0].startswith("--") and not args[0].endswith(".json"):
                k = args.pop(0)
                r = next(x for x in d["rows"] if x["id"] == k)
                photo = cv2.imread(sorted(glob.glob(str(base / k / "take_*/photos/*_camera.png")))[0])
                row_still(out, r, photo, f"task3_{k}_pick.jpg", jf.parent)
        elif a == "--live":
            jf = pathlib.Path(args.pop(0))
            d = json.load(open(jf, encoding="utf-8"))
            sd = pathlib.Path(args.pop(0))
            reqs = {x["id"]: x for x in json.load(open(sd / "set.json", encoding="utf-8"))["requests"]}
            while args and not args[0].startswith("--") and not args[0].endswith(".json"):
                k = args.pop(0)
                r = next(x for x in d["rows"] if x["id"] == k)
                stills = reqs[k].get("photos") or [reqs[k]["photo"]]
                photo = cv2.imread(str(sd / stills[r.get("stills_read", 1) - 1]))
                row_still(out, r, photo, f"live_{sd.name}_{k}_pick.jpg", jf.parent)
        else:
            d = json.load(open(a, encoding="utf-8"))
            run = ev.median_run(d)
            photo = cv2.imread(str(ev.ROOT / ev.PHOTOS[d["set"]]))
            poly = run.get("pick_poly") or run["polys"][run["pick"][0]]
            label = f"\"{d['request']}\"  ->  PICK box {run['pick'][0]} in {run['pick'][1]:.2f} s"
            pic = draw(photo, poly, label, crops=(pathlib.Path(a).parent, run.get("crops")))
            cv2.imwrite(str(out / f"{d['set']}_pick.jpg"), pic, [cv2.IMWRITE_JPEG_QUALITY, 90])
            print(d["set"], run["pick"], flush=True)


if __name__ == "__main__":
    main()

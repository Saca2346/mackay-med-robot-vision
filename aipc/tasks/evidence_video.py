"""Evidence videos of the AI PC runs (4 Nov 2026), drawn from the run logs, on the laptop.

  venv/Scripts/python.exe aipc/tasks/evidence_video.py OUT_DIR TIME_JSON ... [--task3 TASK3_JSON]
      [--live LIVE_JSON SET_DIR] [--drive DRIVE_JSON PHOTO_DIR] [--all-colours]

TIME_JSON = out/<tag>/<set>_time.json of `aipc_tasks.py time` (copied from the AI PC): the median-time run of the file is
replayed. Every box outline, every decision and every time on screen comes from that run's log ("events": when each
box's read ended and with which decision; "pick_poly": the edge-corrected outline drawn for the PICK); nothing is decided
here. The clock is the run's own clock (photo read from disk = 0). Each video plays the run at real speed, then the same
run 4x slower. Colours: thin grey = detected, yellow = being read, grey = another identity, blue = one source only or
needs confirmation, green = MATCH (PICK). --task3: one clip per Task 3 request (real speed + 1 s hold), all 41 in a row.
6 Nov (owner): only the green box is drawn, the others are left out (--all-colours = the earlier look), and once the
PICK is made a zoom of the green box at the photo's full resolution is shown on the right, so its edges can be checked.
The mentor's Task 1 photo (t1_orig) is titled "Task 1 (mentor's photo)" (owner, 9 Nov: no separate
WhatsApp name). --live: the replayed live runs (aipc_tasks.py live).
--drive: Leo's Drive photos (aipc_tasks.py drive), one clip per photo with a green box, all in a row.
8 Nov, night (owner): a run made with --pick-crops 1 also logs the crops cut from the PICKed box (pick_parts.py); from
the PICK on, a strip under the photo shows them: BRAND | DETAIL (size + REF) | BARCODE (located, not decoded).
"""
import json
import pathlib
import statistics
import subprocess
import sys

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
FF = str(ROOT / "venv/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe")
PHOTOS = {"t1_drive": "results/drive_photos_20261029/src_photos/main/DCB_BBraun_SeQuentPleaseNEO.jpg",
          "t2_drive": "results/drive_photos_20261029/src_photos/main/BMS_Biosensors_Chroma.jpg",
          "t1_orig": "results/task_20261015/input/task1_braun.jpg"}
FPS, W, BAR = 30, 1280, 96
GREY, YELLOW, BLUE, GREEN = (150, 150, 150), (0, 220, 255), (220, 120, 20), (60, 200, 0)
ALL = "--all-colours" in sys.argv
TITLE = {"t1_drive": "Task 1 (Drive photo)", "t1_orig": "Task 1 (mentor's photo)", "t2_drive": "Task 2 (Drive photo)"}


STRIP_H = 250
FONT, AA = cv2.FONT_HERSHEY_SIMPLEX, cv2.LINE_AA


def plain(s):
    """Text as OpenCV can draw it (ASCII): "SeQuent(R)" for "SeQuent®"."""
    return (s or "").replace("®", "(R)").replace("™", "(TM)").encode("ascii", "ignore").decode()


def crop_tile(base, info, key, title, text, tw, th, u, bg):
    """One tile of the crops strip: title, the crop (scaled to fit, green frame), what was read under it."""
    out = np.full((th, tw, 3), bg, np.uint8)
    cv2.putText(out, title, (int(14 * u), int(30 * u)), FONT, 0.62 * u, (200, 200, 200), max(1, int(1.5 * u)), AA)
    f = info.get("files", {}).get(key)
    img = cv2.imread(str(base / info["dir"] / f)) if f else None
    bw, bh = tw - int(28 * u), th - int(88 * u)
    if img is not None and img.size:
        if key == "barcode" and img.shape[1] > 3 * img.shape[0]:      # a thin bars-only strip shown taller (laptop panel)
            img = cv2.resize(img, (img.shape[1], max(1, img.shape[1] // 3)), interpolation=cv2.INTER_LINEAR)
        s = min(bw / img.shape[1], bh / img.shape[0])
        img = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))),
                         interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
        y, x = int(44 * u) + (bh - img.shape[0]) // 2, int(14 * u) + (bw - img.shape[1]) // 2
        out[y:y + img.shape[0], x:x + img.shape[1]] = img
        cv2.rectangle(out, (x - 2, y - 2), (x + img.shape[1] + 1, y + img.shape[0] + 1), GREEN, max(1, int(2 * u)))
    put_fit(out, plain(text), (int(14 * u), th - int(16 * u)), tw - int(28 * u), 0.62 * u, (255, 255, 255),
            max(1, int(1.5 * u)))
    return out


def put_fit(img, text, org, width, scale, colour, thick):
    """Text in one line no wider than `width`: the font made smaller down to 60 %, then cut with "..."."""
    k = scale
    while k > 0.6 * scale and cv2.getTextSize(text, FONT, k, thick)[0][0] > width:
        k *= 0.95
    while len(text) > 4 and cv2.getTextSize(text, FONT, k, thick)[0][0] > width:
        text = text[:-4] + "..."
    cv2.putText(img, text, org, FONT, k, colour, thick, AA)


def crops_strip(base, info, width, bg=(29, 30, 34), stacked=False):
    """The PICK's crops (aipc_tasks.py --pick-crops 1, pick_parts.py) as three tiles, BRAND | DETAIL | BARCODE, each
    with what was read under it. info = the log's "crops" (None: the strip says no box was cut); base = the folder of
    the run log. Sizes for a 1280 px wide strip, scaled to `width`; stacked: the three tiles one under the other, each
    as wide as the strip (under a portrait photo)."""
    u = width / 1280
    if not info:
        out = np.full((int(STRIP_H * u), width, 3), bg, np.uint8)
        cv2.putText(out, "no PICK: no box cut", (int(16 * u), int(40 * u)), FONT, 0.7 * u, (170, 170, 170),
                    max(1, int(2 * u)), AA)
        return out
    files = info.get("files", {})
    bars = info.get("barcode_bars") or 0
    tiles = (("brand", "BRAND (as the OCR read it)", info.get("brand_text") or "not read"),
             ("detail", "DETAIL: size, REF, GTIN text (OCR)", info.get("detail_text") or "not read"),
             ("barcode", "BARCODE: located, not decoded", f"bars only, {bars} bars" if files.get("barcode") else "not found"))
    if stacked:
        u = width / 426                                  # a tile as wide as one third of a 1280 px strip, scaled up
        return np.vstack([crop_tile(base, info, *t, width, int(180 * u), u, bg) for t in tiles])
    tw = width // 3
    out = np.hstack([crop_tile(base, info, *t, tw, int(STRIP_H * u), u, bg) for t in tiles])
    return cv2.copyMakeBorder(out, 0, 0, 0, width - out.shape[1], cv2.BORDER_CONSTANT, value=bg)


def colour(d):
    return GREEN if d == "match" else BLUE if d in ("candidate", "needs_confirmation") else GREY


def w_h(w):
    return w.h


def fit(img, h):
    """A frame of another height into the clip's W x h, aspect kept (scaled down and/or padded), so outlines are not
    distorted."""
    if img.shape[0] == h:
        return img
    if img.shape[0] > h:
        f = h / img.shape[0]
        img = cv2.resize(img, (int(img.shape[1] * f), h), interpolation=cv2.INTER_AREA)
    out = np.full((h, W, 3), 30, np.uint8)
    out[:img.shape[0], (W - img.shape[1]) // 2:(W - img.shape[1]) // 2 + img.shape[1]] = img
    return out


class Writer:
    def __init__(self, path, h):
        self.h = h
        self.p = subprocess.Popen([FF, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s",
                                   f"{W}x{h}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "23", "-preset",
                                   "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
                                  stdin=subprocess.PIPE)

    def put(self, img):
        self.p.stdin.write(img.tobytes())

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def frames(photo, run, text, slow, hold, label="", cdir=None):
    """Yield the frames of one run: run = {"events", "polys", "pick_poly", "detect_s", "read_photo_s", "skim_s"}.
    cdir (a run made with --pick-crops 1): the folder of its log; the crops strip goes under the photo."""
    s = W / photo.shape[1]
    base = cv2.resize(photo, (W, int(photo.shape[0] * s)), interpolation=cv2.INTER_AREA)
    strip = crops_strip(cdir, run.get("crops"), W) if cdir is not None else None
    h = base.shape[0] + BAR + (strip.shape[0] if strip is not None else 0)
    h += h % 2
    polys = [np.rint(np.array(p) * s).astype(np.int32) + [0, BAR] for p in run["polys"]]
    pick_poly = np.rint(np.array(run["pick_poly"]) * s).astype(np.int32) + [0, BAR] if run.get("pick_poly") else None
    ends = {i: (d, t) for i, d, t, *_ in run["events"]}
    t_det = run["read_photo_s"] + run["detect_s"]
    t_skim = run.get("skim_s") or t_det
    t_end = max([t for _, t in ends.values()] + [t_det])
    pick = min(((t, i) for i, (d, t) in ends.items() if d == "match"), default=None)
    # 8 Nov: the PICK at the time the run logged it (the read's MATCH plus drawing its outline, as in the deck's times)
    ps = run.get("pick_s") or (run["pick"][1] if run.get("pick") else None)
    if pick and ps:
        pick = (max(pick[0], ps), pick[1])
    inset = zoom(photo, run["pick_poly"] if run.get("pick_poly") else run["polys"][pick[1]], base.shape[0]) if pick else None
    # 8 Nov: the zoom goes on the side away from the green box, so it never covers it (portrait photos)
    gx = float(np.mean((pick_poly if pick_poly is not None else polys[pick[1]])[:, 0])) if pick else W
    inset_x = 0 if gx > W / 2 else (W - inset.shape[1] if inset is not None else 0)
    n = int((t_end * slow + hold) * FPS) + 1
    for k in range(n):
        t = min(k / FPS / slow, t_end)
        img = np.full((h, W, 3), 30, np.uint8)
        img[BAR:BAR + base.shape[0]] = base
        for i, p in enumerate(polys):
            if t < t_det:
                continue
            if not ALL:                                   # green only, once its read has ended in MATCH
                if pick and pick[1] == i and pick[0] <= t:
                    cv2.polylines(img, [pick_poly if pick_poly is not None else p], True, GREEN, 3)
                continue
            if i in ends and ends[i][1] <= t:
                d = ends[i][0]
                q = pick_poly if (d == "match" and pick and pick[1] == i and pick_poly is not None) else p
                cv2.polylines(img, [q], True, colour(d), 5 if d == "match" else 2)
            elif t >= t_skim:
                cv2.polylines(img, [p], True, YELLOW, 2)
            else:
                cv2.polylines(img, [p], True, GREY, 1)
        if t < run["read_photo_s"]:
            status = "reading the photo from disk"
        elif t < t_det:
            status = "detecting boxes"
        elif t < t_skim:
            status = f"{len(polys)} boxes: skim (reading order only)"
        else:
            left = sum(1 for i in range(len(polys)) if not (i in ends and ends[i][1] <= t))
            status = f"{len(polys)} boxes read at once, {left} still reading"
        if pick and t >= pick[0]:
            status = f"PICK box {pick[1]} at {pick[0]:.2f} s (two sources)" + (
                f" - other boxes finish at {t_end:.2f} s" if t >= t_end and t_end > pick[0] else "")
        elif t >= t_end:
            status = "no box verified: every box held or another identity"
        if inset is not None and t >= pick[0]:
            img[BAR:BAR + inset.shape[0], inset_x:inset_x + inset.shape[1]] = inset
        if strip is not None:
            y = BAR + base.shape[0]
            if (pick and t >= pick[0]) or (not pick and t >= t_end):
                img[y:y + strip.shape[0]] = strip
            else:
                cv2.putText(img, "brand / detail / barcode: cut from the box when it is PICKed", (16, y + 40),
                            FONT, 0.7, (150, 150, 150), 2, AA)
        cv2.rectangle(img, (0, 0), (W, BAR), (29, 30, 34), -1)
        put_fit(img, plain(text), (16, 36), W - 380, 0.9, (255, 255, 255), 2)   # never under the clock (8 Nov)
        cv2.putText(img, f"t = {t:5.2f} s {label}", (W - 330, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    (255, 255, 255) if slow == 1 else (0, 220, 255), 2)
        cv2.putText(img, status, (16, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    GREEN if pick and t >= pick[0] else (220, 220, 220), 2)
        yield img


def zoom(photo, poly, height):
    """The PICKed box at the photo's own resolution, its green outline 1 px wide, scaled to the video height; at most
    40% of the video width."""
    q = np.array(poly, np.float32)
    x0, y0 = np.maximum(np.floor(q.min(0)).astype(int) - 12, 0)
    x1, y1 = np.minimum(np.ceil(q.max(0)).astype(int) + 12, [photo.shape[1], photo.shape[0]])
    c = photo[y0:y1, x0:x1].copy()
    k = max(1, int(round(max(c.shape[:2]) / 600)))
    cv2.polylines(c, [np.rint(q - [x0, y0]).astype(np.int32)], True, GREEN, k, cv2.LINE_AA)
    f = min(height / c.shape[0], 0.4 * W / c.shape[1])
    z = cv2.resize(c, (max(1, int(c.shape[1] * f)), max(1, int(c.shape[0] * f))), interpolation=cv2.INTER_AREA)
    return cv2.copyMakeBorder(z, 0, 0, 4, 4, cv2.BORDER_CONSTANT, value=(29, 30, 34))


def median_run(d):
    runs = [r for r in d["raw"] if r["pick"]] or d["raw"]
    m = statistics.median_low([r["pick"][1] if r["pick"] else r["all_read_s"] for r in runs])
    return next(r for r in runs if (r["pick"][1] if r["pick"] else r["all_read_s"]) == m)


def task_video(out, tj):
    d = json.load(open(tj, encoding="utf-8"))
    run = median_run(d)
    photo = cv2.imread(str(ROOT / PHOTOS[d["set"]]))
    text = f"{TITLE.get(d['set'], d['set'])}: {d['request']}"
    cdir = pathlib.Path(tj).parent if run.get("crops") else None
    first = next(frames(photo, run, text, 1, 0, cdir=cdir))
    w = Writer(out / f"{d['set']}_run.mp4", first.shape[0])
    last = None
    for slow, label in ((1, "(real speed)"), (4, "(4x slower)")):
        for img in frames(photo, run, text, slow, 2.0, label, cdir):
            w.put(img)
            last = img
    w.close()
    cv2.imwrite(str(out / f"{d['set']}_run.jpg"), last, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(d["set"], "pick", run["pick"], "all", round(run["all_read_s"], 3), flush=True)


def task3_video(out, tj, photos, name="task3_run.mp4"):
    d = json.load(open(tj, encoding="utf-8"))
    w = None
    cdir = pathlib.Path(tj).parent if any(r.get("crops") for r in d["rows"]) else None
    for r in d["rows"]:
        # 8 Nov: a set with several stills per request (*_multi) is read still after still until one gives a PICK; the
        # log's outlines and decisions are those of the still read last, so that still is shown (the clock still runs
        # from the first still)
        p = photos[r["id"]]
        photo = cv2.imread(p[r.get("stills_read", 1) - 1] if isinstance(p, list) else p)
        run = {"events": r["events"], "polys": r["polys"], "pick_poly": r.get("pick_poly"), "read_photo_s": 0.0,
               "detect_s": r["detect_s"], "skim_s": None, "pick_s": r.get("pick_s"), "crops": r.get("crops")}
        truth = "on the shelf" if r["target"] else "not on the shelf"
        title = {"box20": "Live, 20 boxes", "box12": "Live, 12 boxes", "box10close": "Live, close"}.get(
            d["summary"].get("set", "").replace("_multi", ""), "Task 3")
        still = f", still {r['stills_read']} of {r['stills']}" if r.get("stills", 1) > 1 else ""
        text = f"{title} {r['id']}  \"{r['request']}\"  ({truth}{still})"
        for img in frames(photo, run, text, 1, 1.0, cdir=cdir):
            if w is None:
                w = Writer(out / name, img.shape[0])
            w.put(img)
    w.close()
    print("task3", len(d["rows"]), "requests", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    out = pathlib.Path(args.pop(0))
    out.mkdir(parents=True, exist_ok=True)
    if "--task3" in args:
        k = args.index("--task3")
        t3 = args[k + 1]
        del args[k:k + 2]
        import glob
        base = ROOT / "results/tasks_v2_20261029/task3"
        photos = {}
        for c in glob.glob(str(base / "live_20261029_155853/cases/*")) + glob.glob(str(base / "absent_20261029_155853/N*")):
            photos[pathlib.Path(c).name] = sorted(glob.glob(c + "/take_*/photos/*_camera.png"))[0]
        task3_video(out, t3, photos)
    if "--live" in args:
        k = args.index("--live")
        lj, sd = args[k + 1], pathlib.Path(args[k + 2])
        del args[k:k + 3]
        sj = json.load(open(sd / "set.json", encoding="utf-8"))
        task3_video(out, lj, {r["id"]: [str(sd / p) for p in (r.get("photos") or [r["photo"]])] for r in sj["requests"]},
                    f"live_{sd.name}_run.mp4")
    if "--drive" in args:
        k = args.index("--drive")
        dj, pd = args[k + 1], pathlib.Path(args[k + 2])
        del args[k:k + 3]
        d = json.load(open(dj, encoding="utf-8"))
        w = None
        for r in d["rows"]:
            c = r["chosen"]
            if not c or c["pick_box"] is None:
                continue
            run = {"events": c["events"], "polys": c["polys"], "pick_poly": c.get("pick_poly"), "read_photo_s": 0.0,
                   "detect_s": c["detect_s"], "skim_s": None, "pick_s": c.get("pick_s"), "crops": c.get("crops")}
            cdir = pathlib.Path(dj).parent if c.get("crops") else None
            for img in frames(cv2.imread(str(pd / r["photo"])), run, f"Drive {r['photo'][:-4]}: \"{c['request']}\"", 1, 1.5,
                              cdir=cdir):
                if w is None:
                    w = Writer(out / "drive_run.mp4", img.shape[0])
                w.put(fit(img, w_h(w)))
        if w:
            w.close()
    for tj in [a for a in args if a != "--all-colours"]:
        task_video(out, tj)

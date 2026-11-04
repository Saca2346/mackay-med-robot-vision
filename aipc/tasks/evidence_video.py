"""Evidence videos of the AI PC runs (4 Nov 2026), drawn from the run logs, on the laptop.

  venv/Scripts/python.exe aipc/tasks/evidence_video.py OUT_DIR TIME_JSON ... [--task3 TASK3_JSON]

TIME_JSON = out/<tag>/<set>_time.json of `aipc_tasks.py time` (copied from the AI PC): the median-time run of the file is
replayed. Every box outline, every decision and every time on screen comes from that run's log ("events": when each
box's read ended and with which decision; "pick_poly": the edge-corrected outline drawn for the PICK); nothing is decided
here. The clock is the run's own clock (photo read from disk = 0). Each video plays the run at real speed, then the same
run 4x slower. Colours: thin grey = detected, yellow = being read, grey = another identity, blue = one source only or
needs confirmation, green = MATCH (PICK). --task3: one clip per Task 3 request (real speed + 1 s hold), all 41 in a row.
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


def colour(d):
    return GREEN if d == "match" else BLUE if d in ("candidate", "needs_confirmation") else GREY


class Writer:
    def __init__(self, path, h):
        self.p = subprocess.Popen([FF, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s",
                                   f"{W}x{h}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "23", "-preset",
                                   "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
                                  stdin=subprocess.PIPE)

    def put(self, img):
        self.p.stdin.write(img.tobytes())

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def frames(photo, run, text, slow, hold, label=""):
    """Yield the frames of one run: run = {"events", "polys", "pick_poly", "detect_s", "read_photo_s", "skim_s"}."""
    s = W / photo.shape[1]
    base = cv2.resize(photo, (W, int(photo.shape[0] * s)), interpolation=cv2.INTER_AREA)
    h = base.shape[0] + BAR
    h += h % 2
    polys = [np.rint(np.array(p) * s).astype(np.int32) + [0, BAR] for p in run["polys"]]
    pick_poly = np.rint(np.array(run["pick_poly"]) * s).astype(np.int32) + [0, BAR] if run.get("pick_poly") else None
    ends = {i: (d, t) for i, d, t, *_ in run["events"]}
    t_det = run["read_photo_s"] + run["detect_s"]
    t_skim = run.get("skim_s") or t_det
    t_end = max([t for _, t in ends.values()] + [t_det])
    pick = min(((t, i) for i, (d, t) in ends.items() if d == "match"), default=None)
    n = int((t_end * slow + hold) * FPS) + 1
    for k in range(n):
        t = min(k / FPS / slow, t_end)
        img = np.full((h, W, 3), 30, np.uint8)
        img[BAR:BAR + base.shape[0]] = base
        for i, p in enumerate(polys):
            if t < t_det:
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
        cv2.rectangle(img, (0, 0), (W, BAR), (29, 30, 34), -1)
        cv2.putText(img, f"{text}", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(img, f"t = {t:5.2f} s {label}", (W - 330, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    (255, 255, 255) if slow == 1 else (0, 220, 255), 2)
        cv2.putText(img, status, (16, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    GREEN if pick and t >= pick[0] else (220, 220, 220), 2)
        yield img


def median_run(d):
    runs = [r for r in d["raw"] if r["pick"]] or d["raw"]
    m = statistics.median_low([r["pick"][1] if r["pick"] else r["all_read_s"] for r in runs])
    return next(r for r in runs if (r["pick"][1] if r["pick"] else r["all_read_s"]) == m)


def task_video(out, tj):
    d = json.load(open(tj, encoding="utf-8"))
    run = median_run(d)
    photo = cv2.imread(str(ROOT / PHOTOS[d["set"]]))
    first = next(frames(photo, run, d["request"], 1, 0))
    w = Writer(out / f"{d['set']}_run.mp4", first.shape[0])
    last = None
    for slow, label in ((1, "(real speed)"), (4, "(4x slower)")):
        for img in frames(photo, run, d["request"], slow, 2.0, label):
            w.put(img)
            last = img
    w.close()
    cv2.imwrite(str(out / f"{d['set']}_run.jpg"), last, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(d["set"], "pick", run["pick"], "all", round(run["all_read_s"], 3), flush=True)


def task3_video(out, tj, photos):
    d = json.load(open(tj, encoding="utf-8"))
    w = None
    for r in d["rows"]:
        photo = cv2.imread(photos[r["id"]])
        run = {"events": r["events"], "polys": r["polys"], "pick_poly": r.get("pick_poly"), "read_photo_s": 0.0,
               "detect_s": r["detect_s"], "skim_s": None}
        truth = "on the shelf" if r["target"] else "not on the shelf"
        text = f"{r['id']}  \"{r['request']}\"  ({truth})"
        for img in frames(photo, run, text, 1, 1.0):
            if w is None:
                w = Writer(out / "task3_run.mp4", img.shape[0])
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
    for tj in args:
        task_video(out, tj)

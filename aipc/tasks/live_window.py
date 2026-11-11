"""10 Nov 2026: the program's model running live in a window, for the slide-1 recording (documentation, not a result).

Imports aipc_tasks with the final flags of job 103 (same detector, reader and decision code as the timed runs) and
runs a few requests one after another, showing on screen what the model does at each step:
photo -> every box the detector finds (cyan) -> the PICKed box (green) and its crops, or "no box verified".
The clock is the program's own (read_all's t0 at the photo read); the window's drawing time is inside it, so the
times shown are a little slower than the timed runs. Nothing is written outside out/live1110/.

  OCR_ALGO=HEURISTIC PYTHONPATH=../pylib ~/trse_venv/bin/python live_window.py
"""
import os
import sys
import time

FLAGS = ("--imgsz 544x960 --workers 27 --cls-cpu 1 --proc-threads 1 --order skim --snap 1 --refine 1 "
         "--refine-device cuda --draw-imgsz 1280 --variants 1 --lazy 1 --ordered-prefetch 1 --priority 1 "
         "--refine-webcam 0 --webcam-ends 1 --views B:1280:0+B:544x960:0 --alt-views 1 --alt-lazy 1 --skim-name 1 "
         "--split-sizes 1 --spine-size 1 --par-variants 1 --pick-crops 1 --procs 24 --tag live1110").split()

WIN, CW, CH = "AI PC - model running live", 1180, 1080


def main():
    sys.argv = ["aipc_main.py", "live_window"] + FLAGS
    import cv2
    import numpy as np
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
    cv2.moveWindow(WIN, 740, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    c = np.full((CH, CW, 3), (34, 30, 29), np.uint8)
    cv2.putText(c, "loading the detector and 24 OCR processes ...", (24, 96), font, 1.05, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow(WIN, c)
    cv2.waitKey(300)
    print("loading the detector and 24 OCR processes (final flags of job 103) ...", flush=True)
    t_load = time.perf_counter()
    import aipc_tasks as T
    print(f"loaded in {time.perf_counter() - t_load:.1f} s", flush=True)

    t3 = {r["id"]: r for r in T.task3_requests()}
    jobs = [("Task 1 - Drive photo", T.SETS["t1_drive"][0], T.SETS["t1_drive"][1], False, "box on the photo"),
            ("Task 2 - Drive photo", T.SETS["t2_drive"][0], T.SETS["t2_drive"][1], False, "box on the photo"),
            ("Task 3 - shelf, webcam still K04", t3["K04"]["photo"], t3["K04"]["request"], True, "box K04"),
            ("Task 3 - shelf, webcam still K10", t3["K10"]["photo"], t3["K10"]["request"], True, "box K10"),
            ("Task 3 - absent product N01", t3["N01"]["photo"], t3["N01"]["request"], True, "not on the shelf")]

    def canvas(frame, title, line1, line2="", col=(255, 255, 255), polys=(), pick_poly=None, crops=()):
        c = np.full((CH, CW, 3), (34, 30, 29), np.uint8)
        cv2.putText(c, title, (24, 46), font, 1.1, (200, 200, 200), 2, cv2.LINE_AA)
        cv2.putText(c, line1, (24, 96), font, 1.05, col, 2, cv2.LINE_AA)
        if line2:
            cv2.putText(c, line2, (24, 140), font, 0.9, (190, 196, 204), 2, cv2.LINE_AA)
        top, bh = 160, 720 if crops else 900
        s = min((CW - 40) / frame.shape[1], bh / frame.shape[0])
        vis = frame.copy()
        t = max(2, frame.shape[1] // 600)
        for p in polys:
            cv2.polylines(vis, [np.rint(p).astype(np.int32)], True, (255, 210, 0), t, cv2.LINE_AA)
        if pick_poly is not None:
            cv2.polylines(vis, [np.rint(pick_poly).astype(np.int32)], True, (60, 220, 0), 3 * t, cv2.LINE_AA)
        th = cv2.resize(vis, (int(frame.shape[1] * s), int(frame.shape[0] * s)), interpolation=cv2.INTER_AREA)
        x0 = (CW - th.shape[1]) // 2
        c[top:top + th.shape[0], x0:x0 + th.shape[1]] = th
        y0 = top + th.shape[0] + 12
        for j, (label, im) in enumerate(crops[:4]):
            x, y = 24 + (j % 2) * (CW // 2), y0 + (j // 2) * 92
            cv2.putText(c, label, (x, y + 28), font, 0.75, (124, 198, 126), 2, cv2.LINE_AA)
            h = min(82, CH - y - 6)
            if h < 20:
                continue
            sc = min((CW // 2 - 160) / im.shape[1], h / im.shape[0])
            r = cv2.resize(im, (max(1, int(im.shape[1] * sc)), max(1, int(im.shape[0] * sc))), interpolation=cv2.INTER_AREA)
            c[y:y + r.shape[0], x + 120:x + 120 + r.shape[1]] = r
        cv2.imshow(WIN, c)
        cv2.waitKey(1)

    def hold(sec):
        end = time.time() + sec
        while time.time() < end:
            cv2.waitKey(50)

    # warm-up, as every timed run does (the first call builds the GPU kernels); shown, not counted
    f0 = cv2.imread(jobs[0][1])
    canvas(f0, "warm-up", "first call builds the GPU kernels (not counted)")
    T.WEBCAM_FRAME = False
    T._CACHE.clear()
    T.read_all(f0, T.boxes(f0), T.parse_request(jobs[0][2]), time.perf_counter())
    print("warm-up done", flush=True)

    for k, (title, photo, text, webcam, expect) in enumerate(jobs):
        T.WEBCAM_FRAME = webcam
        T._CACHE.clear()
        T.SNAPPED.clear()
        req = T.parse_request(text)
        frame = cv2.imread(photo)
        canvas(frame, title, f'request: "{text}"', f"expected: {expect}")
        hold(2.0)
        t0 = time.perf_counter()                      # the clock starts at the photo read, as in request_once
        frame = cv2.imread(photo)
        groups = T.boxes(frame)
        t_det = time.perf_counter() - t0
        polys = [g.box.poly for g in groups]
        canvas(frame, title, f"{len(groups)} boxes detected in {t_det:.2f} s - reading ...", f'request: "{text}"',
               (255, 210, 0), polys)
        pick, done, _ = T.read_all(frame, groups, req, t0)
        t_all = time.perf_counter() - t0
        greens = sorted(i for i, d, *_ in done if d == T.MATCH)
        print(f"[{k + 1}/{len(jobs)}] {title} | request \"{text}\" | {len(groups)} boxes, detect {t_det:.2f} s | "
              + (f"PICK box {pick[0]} at {pick[1]:.2f} s" if pick else "no box verified")
              + f" | green {greens} | all read {t_all:.2f} s", flush=True)
        crops = []
        if pick:
            info = T.save_parts(f"live{k + 1}", pick)          # after the clock, as in the timed runs
            d = os.path.join(T.OUT, info["crops"]["dir"]) if info else ""
            for name, label in (("product", "product"), ("brand", "brand"), ("detail", "detail"),
                                ("barcode", "barcode")):
                p = os.path.join(d, name + ".png")
                if d and os.path.exists(p):
                    crops.append((label, cv2.imread(p)))
            canvas(frame, title, f"PICK box {pick[0]} in {pick[1]:.2f} s  ({len(groups)} boxes)",
                   f'request: "{text}" - two sources agree', (60, 220, 0), polys,
                   T.SNAPPED.get(pick[0], groups[pick[0]].box.poly), crops)
        else:
            canvas(frame, title, f"no box verified ({len(groups)} boxes read in {t_all:.2f} s)",
                   f'request: "{text}" - expected: {expect}', (120, 160, 255), polys)
        hold(5.0)
    canvas(f0, "done", "5 requests, same program and flags as the final runs")
    hold(2.0)
    for pool in (T.PPOOL, T.SKIMPOOL, T.CROPPOOL):
        if pool:
            pool.shutdown()


if __name__ == "__main__":
    main()

"""
Camera-position calibration before a recording session (RealSense D455 or webcam).

  python scripts/kalibrasi_kamera.py --sesi s2_mid --config results/bukti_laporan_20261008/config_realsense.yaml
  python scripts/kalibrasi_kamera.py --sesi s3_close --ref results/data_rak/s2_mid_kalibrasi.json --same-position

Grabs ~3 s of frames, detects the boxes, and measures what decides whether the session's shelf data is usable:
box width in pixels, sharpness, camera still, camera roll, free margin left/right of the shelf. Prints PASS / WARN / FAIL
per check with advice, saves results/data_rak/<sesi>_kalibrasi.json and an annotated picture, and shows a live preview
while you adjust the camera (q or Enter = measure now, ESC = cancel). Exit code 0 = PASS or WARN, 1 = FAIL.
--ref compares with the card of an earlier session: how far the camera moved (px), the change of distance (scale) and
roll; with --same-position a move beyond 25 px / 8 % / 2 deg is a WARN (use it to repeat a position between sessions).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.box_calib import compare, judge, judge_move, measure, spines, verdict  # noqa: E402
from src.box_camera import open_camera, sharpness  # noqa: E402
from src.box_detector import BoxDetector  # noqa: E402
from src.config import load_config, resolve  # noqa: E402

COL = {"PASS": (60, 200, 90), "WARN": (40, 190, 245), "FAIL": (60, 60, 235), "INFO": (200, 200, 200)}


def grab(cam, seconds: float, fps_cap: float = 8.0):
    """Frames over `seconds`, at most fps_cap per second (we only need a few)."""
    frames, t0, last = [], time.time(), 0.0
    while time.time() - t0 < seconds:
        ok, f = cam.read()
        if not ok:
            continue
        if time.time() - last >= 1.0 / fps_cap:
            frames.append(f)
            last = time.time()
    return frames


def draw(frame, polys, checks, title):
    vis = frame.copy()
    for p in spines(polys):
        cv2.polylines(vis, [np.int32(p)], True, (60, 200, 90), 2)
    h = 26 * (len(checks) + 2)
    cv2.rectangle(vis, (0, 0), (min(vis.shape[1], 900), h), (20, 20, 20), -1)
    cv2.putText(vis, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    for i, c in enumerate(checks):
        cv2.putText(vis, f"{c['status']:<5} {c['check']}: {c['value']}", (10, 52 + 26 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL[c["status"]], 2)
    return vis


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sesi", required=True, help="session name, e.g. s2_mid")
    ap.add_argument("--config", default="config/box_pipeline_config_realsense.yaml")
    ap.add_argument("--ref", default=None, help="calibration card (.json) of an earlier session to compare with")
    ap.add_argument("--same-position", action="store_true", help="warn when the camera moved since --ref")
    ap.add_argument("--out", default="results/data_rak")
    ap.add_argument("--no-window", action="store_true", help="no live preview: measure straight away")
    a = ap.parse_args()

    cfg = load_config(resolve(a.config))
    m = cfg["model"]
    det = BoxDetector(str(resolve(m["weights"])), m["task"], m["imgsz"], m["device"], m["conf"], m["iou"], cfg["classes"])
    cam, idx, size = open_camera(cfg["camera"], None)
    if cam is None:
        print("Cannot open the camera: plug the D455 into a USB 3 port and close other programs using it.")
        return 1
    print(f"camera {idx} {size[0]}x{size[1]}; model {pathlib.Path(m['weights']).name}")

    if not a.no_window:
        win = f"Calibration {a.sesi} - adjust the camera; q / Enter = measure, ESC = cancel"
        while True:
            ok, f = cam.read()
            if not ok:
                continue
            dets, _ = det.detect(f)
            polys = [d.poly for d in dets if d.cls == "box"]
            card = measure(polys, (f.shape[1], f.shape[0]), [sharpness(f)], 0.0)
            card["still_diff"] = 0.0                                     # stillness is only measured at the end
            checks = [c for c in judge(card) if c["check"] != "camera still"]
            cv2.imshow(win, draw(f, polys, checks, f"{a.sesi}: live"))
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 13):
                break
            if k == 27:
                cv2.destroyAllWindows()
                cam.release()
                print("cancelled")
                return 1
        cv2.destroyAllWindows()
        print("hold the camera still for 3 s ...")

    frames = grab(cam, 3.0)
    cam.release()
    if len(frames) < 4:
        print("Too few frames from the camera.")
        return 1
    g = [cv2.cvtColor(cv2.resize(x, (320, 200), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY) for x in frames]
    still = float(np.mean(cv2.absdiff(g[0], g[-1])))
    sharp = [sharpness(x) for x in frames]
    mid = frames[len(frames) // 2]
    dets, _ = det.detect(mid)
    polys = [d.poly for d in dets if d.cls == "box"]
    card = measure(polys, (mid.shape[1], mid.shape[0]), sharp, still)
    checks = judge(card)
    if a.ref:
        ref = json.loads(pathlib.Path(resolve(a.ref)).read_text(encoding="utf-8"))
        cmp = compare(card, ref["card"])
        card["vs_reference"] = {"ref": a.ref, **cmp}
        checks.append(judge_move(cmp, a.same_position))
    v = verdict(checks)

    out = resolve(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().isoformat(timespec="seconds")
    (out / f"{a.sesi}_kalibrasi.json").write_text(
        json.dumps({"session": a.sesi, "time": stamp, "verdict": v, "card": card, "checks": checks}, indent=1),
        encoding="utf-8")
    cv2.imwrite(str(out / f"{a.sesi}_kalibrasi.jpg"), draw(mid, polys, checks, f"{a.sesi}  {stamp}  ->  {v}"),
                [cv2.IMWRITE_JPEG_QUALITY, 92])

    print(f"\nCalibration {a.sesi}: {v}")
    for c in checks:
        print(f"  {c['status']:<5} {c['check']:<28} {c['value']:<34} limit {c['limit']}"
              + (f"\n        -> {c['advice']}" if c["advice"] else ""))
    print(f"card -> {out / (a.sesi + '_kalibrasi.json')}")
    return 1 if v == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())

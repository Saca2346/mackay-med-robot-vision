"""
Camera-position calibration before a recording session (RealSense D455 or webcam).

  python scripts/kalibrasi_kamera.py --sesi s2_mid --config results/bukti_laporan_20261008/config_realsense.yaml
  python scripts/kalibrasi_kamera.py --sesi s3_close --ref results/data_rak/s2_mid_kalibrasi.json --same-position

Live preview while you adjust the camera (Enter / q = measure, ESC = cancel), then a hands-off countdown (5 s) and a 3 s
measurement: box width in pixels, sharpness, camera still (image shift in px, not brightness), camera roll, free margin
left/right of the shelf. When the camera shifted, it measures again by itself (up to 3 times). Prints PASS / WARN / FAIL
per check with advice, saves results/data_rak/<sesi>_kalibrasi.json, an unannotated PNG and an annotated picture.
The preview fits within 1280 x 720 without cropping; measurements use original camera pixels. Titles stay above the
image. These are image checks, not intrinsic lens calibration or proof of barcode decoding. Exit code 0 = PASS or WARN,
1 = FAIL. --ref compares with the card of an earlier session: how far the camera moved (px), the change of distance
(scale) and roll; with --same-position a move beyond 25 px / 8 % / 2 deg is a WARN.

The camera is read in a background thread: the window never waits for the camera, so it does not freeze ("Not
Responding") when the D455 is slow or reconnects; a stalled camera is shown on screen instead.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import pathlib
import sys
import threading
import time

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.box_calib import (STILL_MAX, camera_motion, compare, judge, judge_move, measure,
                          parse_region, region_polygons, spines, verdict)  # noqa: E402
from src.box_camera import open_camera, sharpness  # noqa: E402
from src.box_detector import BoxDetector  # noqa: E402
from src.box_outline import refine_box_outline, inset_display_outline  # noqa: E402
from src.box_preview import fit_preview  # noqa: E402
from src.config import load_config, resolve  # noqa: E402

COL = {"PASS": (60, 200, 90), "WARN": (40, 190, 245), "FAIL": (60, 60, 235), "INFO": (200, 200, 200)}
YELLOW, GREEN, RED = (40, 190, 245), (60, 200, 90), (60, 60, 235)
STALL_S = 3.0          # no new camera frame for this long = show "camera not sending"
MAX_TRIES = 3          # automatic re-measurements when the camera shifted


class Grabber:
    """Reads the camera in a daemon thread and keeps only the newest frame (read() may block for seconds)."""

    def __init__(self, cam):
        self.cam, self.frame, self.n, self.t = cam, None, 0, time.time()
        self._stop = False
        self._lock = threading.Lock()
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def _run(self):
        while not self._stop:
            ok, f = self.cam.read()
            if ok and f is not None:
                with self._lock:
                    self.frame, self.n, self.t = f, self.n + 1, time.time()
            else:
                time.sleep(0.05)

    def latest(self):
        """-> (frame or None, frame number, seconds since that frame)."""
        with self._lock:
            return self.frame, self.n, time.time() - self.t

    def stop(self):
        self._stop = True
        self._th.join(timeout=1.0)
        try:
            self.cam.release()
        except Exception:
            pass


def gray(f):
    return cv2.cvtColor(cv2.resize(f, (320, 200), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)


def banner(frame, big, small, color):
    """Large centred text over the live picture."""
    vis = frame.copy()
    h, w = vis.shape[:2]
    cv2.rectangle(vis, (0, h // 2 - 110), (w, h // 2 + 80), (20, 20, 20), -1)
    for text, scale, y in ((big, 2.2, h // 2 - 20), (small, 0.8, h // 2 + 50)):
        th = 4 if scale > 1 else 2
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
        cv2.putText(vis, text, (max(5, (w - tw) // 2), y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, th)
    return vis


def display_polygons(frame, polys):
    """Use the focused view's edge helper for display; measurements retain raw polygons."""
    ps = spines(polys)
    return [inset_display_outline(refine_box_outline(frame, p, [q for j, q in enumerate(ps) if j != i]))
            for i, p in enumerate(ps)]


def draw(frame, polys, checks, title, display=None, *, live=False, roi=None):
    vis = frame.copy()
    if roi is not None:
        height, width = frame.shape[:2]
        cv2.rectangle(vis, (round(roi[0]*width), round(roi[1]*height)),
                      (round(roi[2]*width), round(roi[3]*height)), (160, 160, 160), 1)
    for p in display_polygons(frame, polys) if display is None else display:
        cv2.polylines(vis, [np.rint(p).astype(np.int32)], True, (255, 220, 30), 1, cv2.LINE_AA)
    h = 26 * (len(checks) + 3)
    if live:
        # Preserve the full camera viewport. Only the panel background is
        # blended; outlined text remains readable over bright carton labels.
        region = vis[:min(h, vis.shape[0]), :min(1050, vis.shape[1])]
        cv2.addWeighted(region, .82, np.full_like(region, 20), .18, 0, dst=region)
    else:
        vis = cv2.copyMakeBorder(vis, h, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))

    def text(value, xy, scale, color):
        if live:
            cv2.putText(vis, value, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (15, 15, 15), 3, cv2.LINE_AA)
        cv2.putText(vis, value, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1 if live else 2, cv2.LINE_AA)

    text(title, (10, 24), 0.7, (255, 255, 255))
    for i, c in enumerate(checks):
        text(f"{c['status']:<5} {c['check']}: {c['value']}", (10, 52 + 26 * i), 0.6, COL[c["status"]])
    text("Image checks only; outlines are detections, not VERIFIED identities.", (10, h - 10), 0.6, (235, 235, 235))
    return vis


class Screen:
    """Window or terminal output; show() always pumps the window events, so the window stays responsive."""

    def __init__(self, title, size):
        self.title, self.size, self.last_print = title, size, ""

    def blank(self):
        return np.zeros((self.size[1], self.size[0], 3), np.uint8)

    def show(self, img, text_for_terminal="", wait_ms=15):
        if self.title:
            cv2.imshow(self.title, fit_preview(img))
            return cv2.waitKey(wait_ms) & 0xFF
        if text_for_terminal and text_for_terminal != self.last_print:
            print(text_for_terminal, flush=True)
            self.last_print = text_for_terminal
        time.sleep(wait_ms / 1000)
        return 255


def stalled(scr, grab, age):
    img = banner(grab.latest()[0] if grab.latest()[0] is not None else scr.blank(), "WAITING FOR CAMERA",
                 f"no new frame for {age:0.0f} s - check the USB 3 cable / port (ESC = cancel)", RED)
    return scr.show(img, f"waiting for the camera ({age:0.0f} s) ...", 50)


def adjust(scr, grab, det, sesi, min_boxes=3, roi=None):
    """Live view until Enter / q (True) or ESC (False)."""
    last_n, vis = -1, None
    while True:
        f, n, age = grab.latest()
        if f is None or age > STALL_S:
            if stalled(scr, grab, age) == 27:
                return False
            continue
        if n != last_n:                                   # detect only on a new frame
            dets, _ = det.detect(f)
            polys = region_polygons([d.poly for d in dets if d.cls == "box"],
                                    (f.shape[1], f.shape[0]), roi)
            card = measure(polys, (f.shape[1], f.shape[0]), [sharpness(f)], 0.0)
            checks = [c for c in judge(card, min_boxes) if c["check"] != "camera still"]
            vis, last_n = draw(f, polys, checks, f"{sesi}: live - Enter = measure, ESC = cancel", live=True, roi=roi), n
        k = scr.show(vis, wait_ms=10)
        if not scr.title or k in (ord("q"), 13):
            return True
        if k == 27:
            return False


def countdown(scr, grab, seconds):
    """Hands-off time: pressing the key shakes the camera. False = ESC."""
    t0 = time.time()
    while (left := seconds - (time.time() - t0)) > 0:
        f, _, age = grab.latest()
        if f is None or age > STALL_S:
            if stalled(scr, grab, age) == 27:
                return False
            continue
        sec = math.ceil(left)
        k = scr.show(banner(f, f"HANDS OFF  {sec}", "do not touch the laptop, camera, cable or shelf - measuring starts at 0",
                            YELLOW), f"  hands off ... {sec}", 30)
        if k == 27:
            return False
    return True


def collect(scr, grab, seconds=3.0, fps_cap=8.0):
    """New frames over `seconds` with a live shift meter. None = ESC or the camera stopped sending."""
    frames, grays, t0, last_n, last_t = [], [], time.time(), -1, 0.0
    while (left := seconds - (time.time() - t0)) > 0:
        f, n, age = grab.latest()
        if f is None or age > STALL_S:
            stalled(scr, grab, age)
            return None
        if n != last_n and time.time() - last_t >= 1.0 / fps_cap:
            frames.append(f)
            grays.append(gray(f))
            last_n, last_t = n, time.time()
        shift, moving = camera_motion([grays[0], grays[-1]], f.shape[1]) if len(grays) > 1 else (0.0, 0.0)
        ok = shift < STILL_MAX
        k = scr.show(banner(f, f"HOLD STILL  {left:0.1f} s",
                            f"camera shift {shift:0.1f} px (must stay < {STILL_MAX:g})" + ("" if ok else "  - CAMERA MOVES"),
                            GREEN if ok else RED), f"  measuring ... {left:0.0f} s", 20)
        if k == 27:
            return None
    return frames if len(frames) >= 4 else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sesi", required=True, help="session name, e.g. r1_s2_mid")
    ap.add_argument("--config", default="config/box_pipeline_config_realsense.yaml")
    ap.add_argument("--ref", default=None, help="calibration card (.json) of an earlier session to compare with")
    ap.add_argument("--same-position", action="store_true", help="warn when the camera moved since --ref")
    ap.add_argument("--out", default="results/data_rak")
    ap.add_argument("--no-window", action="store_true", help="no live preview: measure straight away")
    ap.add_argument("--countdown", type=float, default=5.0, help="hands-off seconds before the 3 s measurement (default 5)")
    ap.add_argument("--min-boxes", type=int, default=3, help="minimum detected cartons; use 1 for a single-carton session")
    ap.add_argument("--roi", type=parse_region, help="optional target area x1,y1,x2,y2 as fractions of the original frame")
    a = ap.parse_args()
    if a.min_boxes < 1:
        ap.error("--min-boxes must be positive")

    out = resolve(a.out)
    paths = [out / f"{a.sesi}{suffix}" for suffix in ("_kalibrasi.json", "_kalibrasi.jpg", "_camera.png")]
    if any(p.exists() for p in paths):
        ap.error("session evidence already exists; choose a new --sesi name to preserve it")
    cfg = load_config(resolve(a.config))
    m = cfg["model"]
    det = BoxDetector(str(resolve(m["weights"])), m["task"], m["imgsz"], m["device"], m["conf"], m["iou"],
                      cfg["classes"], fixed_input=m.get("fixed_input", False))
    cam, idx, size = open_camera(cfg["camera"], None)
    if cam is None:
        print("Cannot open the configured camera: check its index/backend, USB connection and other camera programs.")
        return 1
    print(f"camera {idx} {size[0]}x{size[1]}; model {pathlib.Path(m['weights']).name}")
    grab = Grabber(cam)
    scr = Screen(None if a.no_window else f"Calibration {a.sesi}", size)
    try:
        if not adjust(scr, grab, det, a.sesi, a.min_boxes, a.roi):
            print("cancelled")
            return 1
        frames, shift, moving = None, 0.0, 0.0
        for attempt in range(1, MAX_TRIES + 1):
            if not countdown(scr, grab, a.countdown):
                print("cancelled")
                return 1
            frames = collect(scr, grab)
            if frames is None:
                print("cancelled, or the camera stopped sending frames")
                return 1
            shift, moving = camera_motion([gray(x) for x in frames], frames[0].shape[1])
            if shift < STILL_MAX or attempt == MAX_TRIES:
                break
            print(f"camera shifted {shift:0.1f} px (limit {STILL_MAX:g}): measuring again ({attempt + 1}/{MAX_TRIES})")
            t0 = time.time()
            while time.time() - t0 < 2.0:
                f = grab.latest()[0]
                scr.show(banner(f, f"MOVED  {shift:0.1f} px", f"measuring again ({attempt + 1}/{MAX_TRIES}) - hands off",
                                RED), "", 30)
    finally:
        grab.stop()
        if scr.title:
            cv2.destroyAllWindows()

    sharp = [sharpness(x) for x in frames]
    mid = frames[len(frames) // 2]
    dets, _ = det.detect(mid)
    polys = region_polygons([d.poly for d in dets if d.cls == "box"], (mid.shape[1], mid.shape[0]), a.roi)
    card = measure(polys, (mid.shape[1], mid.shape[0]), sharp, shift, moving)
    checks = judge(card, a.min_boxes)
    wanted = [cfg["camera"].get("width"), cfg["camera"].get("height")]
    actual_size = card["frame"]
    if all(wanted) and actual_size != wanted:
        checks.append({"check": "capture resolution", "value": f"{actual_size[0]}x{actual_size[1]}",
                       "limit": f"requested {wanted[0]}x{wanted[1]}", "status": "WARN",
                       "advice": "the driver returned another resolution; do not compare raw pixel scales"})
    if a.ref:
        ref = json.loads(pathlib.Path(resolve(a.ref)).read_text(encoding="utf-8"))
        cmp = compare(card, ref["card"])
        card["vs_reference"] = {"ref": a.ref, **cmp}
        checks.append(judge_move(cmp, a.same_position))
        if a.same_position:
            previous = ref.get("capture", {})
            current = {"requested": cfg["camera"], "opened_index": idx}
            known = all(k in previous for k in current)
            same = known and all(previous[k] == value for k, value in current.items())
            checks.append({"check": "camera settings vs reference", "value": "same configured camera" if same else
                           "different settings" if known else "reference has no camera provenance",
                           "limit": "same index/backend/settings (device identity still requires operator check)",
                           "status": "PASS" if same else "WARN",
                           "advice": "" if same else "check the physical camera and create a reference with this version"})
    v = verdict(checks)

    out.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().isoformat(timespec="seconds")
    raw_path = out / f"{a.sesi}_camera.png"
    annotated_path = out / f"{a.sesi}_kalibrasi.jpg"
    display = display_polygons(mid, polys)
    if not cv2.imwrite(str(raw_path), mid):
        raise OSError(f"Cannot save raw calibration frame: {raw_path}")
    if not cv2.imwrite(str(annotated_path), draw(mid, polys, checks, f"{a.sesi}  {stamp}  ->  {v}", display, roi=a.roi),
                       [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise OSError(f"Cannot save calibration preview: {annotated_path}")
    (out / f"{a.sesi}_kalibrasi.json").write_text(
        json.dumps({"session": a.sesi, "time": stamp, "verdict": v, "card": card, "checks": checks,
                    "screening": {"min_boxes": a.min_boxes, "roi": a.roi},
                    "capture": {"requested": cfg["camera"], "opened_index": idx, "actual_size": actual_size,
                                "initial_size": list(size),
                                "device_identity": "not established by index; confirm the physical camera"},
                    "config": {"path": str(resolve(a.config)),
                               "sha256": hashlib.sha256(resolve(a.config).read_bytes()).hexdigest(), "model": m},
                    "geometry": {"detector": [np.asarray(p).tolist() for p in polys],
                                 "display": [p.tolist() for p in display],
                                 "use": "measurements use detector geometry; refined outlines are display only"},
                    "images": {"raw": raw_path.name, "annotated": annotated_path.name,
                               "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest()},
                    "scope": "Image/position screening, not lens calibration, barcode validation or latency certification.",
                    "sharpness_scope": "central half of the image; not a per-label OCR/barcode readability measurement"
                    }, indent=1),
        encoding="utf-8")

    print(f"\nCalibration {a.sesi}: {v}")
    for c in checks:
        print(f"  {c['status']:<5} {c['check']:<28} {c['value']:<34} limit {c['limit']}"
              + (f"\n        -> {c['advice']}" if c["advice"] else ""))
    print(f"card -> {out / (a.sesi + '_kalibrasi.json')}")
    print(f"raw frame -> {raw_path}; configured camera {idx}, {actual_size[0]}x{actual_size[1]} (device identity not certified)")
    return 1 if v == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())

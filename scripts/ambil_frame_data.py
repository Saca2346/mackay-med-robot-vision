"""Turn raw shelf recordings (test_webcam_box --record, RealSense D455) into training photos for the box model.

  python scripts/ambil_frame_data.py results/data_rak/*.mp4
  python scripts/ambil_frame_data.py results/data_rak/s1_center.mp4 --every 3 --min-sharp 300 --out ..\\..\\prelabel_size_label_full\\raw_d455

Keeps one frame every --every seconds, and drops a frame when it is
  - blurry (centre sharpness < --min-sharp, same measure as the HUD "sharpness"), or
  - almost the same as the last kept frame (mean grey difference < --min-change): a still shelf adds nothing.
Frames are named D455_YYYYMMDD_HHMMSS_mmm.jpg from the real recording time, so the labeling tools treat one
continuous recording as one burst: it never ends up split between train and val (near-duplicate leak).
Output goes to a staging folder; move the photos into prelabel_size_label_full/raw/ when you want them labelled.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.box_camera import sharpness  # noqa: E402

DEFAULT_OUT = pathlib.Path(__file__).resolve().parents[3] / "prelabel_size_label_full" / "raw_d455"


def times_of(video: pathlib.Path, n_frames: int, fps: float) -> list[float]:
    """Seconds since the start of the recording for every frame (.times.csv from VideoRecorder, else fps)."""
    tc = video.with_name(video.name + ".times.csv")
    if tc.exists():
        rows = [r for r in csv.reader(open(tc)) if r and r[0].isdigit()]
        return [float(r[1]) for r in rows]
    return [i / (fps or 30.0) for i in range(n_frames)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--every", type=float, default=3.0, help="seconds between kept frames (default 3)")
    ap.add_argument("--min-sharp", type=float, default=300.0, help="drop frames less sharp than this (default 300)")
    ap.add_argument("--min-change", type=float, default=4.0,
                    help="drop frames whose mean grey difference to the last kept frame is below this (default 4)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    total = {"kept": 0, "blurry": 0, "same": 0}
    for v in map(pathlib.Path, a.videos):
        cap = cv2.VideoCapture(str(v))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        times = times_of(v, n, cap.get(cv2.CAP_PROP_FPS))
        # the recorder writes the file while recording: its modification time is the END of the recording
        start = dt.datetime.fromtimestamp(v.stat().st_mtime) - dt.timedelta(seconds=times[-1] if times else 0)
        last_t, last_small, stats = -1e9, None, {"kept": 0, "blurry": 0, "same": 0}
        for i, t in enumerate(times):
            ok = cap.grab()
            if not ok:
                break
            if t - last_t < a.every:
                continue
            ok, f = cap.retrieve()
            if not ok:
                break
            if sharpness(f) < a.min_sharp:
                stats["blurry"] += 1
                last_t = t - a.every / 2           # try again a bit later, not a whole step later
                continue
            small = cv2.cvtColor(cv2.resize(f, (320, 200), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            if last_small is not None and float(np.mean(cv2.absdiff(small, last_small))) < a.min_change:
                stats["same"] += 1
                last_t = t
                continue
            stamp = start + dt.timedelta(seconds=t)
            name = f"D455_{stamp:%Y%m%d_%H%M%S}_{stamp.microsecond // 1000:03d}.jpg"
            cv2.imwrite(str(out / name), f, [cv2.IMWRITE_JPEG_QUALITY, 95])
            last_t, last_small = t, small
            stats["kept"] += 1
        cap.release()
        for k in total:
            total[k] += stats[k]
        print(f"{v.name}: kept {stats['kept']}, blurry {stats['blurry']}, unchanged {stats['same']}")
    print(f"total: {total['kept']} photos -> {out}  (dropped: {total['blurry']} blurry, {total['same']} unchanged)")


if __name__ == "__main__":
    main()

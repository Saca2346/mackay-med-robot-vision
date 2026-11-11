"""Where the time of ONE full box read goes on the AI PC (Task 1 box 2, Task 2 box 16): OCR calls, their input sizes,
GPU OCR time inside them, and the rest (Python / OpenCV). Usage: python profile_read.py [--ocr cuda|cpu]"""
import sys
import time

sys.argv += ["--imgsz", "544x960", "--workers", "1", "--tag", "profile"] + ([] if "--algo" in sys.argv else [])
import aipc_tasks as A  # noqa: E402

calls = []
orig = A.SizeTextReader.read_once


def timed(self, img, *a, **k):
    t0 = time.perf_counter()
    r = orig(self, img, *a, **k)
    calls.append((img.shape[:2], time.perf_counter() - t0, len(r.tokens)))
    return r


A.SizeTextReader.read_once = timed
for name, box in (("t1_drive", 2), ("t2_drive", 16)):
    frame = A.cv2.imread(A.SETS[name][0])
    g = A.boxes(frame)[box]
    req = A.parse_request(A.SETS[name][1])
    for rep in range(3):
        calls.clear()
        t0 = time.perf_counter()
        crops = A.prepare_crops(frame, g, A.cfg)
        t1 = time.perf_counter()
        d, why = A.full(crops, req)
        t2 = time.perf_counter()
        ocr = sum(c[1] for c in calls)
        print(f"{name} box {box} run {rep}: {d} | crops {t1 - t0:.3f}s, read {t2 - t1:.3f}s, {len(calls)} OCR calls "
              f"= {ocr:.3f}s, other {t2 - t1 - ocr:.3f}s", flush=True)
    for c in calls:
        print("   ", c[0], f"{c[1] * 1000:.0f} ms", c[2], "tokens")

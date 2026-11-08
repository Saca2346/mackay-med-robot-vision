"""AI PC photo pipeline (4 Nov 2026): the mentor's request on a photo, box detector and reader on the RTX A5000.

  python aipc_tasks.py score SET ... [--det W --imgsz N --workers K --tag T]   every box x every printed size (confusion)
  python aipc_tasks.py time  SET ... [--det W --imgsz N --workers K --runs R]  the mentor's request -> PICK time
  python aipc_tasks.py task3 [--det W --imgsz N --workers K]          the 41 Task 3 requests, each on its own shelf photo
  python aipc_tasks.py live SETDIR ...                                  live webcam runs replayed (first still per take)

Same decision code as the laptop (src/: prepare_crops, read_target_evidence, the two-source rule, data/gtin_catalog.csv
read-only, production config, barcode decoding off as on the laptop photo test). What changes: the detector (any
weights, on the GPU), OCR on the GPU (onnxruntime-gpu, CUDA), and K boxes read at the same time (one OCR engine per
worker thread). Truth: gt_<SET>.csv (printed size read by Claude from the photo, not owner review), keyed by the
laptop's box index; a box of another detector gets the key of the laptop box whose centre (reads.json) lies inside it.
time: clock starts before the photo is read from disk; PICK = the first box whose full read reaches MATCH; every box is
still read to the end and any second MATCH is reported. Output: out/<tag>/<SET>_{score,time}.json.
"""
import csv
import json
import os
import statistics
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(os.path.dirname(H), "code")
sys.path.insert(0, C)
os.chdir(C)
import onnxruntime  # noqa: E402

onnxruntime.preload_dlls()
from scripts.test_webcam_box import drop_wide, group_labels  # noqa: E402
from src import config  # noqa: E402
from src.box_catalog import Catalog  # noqa: E402
from src.box_detector import BoxDetector  # noqa: E402
from src.box_evidence import prepare_crops  # noqa: E402
from src.box_ocr import SizeTextReader  # noqa: E402
from src.box_target_evidence import read_target_evidence  # noqa: E402
from src.box_verifier import MATCH, Request, parse_request  # noqa: E402

T = os.path.join(C, "results/tasks_final_20261102")
DRIVE = os.path.join(C, "results/drive_photos_20261029/src_photos/main")
TASK = os.path.join(C, "results/task_20261015/input")
SETS = {"t1_orig": (f"{TASK}/task1_braun.jpg", "Braun, DCB, 2.75mm, 30mm"),
        "t1_drive": (f"{DRIVE}/DCB_BBraun_SeQuentPleaseNEO.jpg", "Braun, DCB, 2.75mm, 30mm"),
        "t2_orig": (f"{TASK}/task2_chroma.jpg", "Chroma, 2.75 , 33"),
        "t2_drive": (f"{DRIVE}/BMS_Biosensors_Chroma.jpg", "Chroma, 2.75 , 33")}

opts, args, it = {"--det": "", "--imgsz": "960", "--workers": "8", "--tag": "", "--runs": "5", "--ocr": "cuda",
                  "--algo": "HEURISTIC", "--prefetch": "0", "--ocr-workers": "16", "--order": "index", "--procs": "0",
                  "--proc-device": "cuda", "--proc-threads": "2", "--cls-cpu": "0", "--skim-procs": "0", "--variants": "0", "--claude-slots": "0",
                  "--snap": "0", "--refine": "0", "--refine-device": "cpu", "--draw-imgsz": "0", "--lazy": "0", "--barcode": "0",
                  "--ordered-prefetch": "0", "--eager-hits": "2", "--priority": "0", "--views": "", "--split-sizes": "0", "--spine-size": "0", "--refine-webcam": "1", "--alt-views": "0", "--skim-name": "0", "--drive-outlines": "0", "--alt-lazy": "0", "--webcam-ends": "0", "--par-variants": "0", "--pick-crops": "0"}, [], iter(sys.argv[1:])
for a in it:
    if a in opts:
        opts[a] = next(it)
    else:
        args.append(a)
cfg = config.load_config(config.resolve("config/box_pipeline_config_webcam_eleven.yaml"))
# barcode decoding off as in the production config; --barcode 1 = experiment (zxing-cpp on the spine crop, an independent
# source: the decoded GTIN must be an owner-verified catalog row), not a production change
cfg["verify"] = dict(cfg["verify"], decode_barcodes=opts["--barcode"] == "1")
catalog = Catalog(str(config.resolve(cfg["verify"]["catalog_csv"])))
if opts["--split-sizes"] == "1":             # 7 Nov: joined printed sizes "3.533" -> "3.5 33" (size_split.py)
    import size_split  # noqa: E402
    size_split.install()
if opts["--spine-size"] == "1":              # 7 Nov: the printed size on a spine without a detected size label
    import size_split  # noqa: E402
    size_split.install_spine_size(cfg)
_m, _o = cfg["model"], cfg["ocr"]
weights = opts["--det"] or os.path.join(os.path.dirname(H), "..", "models", "box_y26_maxB.pt")
imgsz = [int(v) for v in opts["--imgsz"].split("x")] if "x" in opts["--imgsz"] else int(opts["--imgsz"])
det = BoxDetector(weights, _m["task"], imgsz, "0", _m["conf"], _m["iou"], cfg["classes"],
                  fixed_input=isinstance(imgsz, list))
if opts["--views"]:                     # 7 Nov: several views merged (multiview.py), for tilted and missed boxes
    from multiview import MultiDetector  # noqa: E402
    det = MultiDetector(opts["--views"], weights, _m["task"], _m["conf"], _m["iou"], cfg["classes"])
WORKERS = int(opts["--workers"])
TAG = opts["--tag"] or f"{os.path.basename(os.path.dirname(os.path.dirname(weights))) or 'modelB'}_{opts['--imgsz']}_w{WORKERS}"
OUT = os.path.join(H, "out", TAG)
os.makedirs(OUT, exist_ok=True)
_local = threading.local()
# --algo sets the CUDA provider's cudnn_conv_algo_search (src/box_ocr.py builds RapidOCR itself, so the class is
# wrapped here; laptop code unchanged). Measured 4 Nov (ort_shapes.py): EXHAUSTIVE = HEURISTIC, DEFAULT 2x slower.
import rapidocr  # noqa: E402

_RapidOCR = rapidocr.RapidOCR


class _AlgoRapidOCR(_RapidOCR):
    def __init__(self, params=None, **kw):
        p = dict(params or {})
        if p.get("EngineConfig.onnxruntime.use_cuda"):
            p["EngineConfig.onnxruntime.cuda_ep_cfg.cudnn_conv_algo_search"] = opts["--algo"]
        super().__init__(params=p, **kw)


rapidocr.RapidOCR = _AlgoRapidOCR


def reader():
    if not hasattr(_local, "r"):
        # with --procs every OCR call goes to the processes (_run is the cache), so the thread's own engine is never
        # run: keep it on the CPU instead of holding GPU memory (out of memory on the shared GPU, 4 Nov evening)
        cuda = opts["--ocr"] == "cuda" and not PROCS
        _local.r = SizeTextReader(retry_flip_below=_o.get("retry_flip_below", 0.8), threads=2, use_cuda=cuda,
                                  backend="onnxruntime")
        if opts["--cls-cpu"] == "1" and cuda:
            import ocr_worker
            ocr_worker.cls_on_cpu(_local.r._engine)
    return _local.r


# OCR result cache (--prefetch 1): one RapidOCR call per distinct image. The OCR calls one box read will make are
# started at once on the OCR pool (they run in parallel on the GPU); the unchanged box read code then gets the very
# same results from the cache instead of waiting for each call in turn. Key = the exact image bytes, so a read can only
# get the result of OCR on an identical image: decisions cannot change. Cleared at every timed request.
import hashlib  # noqa: E402
from concurrent.futures import Future  # noqa: E402

from src.box_evidence import FULL_BOX_UPSCALE, FULL_BOX_UPSCALE_BELOW, OCR_MAX_SIDE, _cap, _tiles  # noqa: E402

PROCS = int(opts["--procs"])                 # > 0: OCR in PROCS separate processes (ocr_worker.py), prefetch on
OCR_VARIANT = os.environ.get("OCR_VARIANT", "v6s")   # the OCR engine of the processes (ocr_worker.py); v6s = laptop's
PREFETCH = opts["--prefetch"] == "1" or PROCS > 0
_CACHE, _LOCK, STATS = {}, threading.Lock(), Counter()
_orig_run = SizeTextReader._run
PPOOL = None
if PROCS:
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    import ocr_worker
    from src.box_ocr import OcrToken
    PPOOL = ProcessPoolExecutor(PROCS, mp_context=mp.get_context("spawn"), initializer=ocr_worker.init,
                                initargs=(C, opts["--proc-device"], int(opts["--proc-threads"]), opts["--cls-cpu"] == "1"))
    list(PPOOL.map(ocr_worker.run, [np.full((48, 200, 3), 255, np.uint8)] * PROCS * 3))         # start + warm up
SKIMPOOL = None
if int(opts["--skim-procs"]):                 # the order-only skim on CPU processes, away from the GPU queue
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    import ocr_worker
    SKIMPOOL = ProcessPoolExecutor(int(opts["--skim-procs"]), mp_context=mp.get_context("spawn"),
                                   initializer=ocr_worker.init, initargs=(C, "cpu", 1, False))
    list(SKIMPOOL.map(ocr_worker.run, [np.full((48, 200, 3), 255, np.uint8)] * int(opts["--skim-procs"]) * 3))


class _Dispatcher:
    """--priority 1: OCR calls wait in a priority queue and go to the processes at most 2 per process at a time:
    0 = a call a running read waits for (cache miss), 1 = the other outlines of a box its first outline did not rule
    out, 2 = first-outline calls in skim order. Without it the queue is first in, first out, and the other outlines
    of the likely box wait behind every box's first outline (Task 3: the PICK came last, 5 Nov). Scheduling only:
    every call still runs on the same input."""

    def __init__(self, pool, limit):
        import heapq
        import itertools
        self.heapq, self.pool, self.limit, self.heap, self.n = heapq, pool, limit, [], 0
        self.seq, self.cv = itertools.count(), threading.Condition()
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, img, use_cls, prio):
        from concurrent.futures import Future
        f = Future()
        with self.cv:
            self.heapq.heappush(self.heap, (prio, next(self.seq), img, use_cls, f))
            self.cv.notify_all()
        return f

    def _loop(self):
        while True:
            with self.cv:
                while not self.heap or self.n >= self.limit:
                    self.cv.wait()
                _, _, img, use_cls, f = self.heapq.heappop(self.heap)
                self.n += 1
            self.pool.submit(ocr_worker.run, img, use_cls).add_done_callback(lambda g, f=f: self._done(g, f))

    def _done(self, g, f):
        with self.cv:
            self.n -= 1
            self.cv.notify_all()
        if g.exception() is not None:
            f.set_exception(g.exception())
        else:
            f.set_result(g.result())


DISPATCH = _Dispatcher(PPOOL, 2 * PROCS) if PPOOL and opts["--priority"] == "1" else None
_prio = threading.local()          # priority class of the prefetch this thread is queueing


def _submit(img, use_cls, prio=0):
    """Future of the tokens of one OCR call (process pool, or the thread pool's own engines)."""
    if DISPATCH:
        return DISPATCH.submit(img, use_cls, prio)
    if PPOOL:
        return PPOOL.submit(ocr_worker.run, img, use_cls)
    return OCRPOOL.submit(lambda im: [(t.text, t.conf, t.poly) for t in _orig_run(reader(), im, use_cls)], img)


def _key(img, use_cls):
    a = np.ascontiguousarray(img)
    return a.shape, use_cls, hashlib.blake2b(a.tobytes(), digest_size=16).digest()


def _cached_run(self, img, use_cls=True, count=True):
    k = _key(img, use_cls)
    with _LOCK:
        f = _CACHE.get(k)
        if f is None:
            f = _CACHE[k] = _submit(img, use_cls, 0 if count else getattr(_prio, "p", 2))
            STATS["miss" if count else "prefetched"] += 1
        elif count:
            STATS["hit"] += 1
    if not count:
        return None
    from src.box_ocr import OcrToken
    return [OcrToken(t, c, p) for t, c, p in f.result()]


if PREFETCH:
    SizeTextReader._run = _cached_run

# --pick-crops 1 (8 Nov, owner): the brand, detail and barcode of the PICKed box are cut after its MATCH, inside the PICK
# time (pick_parts.py). Brand and detail come from the OCR inputs and tokens of the outline that gave the MATCH, recorded
# here per read (thread-local; nothing else changes: the same calls, the same results, the same decision).
PICK_CROPS = opts["--pick-crops"] == "1"
_REC = threading.local()
if PICK_CROPS:
    _base_run = SizeTextReader._run

    def _rec_run(self, img, use_cls=True, *a, **kw):
        toks = _base_run(self, img, use_cls, *a, **kw)
        seen = getattr(_REC, "l", None)
        if seen is not None:
            seen.append((img, toks))
        return toks

    SizeTextReader._run = _rec_run


def _rec_start(keep):
    if keep is not None:
        _REC.l = []


def _rec_stop(keep):
    seen = getattr(_REC, "l", None) if keep is not None else None
    _REC.l = None
    return seen


def ocr_inputs(crops):
    """The images a full box read asks OCR for (as logged: size labels in 3 orientations, spine tiles)."""
    sl, box, far = crops
    out = []
    for c in sl:
        out += [c, cv2.rotate(c, cv2.ROTATE_90_COUNTERCLOCKWISE), cv2.rotate(c, cv2.ROTATE_90_CLOCKWISE)]
    if box is not None and not far:
        short = min(box.shape[:2])
        if short < FULL_BOX_UPSCALE_BELOW:
            f = min(cfg.get("verify", {}).get("full_box_upscale", FULL_BOX_UPSCALE), OCR_MAX_SIDE / short)
            spine = cv2.resize(box, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        else:
            spine = _cap(box)
        out += _tiles(spine)
    return out


POOL = ThreadPoolExecutor(WORKERS)
OCRPOOL = ThreadPoolExecutor(int(opts["--ocr-workers"])) if PREFETCH and not PROCS else None
# --par-variants 1 (8 Nov): the outlines of a likely box (skim showed both numbers or the requested name) are read at
# the same time on these threads instead of one after another; the variant rule sees the same results in the same order
PAR_VARIANTS = opts["--par-variants"] == "1"
VPOOL = ThreadPoolExecutor(24) if PAR_VARIANTS else None
for pool in (POOL, OCRPOOL, VPOOL):
    if pool:
        list(pool.map(lambda _: reader().read_once(np.full((48, 200, 3), 255, np.uint8)), range(WORKERS * 2)))
_CACHE.clear()


def prefetch(crops, prio=2):
    _prio.p = prio
    for img in [i for c in (crops if isinstance(crops, list) else [crops]) for i in ocr_inputs(c)]:
        _cached_run(None, img, True, count=False)


def boxes(frame):
    dets = det.detect(frame)[0]
    groups, _ = group_labels(dets, cfg["grouping"]["min_label_overlap"])
    groups = drop_wide(groups)
    if ALT_VIEWS:                    # 8 Nov, --alt-views 1: the other views' outlines of the same box (multiview.py)
        from multiview import attach_alts
        attach_alts(groups, getattr(dets, "alt", ()), cfg["grouping"]["min_label_overlap"])
    return sorted(groups, key=lambda g: tuple(np.round(g.box.poly.mean(axis=0)[::-1] / 150)))


def keys_of(name, groups):
    """laptop key of each detected box (by the laptop box centre inside it), None if no laptop box is inside."""
    reads = json.load(open(os.path.join(T, name, "reads.json")))
    out = []
    for g in groups:
        poly = g.box.poly.astype(np.float32)
        inside = [(cv2.pointPolygonTest(poly, tuple(map(float, r["center"])), True), r["key"]) for r in reads]
        inside = [x for x in inside if x[0] >= 0]
        out.append(max(inside)[1] if inside else None)
    return out


VARIANTS = [(0.0, 0.0), (0.06, 0.0), (-0.06, 0.0), (0.0, 0.02)] if opts["--variants"] == "1" else [(0.0, 0.0)]
LAZY = opts["--lazy"] == "1"   # skip the other outlines of a box whose first outline reads another identity
ORDERED = opts["--ordered-prefetch"] == "1"
ALT_VIEWS = opts["--alt-views"] == "1" and bool(opts["--views"])   # 8 Nov: + the other views' outlines of a box
EAGER_HITS = int(opts["--eager-hits"])   # --lazy 1: all 4 outlines prefetched for boxes whose skim shows >= this many
                                        # requested numbers (2 = both; speed only)


def variant_crops(frame, g):
    """--variants 1: the box read from 4 outlines (as detected, sides 6% wider, 6% narrower, ends 2% longer), because
    the second source turned out to depend on a few pixels of outline (4 Nov: OpenVINO vs .pt outlines of model B
    gave Task 1 TP 6 vs 4 / 3). Size-label outlines unchanged. The first outline's crops are made at once, the other
    three only when a read or a prefetch asks for them (same crops, later: the skim and the boxes the first outline
    already rules out no longer pay for them)."""
    return Outlines(frame, g)


class Outlines(list):
    """The crops of the 4 outlines of one box; items 1-3 are made on first access (score mode shares one box's
    Outlines between the threads of its sizes, hence the lock)."""

    def __init__(self, frame, g):
        super().__init__([prepare_crops(frame, g, cfg)])
        self._frame, self._g, self._lock = frame, g, threading.Lock()
        self._n = len(VARIANTS) * (1 + len(getattr(g, "alts", ())))   # --alt-views 1: + the other views' outlines
        if os.environ.get("AIPC_EAGER_CROPS") == "1":       # the earlier behaviour, for the speed A/B only
            self._fill()

    def _fill(self):
        with self._lock:
            if self._g is not None:
                t = time.perf_counter()
                self.extend(_other_outlines(self._frame, self._g))
                with _STAGE_LOCK:
                    STAGE["crop_more_s"] = STAGE.get("crop_more_s", 0.0) + time.perf_counter() - t
                    STAGE["crop_more_boxes"] = STAGE.get("crop_more_boxes", 0) + 1
                self._g = None

    def __getitem__(self, k):
        if k != 0:
            self._fill()
        return super().__getitem__(k)

    def __iter__(self):
        self._fill()
        return super().__iter__()

    def __len__(self):
        return self._n


def _other_outlines(frame, g):
    out = _variants(frame, g)
    for a in getattr(g, "alts", ()):     # --alt-views 1: the box as another detector view outlined it, with its labels,
        out.append(prepare_crops(frame, a, cfg))         # and the same 3 variants of that outline (8 Nov: rd_vb read
        out += _variants(frame, a)                       # t1_drive:3 and t2_drive:13 with them, rd_alta without did not)
    return out


def _variants(frame, g):
    from copy import copy
    out = []
    for fw, fl in VARIANTS:
        if fw == 0 and fl == 0:
            continue
        p = g.box.poly.astype(np.float64)
        c = p.mean(0)
        e1, e2 = p[1] - p[0], p[2] - p[1]
        lg = e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2
        u = lg / np.linalg.norm(lg)
        v = np.array([-u[1], u[0]])
        q = c + np.outer((p - c) @ u, u) * (1 + fl) + np.outer((p - c) @ v, v) * (1 + fw)
        gv = copy(g)
        gv.box = copy(g.box)
        gv.box.poly = q.astype(np.float32)
        out.append(prepare_crops(frame, gv, cfg))
    return out


def full(crops, req, par=False, keep=None):
    """crops = one prepare_crops result, or a list of them (outline variants). Variants: MATCH only if one variant
    reaches MATCH and no variant reads a conflicting identity (ignored / expired); then 'needs_confirmation'.
    par (--par-variants 1, likely boxes only): the same reads at the same time, then the same rule on the same list.
    keep (--pick-crops 1): a dict that gets the OCR inputs and tokens ("seen") of the outline whose read gave the MATCH."""
    many = crops if isinstance(crops, list) else [crops]
    res, recs = [], []
    n0 = min(len(VARIANTS), len(many)) if ALT_LAZY else len(many)
    if par and VPOOL is not None and len(many) > 1:
        def rd(k):
            _rec_start(keep)
            ev, d, reason, *_ = read_target_evidence(*many[k], reader(), cfg, req, catalog, attempt=3,
                                                    identity_verified=True)
            return d, reason, _rec_stop(keep)
        for k in range(1, n0):                          # the outline crops made here, one thread (Outlines is lazy)
            many[k]
        if PREFETCH:
            prefetch(many[1:n0], 1)
        out = list(VPOOL.map(rd, range(n0)))
        res, recs = [(d, r) for d, r, _ in out], [x for *_, x in out]
        if LAZY and res[0][0] in ("ignored", "expired"):    # as the sequential read: a conflict on the first outline
            return res[0]
        ds0 = [d for d, _ in res]
        if n0 < len(many) and not (MATCH in ds0 or any(d in ("ignored", "expired") for d in ds0)):
            for k in range(n0, len(many)):
                many[k]
            if PREFETCH:
                prefetch(many[n0:], 1)
            out = list(VPOOL.map(rd, range(n0, len(many))))
            res, recs = res + [(d, r) for d, r, _ in out], recs + [x for *_, x in out]
    for k in range(len(many) if not res else 0):                          # by index: Outlines makes items 1-3 only when reached
        if k == 1 and LAZY:
            if res[0][0] in ("ignored", "expired"):     # --lazy 1: a conflict on the first outline already rules out
                return res[0]                           # MATCH, so the other outlines cannot change the cell
            if PREFETCH:
                prefetch(many[1:n0], 1)
        if k == n0:                                     # --alt-lazy 1 (8 Nov): the other view's outlines only when the
            ds0 = [d for d, _ in res]                   # box's own 4 give no MATCH and no conflict; a MATCH there is
            if MATCH in ds0 or any(d in ("ignored", "expired") for d in ds0):   # decided by the 4-outline rule of 7 Nov
                break
            if PREFETCH:
                prefetch(many[n0:], 1)
        _rec_start(keep)
        ev, d, reason, *_ = read_target_evidence(*many[k], reader(), cfg, req, catalog, attempt=3, identity_verified=True)
        res.append((d, reason))
        recs.append(_rec_stop(keep))
    if len(res) == 1:
        if keep is not None and res[0][0] == MATCH:
            keep["seen"] = recs[0]
        return res[0]
    ds = [d for d, _ in res]
    if MATCH in ds:
        if any(d in ("ignored", "expired") for d in ds):
            return "needs_confirmation", "outline variants disagree: " + "; ".join(f"{d}: {r}" for d, r in res)
        j = ds.index(MATCH)
        if keep is not None:
            keep["seen"] = recs[j]
        return res[j]
    return res[0]


def score(name):
    frame = cv2.imread(SETS[name][0])
    groups = boxes(frame)
    keys = keys_of(name, groups)
    gt = {r["key"]: r for r in csv.DictReader(open(os.path.join(T, f"gt_{name}.csv"), encoding="utf-8"))}
    product = parse_request(SETS[name][1]).product
    truth = {k: (r["product"], float(r["diameter_mm"] or 0), float(r["length_mm"] or 0))
             for k, r in gt.items() if r["product"] != "unscored"}
    sizes = sorted({(d, l) for p, d, l in truth.values() if p == product})
    jobs = {}
    for i, g in enumerate(groups):
        if keys[i] in truth:
            crops = variant_crops(frame, g) if len(VARIANTS) > 1 else prepare_crops(frame, g, cfg)
            for size in sizes:
                jobs[POOL.submit(full, crops, Request(product, *size))] = (i, keys[i], size)
    cells, pairs = Counter(), []
    for f in as_completed(jobs):
        i, key, size = jobs[f]
        d, reason = f.result()
        pos = truth[key] == (product, *size)
        cell = ("TP" if d == MATCH else "FN") if pos else ("FP" if d == MATCH else "TN")
        cells[cell] += 1
        pairs.append({"box": i, "key": key, "printed": "%s %g x %g" % truth[key], "asked": "%s %g x %g" % (product, *size),
                      "decision": d, "cell": cell, "reason": reason})
    missed = sorted(set(truth) - set(keys))
    res = {"set": name, "detector": weights, "imgsz": opts["--imgsz"], "boxes_detected": len(groups),
           "boxes_scored": len({p["key"] for p in pairs}), "truth_boxes": len(truth),
           "ocr_variant": OCR_VARIANT, "views": opts["--views"], "alt_views": ALT_VIEWS, "skim_name": SKIM_NAME, "alt_lazy": ALT_LAZY, "par_variants": PAR_VARIANTS,
           "truth_boxes_not_detected": missed, "detected_without_truth": sum(k is None for k in keys),
           "pairs": len(pairs), "cells": dict(cells)}
    json.dump({"summary": res, "pairs": sorted(pairs, key=lambda p: (p["box"], p["asked"]))},
              open(os.path.join(OUT, f"{name}_score.json"), "w"), indent=1)
    print(json.dumps(res), flush=True)


def hits_of(text, req):
    import re
    nums = {float(n.replace(",", ".")) for n in re.findall(r"(?<![\d.,])\d{1,2}(?:[.,]\d{1,2})?(?![\d.,]*\d)", text)}
    for a, b, c in re.findall(r"(?<![\d.,])(\d)[.,](\d{1,2})(\d{2})(?![\d.,]*\d)", text):
        nums |= {float(f"{a}.{b}"), float(c)}
    return (req.diameter in nums) + (req.length in nums)


def label_hits(crops, req):
    """Scheduling only: the size-label crops read in the first direction SizeTextReader.read tries (that OCR call is
    the first one of the box's full read, so it is reused from the cache, not wasted); spine skim if no label."""
    sl, box, far = crops
    if not sl:
        return quick_hits(box, req)
    text = []
    for c in sl:
        img = cv2.rotate(c, cv2.ROTATE_90_COUNTERCLOCKWISE) if c.shape[0] > 1.5 * c.shape[1] else c
        text += [t.text for t in reader().read_once(img).tokens if t.conf >= 0.5]
    return hits_of(" ".join(text), req)


def skim_image(box_crop):
    s = min(1.0, 640 / max(box_crop.shape[:2]))
    return cv2.resize(box_crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else box_crop


def quick_hits(box_crop, req):
    """Scheduling only, never evidence (same rule as photo_tasks.quick_hits on the laptop): how many requested size
    numbers one cheap OCR pass of the whole spine at 640 px shows."""
    import re
    if box_crop is None:
        return 0
    if SKIMPOOL:
        toks = SKIMPOOL.submit(ocr_worker.run, skim_image(box_crop)).result()
        return hits_of(" ".join(t for t, c, _ in toks if c >= 0.5), req)
    text = " ".join(t.text for t in reader().read_once(skim_image(box_crop)).tokens if t.conf >= 0.5)
    nums = {float(n.replace(",", ".")) for n in re.findall(r"(?<![\d.,])\d{1,2}(?:[.,]\d{1,2})?(?![\d.,]*\d)", text)}
    for a, b, c in re.findall(r"(?<![\d.,])(\d)[.,](\d{1,2})(\d{2})(?![\d.,]*\d)", text):
        nums |= {float(f"{a}.{b}"), float(c)}
    return (req.diameter in nums) + (req.length in nums)


SKIM_NAME = opts["--skim-name"] == "1"
ALT_LAZY = opts["--alt-lazy"] == "1"


def skim_text(box_crop):
    if box_crop is None:
        return ""
    if SKIMPOOL:
        return " ".join(t for t, c, _ in SKIMPOOL.submit(ocr_worker.run, skim_image(box_crop)).result() if c >= 0.5)
    return " ".join(t.text for t in reader().read_once(skim_image(box_crop)).tokens if t.conf >= 0.5)


def name_hit(text, req):
    """Scheduling only, never evidence: 1 if a word (>= 4 letters) of the requested product name is in the skim text."""
    import re
    words = [w for w in re.findall(r"[a-z]+", (req.product or "").lower()) if len(w) >= 4]
    low = text.lower()
    return int(any(w in low for w in words))


def skim_both(crops, req):
    """--skim-name 1 (8 Nov): one skim pass -> (requested numbers seen, requested product name seen). Order only:
    boxes showing the product name go first, then by numbers (Task 3: the Accuforce boxes' sizes are small on a dark
    band and the 640 px skim missed them, so they were read 19th of 20)."""
    text = skim_text(crops[1])
    return hits_of(text, req), name_hit(text, req)


def read_all(frame, groups, req, t0):
    """Every box read for the request at once (POOL); --prefetch 1: its OCR calls started first on OCRPOOL;
    --order skim: boxes whose one-pass skim shows more requested numbers go first (order only). -> pick, done, t_skim"""
    SNAPPED.clear()
    STAGE.clear()
    KEPT.clear()
    PARTS.clear()
    t_c = time.perf_counter()
    crops = [variant_crops(frame, g) if len(VARIANTS) > 1 else prepare_crops(frame, g, cfg) for g in groups]
    STAGE["crop_s"] = time.perf_counter() - t_c      # first outline of every box (the other 3: crop_more_s, lazily)
    order, t_skim, hits = list(range(len(groups))), None, [0] * len(groups)
    names = [0] * len(groups)
    if SKIM_NAME and opts["--order"] == "skim":
        both = list(POOL.map(lambda c: skim_both(c[0] if isinstance(c, list) else c, req), crops))
        hits = [h for h, _ in both]
        few = sum(n for _, n in both) <= max(3, len(groups) // 4)  # the name picks out a few boxes (not Task 2's 27
        if few:                                                    # Chroma): they go first and their 4 outlines are
            names = [n for _, n in both]                           # queued at once; otherwise the numbers alone
        order.sort(key=lambda i: (-names[i], -hits[i]))
        t_skim = time.perf_counter() - t0
        STAGE["skim_s"], STAGE["order"] = round(t_skim, 4), order[:]            # diagnosis only (8 Nov)
        STAGE["name_seen"], STAGE["hits"] = [n for _, n in both], hits[:]
    elif opts["--order"] in ("skim", "label") and req.diameter is not None:
        fn = label_hits if opts["--order"] == "label" else (lambda c, r: quick_hits(c[1], r))
        hits = list(POOL.map(lambda c: fn(c[0] if isinstance(c, list) else c, req), crops))
        order.sort(key=lambda i: -hits[i])
        t_skim = time.perf_counter() - t0
    def first_calls(i):              # --lazy 1: only the first outline, except boxes whose skim showed both numbers
        c = crops[i]                 # (8 Nov, --skim-name 1: or the requested name, when it picks out a few boxes)
        if not (LAZY and isinstance(c, list)):
            return c
        if hits[i] < EAGER_HITS and not names[i]:
            return c[0]
        return c[:len(VARIANTS)] if ALT_LAZY else c

    if PREFETCH and ORDERED:         # --ordered-prefetch 1: every box's calls queued from here in skim order, so the
        for i in order:              # likely box is first in the OCR queue (with one prefetch per worker thread the 27
            prefetch(first_calls(i))  # threads race and the order is lost on a loaded server; 4 Nov, night)

    def one(i):
        if PREFETCH and not ORDERED:  # this box's OCR calls all at once, when its read starts (--workers boxes at a time)
            prefetch(first_calls(i))
        keep = {} if PICK_CROPS else None
        d = full(crops[i], req, par=PAR_VARIANTS and (hits[i] >= EAGER_HITS or bool(names[i])), keep=keep)
        if keep:
            KEPT[i] = keep
        return d

    futs = {POOL.submit(one, i): i for i in order}
    pick, done = None, []
    for f in as_completed(futs):
        d, reason = f.result()
        done.append((futs[f], d, time.perf_counter() - t0, reason))
        if d == MATCH and pick is None:
            i = futs[f]
            if PICK_CROPS:                             # the barcode is located in its own process, beside the outline
                t_p = time.perf_counter()
                spine, m = pick_parts.spine_of(frame, groups[i].box.poly)
                bc = CROPPOOL.submit(pick_parts.find_bars, spine)
            if opts["--snap"] == "1":                  # the outline to draw is part of the PICK time
                t_s = time.perf_counter()
                SNAPPED[futs[f]] = snap_outline(frame, groups[futs[f]].box.poly)
                STAGE["snap_s"] = time.perf_counter() - t_s
            if PICK_CROPS:                             # the brand / detail / barcode crops are part of the PICK time
                t_w = time.perf_counter()
                PARTS[i] = {**pick_parts.text_parts(KEPT.get(i, {}).get("seen") or [], req),
                            **pick_parts.barcode_from(spine, m, *bc.result())}
                STAGE["crops_s"] = time.perf_counter() - t_p          # from the MATCH, beside the outline
                STAGE["crops_after_outline_s"] = time.perf_counter() - t_w   # what the crops add after the outline
            pick = (futs[f], time.perf_counter() - t0)
    return pick, done, t_skim


sys.path.insert(0, os.path.join(os.path.dirname(H), "det"))
import tightlib  # noqa: E402

SNAP = dict(blur=1.5, end=0.02, lam=0.6, bias=-0.02)   # chosen on the dev photos only (det/snap_eval.py, round 3)
SNAPPED = {}
KEPT, PARTS = {}, {}                        # --pick-crops 1: per box the MATCH outline's OCR record; the PICK's crops
CROPPOOL = None
if PICK_CROPS:
    import pick_parts  # noqa: E402
    import multiprocessing as mp  # noqa: E402
    from concurrent.futures import ProcessPoolExecutor  # noqa: E402
    # the barcode of the PICKed box in its own process: on a thread it held the GIL against the outline drawing (8 Nov,
    # fqzc: outline 0.06-0.1 -> 0.08-0.19 s); only the straightened box (~0.4 MB) is sent. Started and warmed up here.
    CROPPOOL = ProcessPoolExecutor(1, mp_context=mp.get_context("spawn"), initializer=pick_parts.init_worker, initargs=(C,))
    CROPPOOL.submit(pick_parts.find_bars, np.full((400, 80, 3), 200, np.uint8)).result()
STAGE, _STAGE_LOCK = {}, threading.Lock()   # stage times of the last request (logged only, never used by a decision)


REFINER = None
if opts["--refine"] == "1":      # --snap 1 --refine 1: the learned edge refiner (det/train_refiner.py) instead of the snap
    import refine_lib  # noqa: E402
    REFINER = refine_lib.Refiner(os.path.join(os.path.dirname(H), "det", "weights", "refiner.pt"), opts["--refine-device"])
    REFINER(np.zeros((64, 64, 3), np.uint8), np.array([[8, 8], [56, 8], [56, 20], [8, 20]], np.float64))   # warm-up
# --draw-imgsz 1280 (6 Nov): the PICKed box is detected once more at 1280 px and that outline is refined and drawn
# (chosen on the dev photos with the cross-validation nets: tight 18.5% from the 544 x 960 outlines, 24.2% from the
# 1280 ones; 28-Sep test 19.4 -> 25.8%). Reading and the decision keep the 544 x 960 outlines.
DET_DRAW = None
if opts["--draw-imgsz"] != "0":
    DET_DRAW = BoxDetector(weights, _m["task"], int(opts["--draw-imgsz"]), "0", _m["conf"], _m["iou"], cfg["classes"])
    DET_DRAW.detect(np.zeros((640, 640, 3), np.uint8))


def draw_poly(frame, poly):
    """The 1280 px outline of the same box (highest IoU with the PICKed outline, at least 0.3), else the PICKed one."""
    if DET_DRAW is None:
        return poly
    groups, _ = group_labels(DET_DRAW.detect(frame)[0], cfg["grouping"]["min_label_overlap"])
    best = max(((tightlib.iou(np.asarray(g.box.poly, np.float32), np.asarray(poly, np.float32)), g.box.poly)
                for g in drop_wide(groups)), default=(0.0, None), key=lambda t: t[0])
    return best[1] if best[0] >= 0.3 else poly


def snap_outline(frame, poly):
    """--snap 1: the PICKed box's outline, edge-corrected on the full photo (ends moved to the strongest edge, sides
    drawn in by 2% of the width); the gradients are taken on a window around the box only. --refine 1: the edges are
    placed by the learned refiner instead. Drawing only: the reads and the decision use the detector outline and its
    variants."""
    poly = draw_poly(frame, poly)
    if WEBCAM_FRAME and opts["--refine-webcam"] == "0":
        # 7 Nov: the refiner was trained and tested on phone photos only (owner outlines), never on webcam frames;
        # on the Task 3 shelf stills it put the top end of K04 and K10 into the shelf bar. --refine-webcam 0: a webcam
        # frame (Task 3, the recordings) gets the 1280 px outline as detected. Rule set after seeing K04 / K10.
        if opts["--webcam-ends"] == "1" and REFINER is not None:
            # 8 Nov: that outline stops short of a dark box bottom (Accuforce); each END may move to the refiner's
            # end only where the band between them looks like the side it joins (webcam_ends.ends_by_colour). Rule
            # designed after seeing the Task 3 and recording stills; drawing only, the decision is already made.
            import webcam_ends
            return webcam_ends.ends_by_colour(frame, poly, REFINER(frame, poly)).astype(np.float32)
        return np.asarray(poly, np.float32)
    if REFINER is not None:
        return REFINER(frame, poly).astype(np.float32)
    return gradient_snap(frame, poly)


WEBCAM_FRAME = False


def gradient_snap(frame, poly):
    p = np.asarray(poly, np.float32)
    pad = int(0.05 * np.ptp(p, axis=0).max()) + 16
    x0, y0 = np.maximum(np.floor(p.min(0)) - pad, 0).astype(int)
    x1, y1 = np.minimum(np.ceil(p.max(0)) + pad, [frame.shape[1], frame.shape[0]]).astype(int)
    gx, gy = tightlib.gradients(frame[y0:y1, x0:x1], SNAP["blur"])
    return tightlib.snap(p - [x0, y0], gx, gy, SNAP["end"], 0.0, SNAP["lam"], side_bias=SNAP["bias"]) + [x0, y0]


def request_once(name, key=None):
    """The mentor's request: photo from disk -> boxes -> all boxes read at once -> PICK. -> timings dict."""
    req = parse_request(SETS[name][1])
    _CACHE.clear()
    t0 = time.perf_counter()
    frame = cv2.imread(SETS[name][0])
    t_read = time.perf_counter() - t0
    groups = boxes(frame)
    t_det = time.perf_counter() - t0
    pick, done, t_skim = read_all(frame, groups, req, t0)
    t_all = time.perf_counter() - t0
    return {"read_photo_s": t_read, "detect_s": t_det - t_read, "boxes": len(groups), "pick": pick, "skim_s": t_skim,
            "all_read_s": t_all, "matches": sorted(i for i, d, *_ in done if d == MATCH),
            # log for the evidence videos (taken after the clock stopped): when each box's read ended, and the outlines
            "events": [[i, d, round(t, 4), why] for i, d, t, why in done], **outlines(groups, pick), **stages(),
            **(save_parts(key, pick) if key else {})}


def save_parts(key, pick):
    """--pick-crops 1, after the clock: the PICK's crops written to out/<tag>/crops/<key>/ -> {"crops": info} or {}."""
    if not (PICK_CROPS and pick and pick[0] in PARTS):
        return {}
    info = pick_parts.save(os.path.join(OUT, "crops", key), PARTS[pick[0]], reader())
    return {"crops": dict(info, dir=f"crops/{key}")}


def stages():
    return {"stages": {k: round(v, 4) if isinstance(v, float) else v for k, v in STAGE.items()}}


def outlines(groups, pick):
    out = {"polys": [np.round(g.box.poly, 1).tolist() for g in groups]}
    if pick and pick[0] in SNAPPED:
        out["pick_poly"] = np.round(SNAPPED[pick[0]], 1).tolist()
    return out


def timing(name):
    request_once(name)                                     # warm-up (first call builds GPU kernels)
    runs = [request_once(name, f"{name}_run{k + 1}") for k in range(int(opts["--runs"]))]
    picks = [r["pick"][1] for r in runs if r["pick"]]
    res = {"set": name, "request": SETS[name][1], "detector": weights, "imgsz": opts["--imgsz"], "workers": WORKERS,
           "prefetch": PREFETCH, "order": opts["--order"], "ocr_workers": opts["--ocr-workers"], "procs": PROCS, "proc_device": opts["--proc-device"],
           "proc_threads": opts["--proc-threads"], "cls_cpu": opts["--cls-cpu"], "skim_procs": opts["--skim-procs"], "variants": len(VARIANTS), "views": opts["--views"], "alt_views": ALT_VIEWS, "skim_name": SKIM_NAME, "alt_lazy": ALT_LAZY, "par_variants": PAR_VARIANTS, "pick_crops": PICK_CROPS, "lazy": LAZY, "snap": opts["--snap"], "refine": opts["--refine"], "refine_device": opts["--refine-device"], "draw_imgsz": opts["--draw-imgsz"], "barcode": opts["--barcode"], "ordered_prefetch": ORDERED, "eager_hits": EAGER_HITS, "priority": DISPATCH is not None,
           "cache": dict(STATS), "ocr_variant": OCR_VARIANT,
           "runs": len(runs), "picked_box": [r["pick"][0] if r["pick"] else None for r in runs],
           "matches_each_run": [r["matches"] for r in runs],
           "pick_s_median": round(statistics.median(picks), 3) if picks else None,
           "pick_s_max": round(max(picks), 3) if picks else None,
           "all_read_s_median": round(statistics.median(r["all_read_s"] for r in runs), 3),
           "detect_s_median": round(statistics.median(r["detect_s"] for r in runs), 3),
           "read_photo_s_median": round(statistics.median(r["read_photo_s"] for r in runs), 3), "raw": runs}
    json.dump(res, open(os.path.join(OUT, f"{name}_time.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "raw"}), flush=True)


T3 = os.path.join(C, "results/tasks_v2_20261029/task3")


def task3_requests():
    """The 41 Task 3 requests of 29 Oct: 20 boxes on the shelf (live plan) and 21 absent products (cmd.txt), each with
    the first camera still of its own take (1920 x 1080, no overlay) as the photo."""
    import glob
    import re
    live = os.path.join(T3, "live_20261029_155853")
    out = []
    for p in json.load(open(os.path.join(live, "inputs.json")))["plan"]:
        photo = sorted(glob.glob(os.path.join(live, "cases", p["box_id"], "take_*", "photos", "*_camera.png")))[0]
        out.append({"id": p["box_id"], "request": p["request"], "target": p["box_id"], "photo": photo})
    for c in sorted(glob.glob(os.path.join(T3, "absent_20261029_155853", "N*"))):
        cmd = open(glob.glob(os.path.join(c, "take_*", "cmd.txt"))[0], encoding="utf-8").read()
        text = re.search(r'"--request", "([^"]+)"', cmd).group(1)
        photo = sorted(glob.glob(os.path.join(c, "take_*", "photos", "*_camera.png")))[0]
        out.append({"id": os.path.basename(c), "request": text, "target": None, "photo": photo})
    return out


POS = json.load(open(os.path.join(H, "task3_positions.json")))   # box -> centre x of its VERIFIED outlines (29 Oct)
UNKNOWN = sorted({r["box_id"] for r in csv.DictReader(open(os.path.join(T3, "live_20261029_155853",
                                                                      "inventory_reference.csv"), encoding="utf-8"))}
                 - set(POS))


# The 6 slots no run verified, left to right, as read by Claude from the printed spines of the 29 Oct shelf photo
# (Scoreflex, Sapphire 3, Conqueror NC Pro 4 x 8, Conqueror NC Pro 3.5 x 15, NC Emerge 5.5 x 12, Agent 2.75 x 30);
# consistent with the verified K09 / K18 / K13 between them. NOT owner review; used only with --claude-slots 1.
CLAUDE_SLOTS = ["K19", "K14", "K12", "K11", "K08", "K20"]


def shelf_ids(groups):
    """Box id of each detected box: the box whose verified position (live 29 Oct + replays, same camera position) is
    within 35 px of its centre; "U" = a slot no run ever verified (one of UNKNOWN, which one is not known).
    --claude-slots 1: the U slots get CLAUDE_SLOTS in left-to-right order when there are exactly 6 of them."""
    ref = {k: float(np.median(v)) for k, v in POS.items()}
    out = {}
    for i, g in enumerate(groups):
        x = float(g.box.poly[:, 0].mean())
        k, d = min(((k, abs(x - v)) for k, v in ref.items()), key=lambda t: t[1])
        out[i] = k if d <= 35 else "U"
    us = sorted((i for i in out if out[i] == "U"), key=lambda i: float(groups[i].box.poly[:, 0].mean()))
    if opts["--claude-slots"] == "1" and len(us) == len(CLAUDE_SLOTS):
        for i, k in zip(us, CLAUDE_SLOTS):
            out[i] = k
    return out


def task3(_name=None):
    """Every Task 3 request on its own photo: every box read for that request at once; PICK = first MATCH. Cells per
    request x box: requested box green = TP, requested box not green = FN, other box green = FP (wrong box), else TN."""
    global WEBCAM_FRAME
    WEBCAM_FRAME = True
    rows, cells = [], Counter()
    reqs = task3_requests()
    for k, r in enumerate([reqs[0]] + reqs):                   # the first request once more as warm-up, not counted
        req = parse_request(r["request"])
        _CACHE.clear()
        t0 = time.perf_counter()
        frame = cv2.imread(r["photo"])
        groups = boxes(frame)
        t_det = time.perf_counter() - t0
        pick, done, _ = read_all(frame, groups, req, t0)
        dec = {i: (d, why) for i, d, _, why in done}
        t_all = time.perf_counter() - t0
        if k == 0:
            continue
        ids = shelf_ids(groups)
        greens = sorted(i for i, (d, _) in dec.items() if d == MATCH)
        tgt = r["target"]
        row = {"id": r["id"], "request": r["request"], "target": tgt, "target_position_known": tgt in POS,
               "boxes": len(groups), "green_boxes": [ids[i] for i in greens],
               "pick_box": ids[pick[0]] if pick else None, "pick_s": round(pick[1], 3) if pick else None,
               "detect_s": round(t_det, 3), "all_read_s": round(t_all, 3),
               "decisions": [[ids[i], d, why] for i, (d, why) in sorted(dec.items())],
               "events": [[i, d, round(t, 4)] for i, d, t, _ in done], "ids": ids, **outlines(groups, pick), **stages(),
               **save_parts(f"task3_{r['id']}", pick)}
        u_green = False
        for i, (d, _) in dec.items():
            if tgt in UNKNOWN and ids[i] == "U":                # only without --claude-slots             # one of the U slots is the target: truth per box unknown
                if d == MATCH:
                    cells["TP_by_elimination"] += 1
                    u_green = True
                continue
            pos = ids[i] == tgt
            cells[("TP" if d == MATCH else "FN") if pos else ("FP" if d == MATCH else "TN")] += 1
        if tgt in UNKNOWN and not u_green and "U" in ids.values():   # elimination mode only
            cells["FN"] += 1
        rows.append(row)
        print(r["id"], r["request"], "->", row["pick_box"], row["pick_s"], "greens", row["green_boxes"], flush=True)
    live = [x for x in rows if x["target"]]
    right = lambda x: x["pick_box"] == x["target"] or (x["target"] in UNKNOWN and x["pick_box"] == "U")  # noqa: E731
    green_t = [x["pick_s"] for x in live if x["pick_box"] and right(x)]
    res = {"detector": weights, "imgsz": opts["--imgsz"], "workers": WORKERS, "requests": len(rows),
           "claude_slots": opts["--claude-slots"] == "1", "ocr_variant": OCR_VARIANT, "variants": len(VARIANTS), "views": opts["--views"], "alt_views": ALT_VIEWS, "skim_name": SKIM_NAME, "alt_lazy": ALT_LAZY, "par_variants": PAR_VARIANTS, "pick_crops": PICK_CROPS, "procs": PROCS, "lazy": LAZY, "ordered_prefetch": ORDERED, "priority": DISPATCH is not None,
           "snap": opts["--snap"], "refine": opts["--refine"], "webcam_ends": opts["--webcam-ends"], "on_shelf": len(live), "on_shelf_green_right_box": len(green_t),
           "wrong_box_green": sum(1 for x in rows for b in x["green_boxes"]
                                  if b != x["target"] and not (b == "U" and x["target"] in UNKNOWN)),
           "green_by_elimination": sum(1 for x in live if x["target"] in UNKNOWN and "U" in x["green_boxes"]),
           "absent_any_green": sum(1 for x in rows if not x["target"] and x["green_boxes"]),
           "photos_not_20_boxes": [x["id"] for x in rows if x["boxes"] != 20], "unknown_positions": UNKNOWN,
           "pick_s_median": round(statistics.median(green_t), 3) if green_t else None,
           "pick_s_max": max(green_t) if green_t else None,
           "pick_within_1s": sum(t <= 1 for t in green_t), "pick_within_2s": sum(t <= 2 for t in green_t),
           "all_read_s_median": round(statistics.median(x["all_read_s"] for x in rows), 3), "cells": dict(cells)}
    json.dump({"summary": res, "rows": rows}, open(os.path.join(OUT, "task3.json"), "w"), indent=1)
    print(json.dumps(res), flush=True)


def live(setdir):
    """The live webcam runs replayed on the AI PC (6 Nov): set.json (made from a live run folder) lists each request
    with the first camera still of its take. Box ids: the k-th detected box from the left is the k-th box of
    order_left_to_right (the shelf order read by Claude from the owner's calibration photo, not checked by a person),
    used only when exactly that many boxes are detected; else the photo is counted apart (count_mismatch) and none of
    its boxes is scored. Cells per request x box as task3. Writes out/<tag>/live_<set>.json."""
    global WEBCAM_FRAME
    WEBCAM_FRAME = True
    s = json.load(open(os.path.join(setdir, "set.json")))
    order = s["order_left_to_right"]
    rows, cells, mismatch = [], Counter(), []
    reqs = s["requests"]
    for k, r in enumerate([reqs[0]] + reqs):                   # the first request once more as warm-up
        req = parse_request(r["request"])
        _CACHE.clear()
        t0 = time.perf_counter()
        # 7 Nov: "photos" (the take's first camera stills, time order) are read one after another until one gives a
        # PICK, as the laptop reads several seconds of video; each still is decided on its own (two sources on that
        # still). The clock runs from the first still. Without "photos": the first still only, as before.
        stills = r.get("photos") or [r["photo"]]
        for j, name in enumerate(stills):
            frame = cv2.imread(os.path.join(setdir, name))
            groups = boxes(frame)
            t_det = time.perf_counter() - t0
            pick, done, _ = read_all(frame, groups, req, t0)
            if pick or any(d == MATCH for _, d, _, _ in done):
                break
        still_used = j + 1
        t_all = time.perf_counter() - t0
        if k == 0:
            continue
        dec = {i: (d, why) for i, d, _, why in done}
        xs = sorted(range(len(groups)), key=lambda i: float(groups[i].box.poly[:, 0].mean()))
        ids = {i: order[j] for j, i in enumerate(xs)} if len(groups) == len(order) else {i: "?" for i in xs}
        greens = sorted(i for i, (d, _) in dec.items() if d == MATCH)
        if len(groups) != len(order):
            mismatch.append(r["id"])
        else:
            for i, (d, _) in dec.items():
                pos = ids[i] == r["target"]
                cells[("TP" if d == MATCH else "FN") if pos else ("FP" if d == MATCH else "TN")] += 1
        rows.append({"id": r["id"], "request": r["request"], "target": r["target"], "boxes": len(groups),
                     "green_boxes": [ids[i] for i in greens], "pick_box": ids[pick[0]] if pick else None,
                     "pick_s": round(pick[1], 3) if pick else None, "detect_s": round(t_det, 3),
                     "all_read_s": round(t_all, 3), "decisions": [[ids[i], d, why] for i, (d, why) in sorted(dec.items())],
                     "events": [[i, d, round(t, 4)] for i, d, t, _ in done], "ids": ids, **outlines(groups, pick),
                     "stills_read": still_used, "stills": len(stills),
                     **stages(), **save_parts(f"{os.path.basename(os.path.normpath(setdir))}_{r['id']}", pick)})
        print(r["id"], r["request"], "->", rows[-1]["pick_box"], rows[-1]["pick_s"], "greens", rows[-1]["green_boxes"],
              flush=True)
    green_t = [x["pick_s"] for x in rows if x["pick_box"] == x["target"]]
    res = {"set": os.path.basename(os.path.normpath(setdir)), "source_run": s["source_run"], "detector": weights,
           "imgsz": opts["--imgsz"], "ocr_variant": OCR_VARIANT, "variants": len(VARIANTS), "views": opts["--views"], "alt_views": ALT_VIEWS, "skim_name": SKIM_NAME, "alt_lazy": ALT_LAZY, "par_variants": PAR_VARIANTS, "pick_crops": PICK_CROPS, "procs": PROCS,
           "refine": opts["--refine"], "webcam_ends": opts["--webcam-ends"], "requests": len(rows), "slots": len(order), "green_right_box": len(green_t),
           "wrong_box_green": sum(1 for x in rows for b in x["green_boxes"] if b not in (x["target"], "?")),
           "green_count_mismatch_photo": sum(1 for x in rows for b in x["green_boxes"] if b == "?"),
           "count_mismatch": mismatch,
           "pick_s_median": round(statistics.median(green_t), 3) if green_t else None,
           "pick_s_max": max(green_t) if green_t else None,
           "pick_within_1s": sum(t <= 1 for t in green_t), "pick_within_2s": sum(t <= 2 for t in green_t),
           "all_read_s_median": round(statistics.median(x["all_read_s"] for x in rows), 3), "cells": dict(cells)}
    json.dump({"summary": res, "rows": rows}, open(os.path.join(OUT, f"live_{res['set']}.json"), "w"), indent=1)
    print(json.dumps(res), flush=True)


def filename_product(stem):
    import re
    from src.box_verifier import PRODUCTS
    key = re.sub(r"[^a-z0-9]", "", stem.lower())
    hits = [p for p in PRODUCTS if p.replace(" ", "") in key]
    return max(hits, key=len) if hits else None


DRIVE_OUTLINES = opts["--drive-outlines"] == "1"


def discover_all(frame, g, i, req0):
    """--drive-outlines 1 (8 Nov): the Drive discovery reads every outline the reader reads (4 variants, and with
    --alt-views 1 the other view's 4), not only the first: identities = all read on any outline; two sources = an
    identity the reader's own rule (full: one outline MATCH, none reading another identity) greens; size read = the
    diameter AND length read, and the same on every outline that read one (outlines disagreeing = not read)."""
    outl = variant_crops(frame, g) if len(VARIANTS) > 1 else [prepare_crops(frame, g, cfg)]
    evs = [read_target_evidence(*c, reader(), cfg, req0, catalog, attempt=3, identity_verified=True)[0] for c in outl]
    ids, sizes = set(), set()
    for ev in evs:
        t = ev.text
        if t is not None and t.product_ok:
            ids.add((t.product, t.diameter if t.diameter_ok else None, t.length if t.length_ok else None))
        for r in ev.refs:
            ids.add((r.product, r.diameter, r.length))
        if t is not None and t.diameter_ok and t.length_ok:
            sizes.add((t.diameter, t.length))
    two = [x for x in ids if x[1] is not None and x[2] is not None and full(outl, Request(*x))[0] == MATCH]
    return {"index": i, "identities": sorted(map(list, ids), key=str), "two_sources": sorted(map(list, two)),
            "size_read": list(next(iter(sizes))) if len(sizes) == 1 else None, "size_disagree": len(sizes) > 1,
            "outlines": len(outl), "spine": (evs[0].spine or "")[:160], "refs": sorted({r.text for ev in evs for r in ev.refs})}


def drive(photo_dir):
    """Leo's Drive product photos (6 Nov): one box per photo. 1) discovery, not timed: every box read once with the full
    read (as results/drive_photos_20261029/photo_test.py), each identity its text or REF names re-decided with the
    laptop's two-source rule; the chosen box = the leftmost two-source box of the photo's own product (file name), else
    of any product. 2) the mentor's request for that box's identity, timed as `time` (cache cleared, final reader): PICK
    and the drawn outline. A photo with no two-source box: what was read and why it cannot go green. The file name is
    the photo's product (a proxy: photos may hold other products); the printed size of the chosen box is checked by
    Claude from the photo, not by the owner. Writes out/<tag>/drive.json and out/<tag>/drive/<photo>.jpg."""
    import glob
    from src.box_evidence import redecide
    os.makedirs(os.path.join(OUT, "drive"), exist_ok=True)
    rows = []
    for path in sorted(glob.glob(os.path.join(photo_dir, "*.jpg"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        frame = cv2.imread(path)
        expected = filename_product(stem)
        groups = boxes(frame)
        req0 = parse_request(f"{expected} 3 x 20") if expected and expected != "permanent sled bag" else parse_request("angiolite 3 x 20")

        def discover(i):
            if DRIVE_OUTLINES:
                return discover_all(frame, groups[i], i, req0)
            ev, *_ = read_target_evidence(*prepare_crops(frame, groups[i], cfg), reader(), cfg, req0, catalog, attempt=3,
                                          identity_verified=True)
            ids = set()
            if ev.text is not None and ev.text.product_ok:
                ids.add((ev.text.product, ev.text.diameter if ev.text.diameter_ok else None,
                         ev.text.length if ev.text.length_ok else None))
            for r in ev.refs:
                ids.add((r.product, r.diameter, r.length))
            two = [x for x in ids if x[1] is not None and x[2] is not None
                   and redecide(ev, Request(*x), cfg, catalog)[0] == MATCH]
            t = ev.text
            # 7 Nov: the printed size read on this box (diameter AND length at full confidence), whatever the brand:
            # "box read" for brands outside the catalogue too (the name cannot be checked there, the size can)
            size = [t.diameter, t.length] if t is not None and t.diameter_ok and t.length_ok else None
            return {"index": i, "identities": sorted(map(list, ids), key=str), "two_sources": sorted(map(list, two)),
                    "size_read": size, "spine": (ev.spine or "")[:160], "refs": [r.text for r in ev.refs]}

        _CACHE.clear()
        found = list(POOL.map(discover, range(len(groups))))
        xs = {i: float(groups[i].box.poly[:, 0].mean()) for i in range(len(groups))}
        cands = sorted(((b["index"], tuple(x)) for b in found for x in b["two_sources"]), key=lambda c: xs[c[0]])
        own = [c for c in cands if c[1][0] == expected]
        chosen = (own or cands or [None])[0]
        row = {"photo": os.path.basename(path), "product_from_name": expected, "boxes": len(groups),
               "two_source_boxes": len({b["index"] for b in found if b["two_sources"]}),
               "one_source_boxes": sum(1 for b in found if b["identities"] and not b["two_sources"]),
               "nothing_read": sum(1 for b in found if not b["identities"]),
               "size_read_boxes": sum(1 for b in found if b["size_read"]),
               "size_disagree_boxes": sum(1 for b in found if b.get("size_disagree")), "drive_outlines": DRIVE_OUTLINES,
               "chosen": None, "discovery": found}
        vis = frame.copy()
        tk = max(2, frame.shape[1] // 500)
        if chosen:
            i0, ident = chosen
            text = f"{ident[0]} {ident[1]:g} x {ident[2]:g}"
            req = parse_request(text)
            _CACHE.clear()
            SNAPPED.clear()
            t0 = time.perf_counter()
            frame2 = cv2.imread(path)
            groups2 = boxes(frame2)
            t_det = time.perf_counter() - t0
            pick, done, _ = read_all(frame2, groups2, req, t0)
            t_all = time.perf_counter() - t0
            greens = sorted(i for i, d, *_ in done if d == MATCH)
            row["chosen"] = {"box": i0, "request": text, "pick_box": pick[0] if pick else None,
                             "pick_s": round(pick[1], 3) if pick else None, "greens": greens, "all_read_s": round(t_all, 3),
                             "detect_s": round(t_det, 3), "events": [[i, d, round(t, 4)] for i, d, t, _ in done],
                             "stages": dict(STAGE), **outlines(groups2, pick), **save_parts(f"drive_{stem}", pick)}
            if pick:
                poly = SNAPPED.get(pick[0], groups2[pick[0]].box.poly)
                cv2.polylines(vis, [np.rint(poly).astype(np.int32)], True, (60, 200, 0), tk, cv2.LINE_AA)
            banner = f'"{text}" -> ' + (f"PICK in {pick[1]:.2f} s" if pick else "no box verified")
        else:
            best = max(found, key=lambda b: len(b["identities"]), default=None)
            why = ("product not in the catalogue" if expected is None else "no box with two sources")
            banner = f"{expected or stem}: {why}" + (f"; read: {best['identities'][0]}" if best and best["identities"] else "")
        s = max(1.0, frame.shape[1] / 1200)
        cv2.rectangle(vis, (0, 0), (frame.shape[1], int(48 * s)), (20, 20, 20), -1)
        cv2.putText(vis, banner[:90], (int(10 * s), int(33 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s, (255, 255, 255),
                    max(1, int(2 * s)))
        cv2.imwrite(os.path.join(OUT, "drive", stem + ".jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
        rows.append(row)
        print(stem, expected, len(groups), "two-source boxes", row["two_source_boxes"], "->",
              (row["chosen"] or {}).get("request"), (row["chosen"] or {}).get("pick_s"), flush=True)
    res = {"photos": len(rows), "with_green": sum(1 for r in rows if r["chosen"] and r["chosen"]["pick_box"] is not None),
           "catalog_product_photos": sum(1 for r in rows if r["product_from_name"]), "ocr_variant": OCR_VARIANT,
           "refine": opts["--refine"]}
    json.dump({"summary": res, "rows": rows}, open(os.path.join(OUT, "drive.json"), "w"), indent=1, default=str)
    print(json.dumps(res), flush=True)


def demo(name, photo=None, text=None):
    """Deck picture: the request on the photo, ONE green outline on the PICKed box (edge-corrected with --snap 1, else
    as detected)."""
    req = parse_request(text or SETS[name][1])
    _CACHE.clear()
    SNAPPED.clear()
    frame = cv2.imread(photo or SETS[name][0])
    groups = boxes(frame)
    pick, done, _ = read_all(frame, groups, req, time.perf_counter())
    vis = frame.copy()
    t = max(3, frame.shape[1] // 400)
    if os.environ.get("AIPC_DEMO_ALL") == "1":         # 6 Nov (owner): only the green box is shown by default
        for g in groups:
            cv2.polylines(vis, [np.rint(g.box.poly).astype(np.int32)], True, (150, 150, 150), max(1, t // 3))
    if pick:
        poly = SNAPPED.get(pick[0], groups[pick[0]].box.poly)
        cv2.polylines(vis, [np.rint(poly).astype(np.int32)], True, (60, 200, 0), t)
    s = max(1.0, frame.shape[1] / 1600)
    label = f"{text or SETS[name][1]}  ->  " + (f"PICK box {pick[0]}" if pick else "no box verified")
    cv2.rectangle(vis, (0, 0), (frame.shape[1], int(56 * s)), (20, 20, 20), -1)
    cv2.putText(vis, label, (int(14 * s), int(40 * s)), cv2.FONT_HERSHEY_SIMPLEX, 1.0 * s, (255, 255, 255), max(2, int(2 * s)))
    out = os.path.join(OUT, f"{name}_demo.jpg")
    cv2.imwrite(out, vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if pick:                                           # zooms of the PICKed box: both ends and the middle
        d = groups[pick[0]].box.poly
        cols = [(d, (40, 40, 230)), (gradient_snap(frame, d), (230, 120, 20))]
        if REFINER is not None:
            cols.append((REFINER(frame, d), (220, 40, 220)))
        z = frame.copy()
        for poly, col in cols:
            cv2.polylines(z, [np.rint(poly).astype(np.int32)], True, col, max(1, frame.shape[1] // 1500), cv2.LINE_AA)
        c, u, v, L, W = tightlib.frame_of(np.asarray(d, np.float64))
        tiles = []
        for f in (-1.0, 0.0, 1.0):
            pt, r = c + f * L * u, int(1.6 * W)
            x0, y0 = int(max(0, pt[0] - r)), int(max(0, pt[1] - r))
            tiles.append(cv2.resize(z[y0:y0 + 2 * r, x0:x0 + 2 * r], (400, 400), interpolation=cv2.INTER_AREA))
        cv2.imwrite(os.path.join(OUT, f"{name}_zoom.jpg"), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(out, "pick", pick, "matches", sorted(i for i, d, *_ in done if d == MATCH), flush=True)


def main():
    mode, names = args[0], args[1:] or ["t1_drive", "t2_drive"]
    if mode == "task3":
        task3()
    elif mode == "live":
        for d in args[1:]:
            live(d)
    elif mode == "demo_drive":                         # demo_drive DRIVE_JSON: the chosen request of each Drive photo
        d = json.load(open(args[1]))
        base = os.path.join(C, "results/drive_photos_20261029/src_photos/main")
        for r in d["rows"]:
            if r["chosen"]:
                demo(os.path.splitext(r["photo"])[0], os.path.join(base, r["photo"]), r["chosen"]["request"])
    elif mode == "drive":
        drive(args[1] if len(args) > 1 else os.path.join(C, "results/drive_photos_20261029/src_photos/main"))
    elif mode == "demo":
        for n in names:
            if n.startswith("task3:"):                         # task3:K07 -> that request on its own shelf photo
                r = next(x for x in task3_requests() if x["id"] == n.split(":")[1])
                WEBCAM_FRAME = True
                demo(n.replace(":", "_"), r["photo"], r["request"])
            else:
                WEBCAM_FRAME = False
                demo(n)
    else:
        for n in names:
            {"score": score, "time": timing}[mode](n)
    for pool in (PPOOL, SKIMPOOL, CROPPOOL):
        if pool:
            pool.shutdown()


if __name__ == "__main__":
    main()

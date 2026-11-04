"""AI PC photo pipeline (4 Nov 2026): the mentor's request on a photo, box detector and reader on the RTX A5000.

  python aipc_tasks.py score SET ... [--det W --imgsz N --workers K --tag T]   every box x every printed size (confusion)
  python aipc_tasks.py time  SET ... [--det W --imgsz N --workers K --runs R]  the mentor's request -> PICK time
  python aipc_tasks.py task3 [--det W --imgsz N --workers K]          the 41 Task 3 requests, each on its own shelf photo

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
                  "--snap": "0", "--lazy": "0"}, [], iter(sys.argv[1:])
for a in it:
    if a in opts:
        opts[a] = next(it)
    else:
        args.append(a)
cfg = config.load_config(config.resolve("config/box_pipeline_config_webcam_eleven.yaml"))
cfg["verify"] = dict(cfg["verify"], decode_barcodes=False)
catalog = Catalog(str(config.resolve(cfg["verify"]["catalog_csv"])))
_m, _o = cfg["model"], cfg["ocr"]
weights = opts["--det"] or os.path.join(os.path.dirname(H), "..", "models", "box_y26_maxB.pt")
imgsz = [int(v) for v in opts["--imgsz"].split("x")] if "x" in opts["--imgsz"] else int(opts["--imgsz"])
det = BoxDetector(weights, _m["task"], imgsz, "0", _m["conf"], _m["iou"], cfg["classes"],
                  fixed_input=isinstance(imgsz, list))
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


def _submit(img, use_cls):
    """Future of the tokens of one OCR call (process pool, or the thread pool's own engines)."""
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
            f = _CACHE[k] = _submit(img, use_cls)
            STATS["miss" if count else "prefetched"] += 1
        elif count:
            STATS["hit"] += 1
    if not count:
        return None
    from src.box_ocr import OcrToken
    return [OcrToken(t, c, p) for t, c, p in f.result()]


if PREFETCH:
    SizeTextReader._run = _cached_run


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
for pool in (POOL, OCRPOOL):
    if pool:
        list(pool.map(lambda _: reader().read_once(np.full((48, 200, 3), 255, np.uint8)), range(WORKERS * 2)))
_CACHE.clear()


def prefetch(crops):
    for img in [i for c in (crops if isinstance(crops, list) else [crops]) for i in ocr_inputs(c)]:
        _cached_run(None, img, True, count=False)


def boxes(frame):
    groups, _ = group_labels(det.detect(frame)[0], cfg["grouping"]["min_label_overlap"])
    return sorted(drop_wide(groups), key=lambda g: tuple(np.round(g.box.poly.mean(axis=0)[::-1] / 150)))


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


def variant_crops(frame, g):
    """--variants 1: the box read from 4 outlines (as detected, sides 6% wider, 6% narrower, ends 2% longer), because
    the second source turned out to depend on a few pixels of outline (4 Nov: OpenVINO vs .pt outlines of model B
    gave Task 1 TP 6 vs 4 / 3). Size-label outlines unchanged."""
    from copy import copy
    out = []
    for fw, fl in VARIANTS:
        if fw == 0 and fl == 0:
            out.append(prepare_crops(frame, g, cfg))
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


def full(crops, req):
    """crops = one prepare_crops result, or a list of them (outline variants). Variants: MATCH only if one variant
    reaches MATCH and no variant reads a conflicting identity (ignored / expired); then 'needs_confirmation'."""
    many = crops if isinstance(crops, list) else [crops]
    res = []
    for k, c in enumerate(many):
        if k == 1 and LAZY:
            if res[0][0] in ("ignored", "expired"):     # --lazy 1: a conflict on the first outline already rules out
                return res[0]                           # MATCH, so the other outlines cannot change the cell
            if PREFETCH:
                prefetch(many[1:])
        ev, d, reason, *_ = read_target_evidence(*c, reader(), cfg, req, catalog, attempt=3, identity_verified=True)
        res.append((d, reason))
    if len(res) == 1:
        return res[0]
    ds = [d for d, _ in res]
    if MATCH in ds:
        if any(d in ("ignored", "expired") for d in ds):
            return "needs_confirmation", "outline variants disagree: " + "; ".join(f"{d}: {r}" for d, r in res)
        return next(r for r in res if r[0] == MATCH)
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


def read_all(frame, groups, req, t0):
    """Every box read for the request at once (POOL); --prefetch 1: its OCR calls started first on OCRPOOL;
    --order skim: boxes whose one-pass skim shows more requested numbers go first (order only). -> pick, done, t_skim"""
    crops = [variant_crops(frame, g) if len(VARIANTS) > 1 else prepare_crops(frame, g, cfg) for g in groups]
    order, t_skim, hits = list(range(len(groups))), None, [0] * len(groups)
    if opts["--order"] in ("skim", "label") and req.diameter is not None:
        fn = label_hits if opts["--order"] == "label" else (lambda c, r: quick_hits(c[1], r))
        hits = list(POOL.map(lambda c: fn(c[0] if isinstance(c, list) else c, req), crops))
        order.sort(key=lambda i: -hits[i])
        t_skim = time.perf_counter() - t0
    def one(i):
        c = crops[i]
        if PREFETCH:                 # this box's OCR calls all at once, when its read starts (--workers boxes at a time)
            # --lazy 1: only the first outline, except boxes whose skim showed both requested numbers (speed only)
            prefetch(c[0] if LAZY and isinstance(c, list) and hits[i] < 2 else c)
        return full(c, req)

    futs = {POOL.submit(one, i): i for i in order}
    pick, done = None, []
    for f in as_completed(futs):
        d, reason = f.result()
        done.append((futs[f], d, time.perf_counter() - t0, reason))
        if d == MATCH and pick is None:
            if opts["--snap"] == "1":                  # the outline to draw is part of the PICK time
                SNAPPED[futs[f]] = snap_outline(frame, groups[futs[f]].box.poly)
            pick = (futs[f], time.perf_counter() - t0)
    return pick, done, t_skim


sys.path.insert(0, os.path.join(os.path.dirname(H), "det"))
import tightlib  # noqa: E402

SNAP = dict(blur=1.5, end=0.02, lam=0.6, bias=-0.02)   # chosen on the dev photos only (det/snap_eval.py, round 3)
SNAPPED = {}


def snap_outline(frame, poly):
    """--snap 1: the PICKed box's outline, edge-corrected on the full photo (ends moved to the strongest edge, sides
    drawn in by 2% of the width); the gradients are taken on a window around the box only. Drawing only: the reads and
    the decision use the detector outline and its variants."""
    p = np.asarray(poly, np.float32)
    pad = int(0.05 * np.ptp(p, axis=0).max()) + 16
    x0, y0 = np.maximum(np.floor(p.min(0)) - pad, 0).astype(int)
    x1, y1 = np.minimum(np.ceil(p.max(0)) + pad, [frame.shape[1], frame.shape[0]]).astype(int)
    gx, gy = tightlib.gradients(frame[y0:y1, x0:x1], SNAP["blur"])
    return tightlib.snap(p - [x0, y0], gx, gy, SNAP["end"], 0.0, SNAP["lam"], side_bias=SNAP["bias"]) + [x0, y0]


def request_once(name):
    """The mentor's request: photo from disk -> boxes -> all boxes read at once -> PICK. -> timings dict."""
    req = parse_request(SETS[name][1])
    _CACHE.clear()
    t0 = time.perf_counter()
    frame = cv2.imread(SETS[name][0])
    t_read = time.perf_counter() - t0
    groups = boxes(frame)
    t_det = time.perf_counter() - t0
    pick, done, t_skim = read_all(frame, groups, req, t0)
    return {"read_photo_s": t_read, "detect_s": t_det - t_read, "boxes": len(groups), "pick": pick, "skim_s": t_skim,
            "all_read_s": time.perf_counter() - t0, "matches": sorted(i for i, d, *_ in done if d == MATCH)}


def timing(name):
    request_once(name)                                     # warm-up (first call builds GPU kernels)
    runs = [request_once(name) for _ in range(int(opts["--runs"]))]
    picks = [r["pick"][1] for r in runs if r["pick"]]
    res = {"set": name, "request": SETS[name][1], "detector": weights, "imgsz": opts["--imgsz"], "workers": WORKERS,
           "prefetch": PREFETCH, "order": opts["--order"], "ocr_workers": opts["--ocr-workers"], "procs": PROCS, "proc_device": opts["--proc-device"],
           "proc_threads": opts["--proc-threads"], "cls_cpu": opts["--cls-cpu"], "skim_procs": opts["--skim-procs"], "variants": len(VARIANTS), "lazy": LAZY, "snap": opts["--snap"],
           "cache": dict(STATS),
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
               "decisions": [[ids[i], d, why] for i, (d, why) in sorted(dec.items())]}
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
           "claude_slots": opts["--claude-slots"] == "1", "variants": len(VARIANTS),
           "on_shelf": len(live), "on_shelf_green_right_box": len(green_t),
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
    print(out, "pick", pick, "matches", sorted(i for i, d, *_ in done if d == MATCH), flush=True)


def main():
    mode, names = args[0], args[1:] or ["t1_drive", "t2_drive"]
    if mode == "task3":
        task3()
    elif mode == "demo":
        for n in names:
            if n.startswith("task3:"):                         # task3:K07 -> that request on its own shelf photo
                r = next(x for x in task3_requests() if x["id"] == n.split(":")[1])
                demo(n.replace(":", "_"), r["photo"], r["request"])
            else:
                demo(n)
    else:
        for n in names:
            {"score": score, "time": timing}[mode](n)
    for pool in (PPOOL, SKIMPOOL):
        if pool:
            pool.shutdown()


if __name__ == "__main__":
    main()

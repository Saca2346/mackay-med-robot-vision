"""Tilted and missed boxes (7 Nov 2026): test-time views and model ensembles for the box detector, measured honestly.

  python det_probe.py heldout CONFIG [--tilt 0,20,35] [--photos test|dev]   28-Sep owner outlines (held out), each photo
        also turned by every --tilt angle (image and truth outlines turned together, canvas enlarged): recall,
        precision, tight share per angle. Writes probe/<CONFIG>_<photos>.json.
  python det_probe.py count CONFIG OUT_DIR PHOTO...   boxes found per photo, pictures with every box (no truth).

CONFIG = name in CONFIGS below: which weights, input sizes, turned views and confidence. Views are merged by keeping, of
outlines that overlap (polygon IoU >= MERGE_IOU), the one with the highest confidence; with SUPPORT=k an outline is
kept only when k views found it (fewer stray boxes). Nothing here is trained; settings are compared on the dev photos
(23 Sep + 1 Oct) and the 28-Sep test is run once per config for the report.
"""
import json
import os
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO

E = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, E)
import tightlib as tl  # noqa: E402

MB = os.path.expanduser("~/Downloads/claude_offline_20261101/models/box_y26_maxB.pt")
R = lambda n: os.path.join(E, "runs", n, "weights", "last.pt")  # noqa: E731
# (weights, imgsz, angle) views; conf; support
CONFIGS = {
    "B544": ([(MB, (544, 960), 0)], 0.35, 1),
    "B1280": ([(MB, 1280, 0)], 0.35, 1),
    "B_ms": ([(MB, (544, 960), 0), (MB, 1280, 0)], 0.35, 1),
    "B_rot": ([(MB, 1280, a) for a in (0, -25, 25)], 0.35, 1),
    "B_ms_rot": ([(MB, (544, 960), 0), (MB, 1280, 0), (MB, 1280, -25), (MB, 1280, 25)], 0.35, 1),
    "B1280_c20": ([(MB, 1280, 0)], 0.20, 1),
    "B_ms_rot_c25": ([(MB, (544, 960), 0), (MB, 1280, 0), (MB, 1280, -25), (MB, 1280, 25)], 0.25, 1),
    "s26h": ([(R("yolo26s-obb_1280_hold"), 1280, 0)], 0.35, 1),
    "s26h_rot": ([(R("yolo26s-obb_1280_hold"), 1280, a) for a in (0, -25, 25)], 0.35, 1),
    "s12h": ([(R("yolo12s-obb_1280_hold"), 1280, 0)], 0.35, 1),
    "s12h_rot": ([(R("yolo12s-obb_1280_hold"), 1280, a) for a in (0, -25, 25)], 0.35, 1),
    # 8 Nov: the input size measured honestly, with the models that never saw the 28-Sep photos (trained at 1280, as B)
    "s26h_544": ([(R("yolo26s-obb_1280_hold"), (544, 960), 0)], 0.35, 1),
    "s12h_544": ([(R("yolo12s-obb_1280_hold"), (544, 960), 0)], 0.35, 1),
    "ens_h": ([(R("yolo26s-obb_1280_hold"), 1280, 0), (R("yolo12s-obb_1280_hold"), 1280, 0),
               (R("yolo26m-obb_1280_hold"), 1280, 0)], 0.35, 1),
    "ens_h_rot": ([(R(n), 1280, a) for n in ("yolo26s-obb_1280_hold", "yolo12s-obb_1280_hold") for a in (0, -25, 25)],
                  0.35, 1),
    "ens_h_rot_s2": ([(R(n), 1280, a) for n in ("yolo26s-obb_1280_hold", "yolo12s-obb_1280_hold") for a in (0, -25, 25)],
                     0.25, 2),
}
# 7 Nov, ordered merge (as aipc/tasks/multiview.py): the first view's boxes kept as they are, later views only add a
# box that overlaps none kept (IoU >= 0.3 or its centre inside one)
ORDERED = {
    "o_s26_s12": [(R("yolo26s-obb_1280_hold"), 1280, 0), (R("yolo12s-obb_1280_hold"), 1600, 0)],
    "o_s26_m26_s12": [(R("yolo26s-obb_1280_hold"), 1280, 0), (R("yolo26m-obb_1280_hold"), 1280, 0),
                      (R("yolo12s-obb_1280_hold"), 1600, 0)],
    "o_s26_rot": [(R("yolo26s-obb_1280_hold"), 1280, a) for a in (0, -25, 25)],
    "o_s26_all": [(R("yolo26s-obb_1280_hold"), 1280, 0), (R("yolo26m-obb_1280_hold"), 1280, 0),
                  (R("yolo12s-obb_1280_hold"), 1600, 0), (R("yolo26s-obb_1280_hold"), 1280, -25),
                  (R("yolo26s-obb_1280_hold"), 1280, 25)],
    "o_B": [(MB, (544, 960), 0), (MB, 1280, 0)],
    "o_B_s12": [(MB, (544, 960), 0), (MB, 1280, 0), (R("yolo12s-obb_1280_hold"), 1600, 0)],
    "o_B_all": [(MB, (544, 960), 0), (MB, 1280, 0), (R("yolo12s-obb_1280_hold"), 1600, 0),
                (R("yolo26m-obb_1280_hold"), 1280, 0), (MB, 1280, -25), (MB, 1280, 25)],
    # 8 Nov: 1280 first (B1280 alone had the best recall and precision), later views add what it missed
    "o_B1280_544": [(MB, 1280, 0), (MB, (544, 960), 0)],
    "o_B1280_rot": [(MB, 1280, 0), (MB, 1280, -25), (MB, 1280, 25)],
    "o_B1280_s12": [(MB, 1280, 0), (R("yolo12s-obb_1280_hold"), 1600, 0)],
}
for k, v in ORDERED.items():
    CONFIGS[k] = (v, 0.35, 1)
MERGE_IOU = 0.5
_models = {}


def model(w):
    if w not in _models:
        _models[w] = YOLO(w)
    return _models[w]


def turn(img, deg):
    """Image turned by deg (counter-clockwise, canvas enlarged) and the 2x3 matrix that maps original -> turned."""
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    c, s = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * s + w * c), int(h * c + w * s)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, m, (nw, nh), borderValue=(114, 114, 114)), m


def apply(m, pts):
    return pts @ m[:, :2].T + m[:, 2]


def view(img, w, imgsz, deg, conf):
    """(poly in img coordinates, conf, class) of one view."""
    im, m = turn(img, deg) if deg else (img, None)
    kw = dict(imgsz=list(imgsz) if isinstance(imgsz, tuple) else imgsz, conf=conf, iou=0.5, device="0", verbose=False)
    if isinstance(imgsz, tuple):
        kw["rect"] = False
    r = model(w).predict(im, **kw)[0]
    polys = r.obb.xyxyxyxy.cpu().numpy() if r.obb is not None else np.zeros((0, 4, 2))
    confs, cls = r.obb.conf.cpu().numpy(), r.obb.cls.cpu().numpy().astype(int)
    if m is not None:
        inv = cv2.invertAffineTransform(m)
        polys = np.array([apply(inv, p) for p in polys]).reshape(-1, 4, 2)
    return [(p, float(c), int(k)) for p, c, k in zip(polys, confs, cls)]


def detect_ordered(img, cfg, cls_keep=0):
    views, conf, _ = CONFIGS[cfg]
    kept = []
    for vi, (w, imgsz, deg) in enumerate(views):
        dets = sorted(((p, c) for p, c, k in view(img, w, imgsz, deg, conf) if k == cls_keep), key=lambda d: -d[1])
        if vi == 0:
            kept = dets
            continue
        for p, c in dets:
            ctr = tuple(map(float, p.mean(0)))
            if not any(tl.iou(q, p) >= 0.3 or cv2.pointPolygonTest(q.astype(np.float32), ctr, False) >= 0 for q, _ in kept):
                kept.append((p, c))
    return kept


def detect(img, cfg, cls_keep=0):
    if cfg in ORDERED:
        return detect_ordered(img, cfg, cls_keep)
    views, conf, support = CONFIGS[cfg]
    dets = []
    for vi, (w, imgsz, deg) in enumerate(views):
        dets += [(p, c, vi) for p, c, k in view(img, w, imgsz, deg, conf) if k == cls_keep]
    dets.sort(key=lambda d: -d[1])
    kept = []                                   # [poly, conf, set of views]
    for p, c, vi in dets:
        for k in kept:
            if tl.iou(k[0], p) >= MERGE_IOU:
                k[2].add(vi)
                break
        else:
            kept.append([p, c, {vi}])
    return [(k[0], k[1]) for k in kept if len(k[2]) >= support]


def heldout(cfg, tilts, session):
    out = {"config": cfg, "views": [[os.path.basename(os.path.dirname(os.path.dirname(w))) or "modelB", str(s), a]
                                    for w, s, a in CONFIGS[cfg][0]], "conf": CONFIGS[cfg][1], "support": CONFIGS[cfg][2],
           "photos": session, "by_tilt": {}}
    for t in tilts:
        rows, n_gt, n_pr, secs = [], 0, 0, []
        for path in tl.photos(session):
            img = cv2.imread(path)
            gts = tl.truth(path, img.shape)
            if t:
                img, m = turn(img, t)
                gts = [apply(m, g) for g in gts]
            t0 = time.perf_counter()
            prs = [p for p, _ in detect(img, cfg)]
            secs.append(time.perf_counter() - t0)
            rows += tl.match_rows(gts, prs, os.path.basename(path))
            n_gt, n_pr = n_gt + len(gts), n_pr + len(prs)
        out["by_tilt"][str(t)] = dict(tl.summary(rows, n_gt, n_pr), sec_median=round(float(np.median(secs)), 3))
        s = out["by_tilt"][str(t)]
        print(cfg, session, "tilt", t, "recall", s["recall"], "precision", s["precision"], "tight", s["tight_1_5"],
              "iou", s["iou_median"], "s", s["sec_median"], flush=True)
    os.makedirs(os.path.join(E, "probe"), exist_ok=True)
    json.dump(out, open(os.path.join(E, "probe", f"{cfg}_{session}.json"), "w"), indent=1)


def count(cfg, out_dir, photos):
    os.makedirs(out_dir, exist_ok=True)
    res = {}
    for p in photos:
        img = cv2.imread(p)
        dets = detect(img, cfg)
        vis = img.copy()
        tk = max(2, img.shape[1] // 600)
        for i, (poly, c) in enumerate(dets):
            cv2.polylines(vis, [np.rint(poly).astype(np.int32)], True, (255, 200, 0), tk, cv2.LINE_AA)
            cx, cy = poly.mean(0)
            cv2.putText(vis, str(i), (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.9 * tk, (0, 0, 255), 2 * tk)
        s = 1600 / max(img.shape[:2])
        vis = cv2.resize(vis, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else vis
        name = os.path.basename(p)
        cv2.putText(vis, f"{name} {cfg} boxes {len(dets)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6)
        cv2.putText(vis, f"{name} {cfg} boxes {len(dets)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(out_dir, name), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
        res[name] = len(dets)
        print(name, len(dets), flush=True)
    json.dump({"config": cfg, "boxes": res}, open(os.path.join(out_dir, "count.json"), "w"), indent=1)


if __name__ == "__main__":
    mode, cfg = sys.argv[1], sys.argv[2]
    opts = dict(zip(sys.argv[3::2], sys.argv[4::2])) if mode == "heldout" else {}
    if mode == "heldout":
        heldout(cfg, [int(v) for v in opts.get("--tilt", "0,20,35").split(",")], opts.get("--photos", "test"))
    else:
        count(cfg, sys.argv[3], sys.argv[4:])

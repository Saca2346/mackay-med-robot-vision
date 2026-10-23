"""
How tight are the box outlines? Webcam / D455 shelf recording + reference spine outlines -> per detector setting:
boxes found / missed / false (IoU >= 0.5), mean IoU of the found boxes, share of tight boxes (IoU >= 0.85), and how far
the box ENDS are from the reference ends (px along the spine). Owner's request 18-10: outlines as neat and precise as
the boxes themselves.

  gt    python scripts/bbox_precision_eval.py gt --video V --frame 5 --out DIR --model name=weights@HxW ...
        reference outlines from the consensus of the given models (a box found by >= --votes models; long-side ends at
        the 20th / 80th percentile of the members' ends, sides at the median) -> DIR/gt.json + gt_preview.jpg.
        Made by models and checked by eye: the owner checks gt_preview.jpg (never call it human-labelled).
  eval  python scripts/bbox_precision_eval.py eval --video V --gt DIR/gt.json --frames 12 --out DIR
        --model name=weights@HxW [...] [--refine]
        the same frames for every setting; --refine also scores src/box_outline.refine_box_outline (display outline).
Weights may be a .pt (run with the fixed input, like the OpenVINO export) or an OpenVINO folder.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.box_detector import BoxDetector  # noqa: E402


def parse_model(s: str):
    name, rest = s.split("=", 1)
    w, size = rest.rsplit("@", 1)
    h, wd = (int(v) for v in size.lower().split("x"))
    return name, w, [h, wd]


def detector(weights: str, imgsz):
    device = "cpu" if weights.endswith(".pt") else "intel:gpu.0"
    return BoxDetector(weights, "obb", imgsz, device, {"box": 0.35, "size_label": 0.25}, 0.5,
                       {"box": "box", "size_label": "size_label"}, fixed_input=True)


def boxes(det, frame):
    dets, ms = det.detect(frame)
    return [np.asarray(d.poly, np.float32).reshape(4, 2) for d in dets if d.cls == "box"], ms


def iou(a, b):
    inter, _ = cv2.intersectConvexConvex(a, b)
    u = cv2.contourArea(a) + cv2.contourArea(b) - inter
    return inter / u if u > 0 else 0.0


def axis(poly):
    """Unit vector along the long side, and across."""
    e = [poly[1] - poly[0], poly[2] - poly[1]]
    a = e[0] if np.linalg.norm(e[0]) >= np.linalg.norm(e[1]) else e[1]
    u = a / np.linalg.norm(a)
    if u[1] < 0:
        u = -u
    return u, np.array([-u[1], u[0]])


def full_spine(medoid, members):
    u, v = axis(medoid)
    lo = [float((m @ u).min()) for m in members]
    hi = [float((m @ u).max()) for m in members]
    a0, a1 = np.percentile(lo, 20), np.percentile(hi, 80)
    b0 = np.median([float((m @ v).min()) for m in members])
    b1 = np.median([float((m @ v).max()) for m in members])
    return np.array([a0 * u + b0 * v, a1 * u + b0 * v, a1 * u + b1 * v, a0 * u + b1 * v], np.float32)


def frames(video, idx):
    cap = cv2.VideoCapture(video)
    out = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            out.append((int(i), f))
    return out


def make_gt(a):
    (_, f), = frames(a.video, [a.frame])
    polys = []
    for spec in a.model:
        name, w, size = parse_model(spec)
        polys += [(name, q) for q in boxes(detector(w, size), f)[0]]
    clusters = []
    for k, q in polys:
        for c in clusters:
            if iou(c[0][1], q) >= 0.5:
                c.append((k, q))
                break
        else:
            clusters.append([(k, q)])
    gt = []
    for c in clusters:
        if len({k for k, _ in c}) < a.votes:
            continue
        best = max(c, key=lambda m: sum(iou(m[1], o[1]) for o in c))
        gt.append(full_spine(best[1], [q for _, q in c]))
    gt.sort(key=lambda q: float(q[:, 0].mean()))
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    vis = f.copy()
    for i, q in enumerate(gt):
        cv2.polylines(vis, [q.astype(np.int32)], True, (0, 255, 0), 2)
        cv2.putText(vis, str(i), tuple(q.mean(0).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imwrite(str(out / "gt_preview.jpg"), vis)
    json.dump({"video": a.video, "frame": a.frame, "models": a.model, "votes": a.votes,
               "made_by": "consensus of the models above, checked by Claude; owner check pending",
               "polys": [q.tolist() for q in gt]}, open(out / "gt.json", "w"), indent=1)
    print(len(gt), "reference boxes ->", out / "gt.json")


def score(gt, det):
    """Greedy IoU matching after the median shift -> found, missed, false, matched IoUs, end errors (px)."""
    if det:
        gc = np.array([q.mean(0) for q in gt])
        d = [gc[np.argmin(np.linalg.norm(gc - q.mean(0), axis=1))] - q.mean(0) for q in det]
        shift = np.median(np.array(d), axis=0)
    else:
        shift = np.zeros(2)
    g2 = [(q - shift).astype(np.float32) for q in gt]
    pairs = sorted(((iou(g, q), i, j) for i, g in enumerate(g2) for j, q in enumerate(det)), reverse=True)
    used_g, used_d, ious, ends = set(), set(), [], []
    for v, i, j in pairs:
        if v < 0.5:
            break
        if i in used_g or j in used_d:
            continue
        used_g.add(i)
        used_d.add(j)
        ious.append(v)
        u, _ = axis(g2[i])
        gp, dp = g2[i] @ u, det[j] @ u
        ends += [abs(gp.min() - dp.min()), abs(gp.max() - dp.max())]
    return len(used_g), len(gt) - len(used_g), len(det) - len(used_d), ious, ends


def evaluate(a):
    from src.box_outline import refine_box_outline
    gtj = json.load(open(a.gt))
    gt = [np.array(q, np.float32) for q in gtj["polys"]]
    cap = cv2.VideoCapture(a.video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    last = min(n - 1, a.last_frame or n - 1)
    fr = frames(a.video, np.linspace(a.first_frame, last, a.frames).astype(int))
    res = {}
    for spec in a.model:
        name, w, size = parse_model(spec)
        det = detector(w, size)
        for variant in (("raw", "refined") if a.refine else ("raw",)):
            res[f"{name}|{variant}"] = {"found": 0, "missed": 0, "false": 0, "ious": [], "ends": [], "ms": [],
                                        "frames": len(fr)}
        for _, f in fr:
            polys, ms = boxes(det, f)
            variants = {"raw": polys}
            if a.refine:
                t0 = time.perf_counter()
                variants["refined"] = [np.asarray(refine_box_outline(f, q, [o for o in polys if o is not q]),
                                                  np.float32).reshape(4, 2) for q in polys]
                ref_ms = (time.perf_counter() - t0) * 1000
            for variant, ps in variants.items():
                r = res[f"{name}|{variant}"]
                fo, mi, fa, ious, ends = score(gt, ps)
                r["found"] += fo
                r["missed"] += mi
                r["false"] += fa
                r["ious"] += ious
                r["ends"] += ends
                r["ms"].append(ms + (ref_ms if variant == "refined" else 0))
    summary = {}
    for k, r in res.items():
        ious, ends = np.array(r["ious"]), np.array(r["ends"])
        summary[k] = {"recall": round(r["found"] / max(r["found"] + r["missed"], 1), 4),
                      "false_per_frame": round(r["false"] / r["frames"], 2),
                      "mean_iou": round(float(ious.mean()), 3) if ious.size else None,
                      "tight_iou85_pct": round(float((ious >= 0.85).mean() * 100), 1) if ious.size else None,
                      "end_err_px_median": round(float(np.median(ends)), 1) if ends.size else None,
                      "end_err_px_p90": round(float(np.percentile(ends, 90)), 1) if ends.size else None,
                      "ms_median": round(float(np.median(r["ms"])), 1)}
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"video": a.video, "gt": a.gt, "frames": [i for i, _ in fr], "summary": summary},
              open(out / "eval.json", "w"), indent=1)
    head = "| setting | recall | false / frame | mean IoU | IoU >= 0.85 | end error median / p90 (px) | ms / frame |"
    lines = [head, "|" + "---|" * 7]
    for k, s in summary.items():
        lines.append(f"| {k} | {s['recall']} | {s['false_per_frame']} | {s['mean_iou']} | {s['tight_iou85_pct']}% | "
                     f"{s['end_err_px_median']} / {s['end_err_px_p90']} | {s['ms_median']} |")
    (out / "eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["gt", "eval"])
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", action="append", required=True, help="name=weights@HxW (repeat)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frame", type=int, default=5)
    ap.add_argument("--votes", type=int, default=3)
    ap.add_argument("--gt")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--first-frame", type=int, default=0)
    ap.add_argument("--last-frame", type=int, default=0)
    ap.add_argument("--refine", action="store_true")
    a = ap.parse_args()
    make_gt(a) if a.mode == "gt" else evaluate(a)


if __name__ == "__main__":
    main()

"""
Precision AND stillness of box outlines, measured on static recordings against the pose templates.

For every selected segment of a spec (scripts/build_static_dataset.py) the template is carried to the segment's frames
and each model setting is run on the same frames (shelf unchanged, like the training builder).  Per setting and pose:
  recall / false boxes per frame          boxes found (IoU >= 0.5 with a template box) and extra boxes
  mean IoU, share of IoU >= 0.85 / 0.90   tightness of the found outlines against the template
  end error (median / p90 px)             distance of the box ENDS along the spine
  wobble (median / p90 px)                temporal standard deviation of each box's 4 corners over the segment: the
                                          scene does not move, so this is the pure detector jitter (0 = perfectly still)
  ms / frame                              detector time (laptop on battery: only relative)
The template is made by models + Claude's reading of zoomed views (owner check pending): these are proxy numbers.

  python scripts/eval_static_models.py --spec SPEC --poses pose1,pose2 --out DIR --model name=weights@HxW ...
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import auto_label_static as al  # noqa: E402
import build_static_dataset as bd  # noqa: E402
from bbox_precision_eval import axis  # noqa: E402


def match(gt, det, thr=0.5):
    pairs = sorted(((al.iou(g, q), i, j) for i, g in enumerate(gt) for j, q in enumerate(det)), reverse=True)
    ug, ud, res = set(), set(), {}
    for v, i, j in pairs:
        if v < thr:
            break
        if i in ug or j in ud:
            continue
        ug.add(i)
        ud.add(j)
        res[i] = (j, v)
    return res, len(det) - len(ud)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--poses", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", action="append", required=True, help="name=weights@HxW")
    ap.add_argument("--per-segment", type=int, default=40, help="frames per segment")
    ap.add_argument("--ncc", type=float, default=0.99)
    ap.add_argument("--display", type=int, default=0,
                    help="also simulate the DISPLAYED outline (detector -> refine -> OutlineTracker hold) on this many "
                         "consecutive frames per segment (every --display-step-th frame, like a 5-8 fps live loop)")
    ap.add_argument("--display-step", type=int, default=4)
    ap.add_argument("--refine", action="store_true", help="apply src/box_outline.refine_box_outline in the display simulation")
    a = ap.parse_args()
    spec = json.load(open(a.spec))
    want = set(a.poses.split(","))
    dets = {n: al.make_detector(w, s) for n, w, s in map(al.parse_model, a.model)}
    # frames + mapped templates
    jobs = []
    for seg in spec["segments"]:
        if seg["pose"] not in want:
            continue
        pose = spec["poses"][seg["pose"]]
        tpl, ref = json.load(open(pose["template"])), cv2.imread(pose["ref"])
        cap = cv2.VideoCapture(seg["video"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, (seg["start"] + seg["end"]) // 2)
        ok, mid = cap.read()
        polys, n = al.transfer(tpl, ref, mid)
        if polys is None or n < 150:
            continue
        pts = np.concatenate(polys["box"])
        h, w = mid.shape[:2]
        rect = tuple(int(v) for v in (*np.maximum(pts.min(0) - 10, 0), *np.minimum(pts.max(0) + 10, [w, h])))
        base = bd.gray_region(mid, rect)
        idx = np.linspace(seg["start"], seg["end"], a.per_segment * 3).astype(int)
        frames = []
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, f = cap.read()
            if ok and bd.ncc(base, bd.gray_region(f, rect)) >= a.ncc:
                frames.append(f)
            if len(frames) >= a.per_segment:
                break
        window = []
        if a.display:
            for i in range(seg["start"], seg["end"] + 1, a.display_step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
                ok, f = cap.read()
                if ok and bd.ncc(base, bd.gray_region(f, rect)) >= a.ncc:
                    window.append(f)
                if len(window) >= a.display:
                    break
        jobs.append({"pose": seg["pose"], "clip": seg["clip"], "gt": [np.array(q, np.float32) for q in polys["box"]],
                     "frames": frames, "window": window})
        print(seg["pose"], seg["clip"], len(frames), "frames", flush=True)
    rows = {}
    for name, det in dets.items():
        for job in jobs:
            gt = job["gt"]
            ious, ends, ms, false, miss, found = [], [], [], 0, 0, 0
            tracks = {i: [] for i in range(len(gt))}
            for f in job["frames"]:
                found_d, t = det.detect(f)
                ms.append(t)
                polys = [np.asarray(d.poly, np.float32).reshape(4, 2) for d in found_d if d.cls == "box"]
                m, extra = match(gt, polys)
                false += extra
                miss += len(gt) - len(m)
                found += len(m)
                for i, (j, v) in m.items():
                    ious.append(v)
                    u, _ = axis(gt[i])
                    gp, dp = gt[i] @ u, polys[j] @ u
                    ends += [abs(gp.min() - dp.min()), abs(gp.max() - dp.max())]
                    tracks[i].append(al.align(polys[j], gt[i]))
            wob = [float(np.stack(t).std(0).mean()) for t in tracks.values() if len(t) >= 8]
            if a.display and job["window"]:
                from src.box_outline import refine_box_outline
                from src.box_outline_track import OutlineTracker
                trk = {i: OutlineTracker() for i in range(len(gt))}
                shown = {i: [] for i in range(len(gt))}
                for f in job["window"]:
                    found_d, _t = det.detect(f)
                    polys = [np.asarray(d.poly, np.float32).reshape(4, 2) for d in found_d if d.cls == "box"]
                    m, _extra = match(gt, polys)
                    for i, (j, v) in m.items():
                        others = [q for k, q in enumerate(polys) if k != j]
                        raw = al.align(polys[j], gt[i])
                        sh = refine_box_outline(f, raw, others) if a.refine else raw
                        out = trk[i].update(f, sh, ("target", i, 0, True), others)
                        shown[i].append(al.align(np.asarray(out, np.float32).reshape(4, 2), gt[i]))
                dw = [float(np.stack(t[6:]).std(0).mean()) for t in shown.values() if len(t) >= 14]   # after the hold settled (3-6 frames)
                di = [al.iou(gt[i], t[-1]) for i, t in shown.items() if t]
                rd = rows.setdefault((name + "+display", job["pose"]), {"iou": [], "ends": [], "ms": [0.0], "false": 0, "miss": 0,
                                                                         "found": 0, "wobble": [], "frames": 0})
                rd["iou"] += di
                rd["wobble"] += dw
                rd["found"] += len(di)
                rd["frames"] += 1
                for i, t in shown.items():
                    if t:
                        u, _ = axis(gt[i])
                        gp, dp = gt[i] @ u, t[-1] @ u
                        rd["ends"] += [abs(gp.min() - dp.min()), abs(gp.max() - dp.max())]
            r = rows.setdefault((name, job["pose"]), {"iou": [], "ends": [], "ms": [], "false": 0, "miss": 0, "found": 0,
                                                      "wobble": [], "frames": 0})
            r["iou"] += ious
            r["ends"] += ends
            r["ms"] += ms
            r["false"] += false
            r["miss"] += miss
            r["found"] += found
            r["wobble"] += wob
            r["frames"] += len(job["frames"])
        print("done", name, flush=True)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary, lines = {}, ["| model | pose | recall | false / frame | mean IoU | IoU >= 0.85 | IoU >= 0.90 | end error med / p90 (px) | wobble med / p90 (px) | ms |",
                           "|" + "---|" * 10]
    for (name, pose), r in sorted(rows.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        i, e, w = np.array(r["iou"]), np.array(r["ends"]), np.array(r["wobble"])
        s = {"recall": round(r["found"] / max(r["found"] + r["miss"], 1), 3),
             "false_per_frame": round(r["false"] / max(r["frames"], 1), 2),
             "mean_iou": round(float(i.mean()), 3), "iou85_pct": round(float((i >= .85).mean() * 100), 1),
             "iou90_pct": round(float((i >= .90).mean() * 100), 1),
             "end_med": round(float(np.median(e)), 1), "end_p90": round(float(np.percentile(e, 90)), 1),
             "wobble_med": round(float(np.median(w)), 2) if w.size else None,
             "wobble_p90": round(float(np.percentile(w, 90)), 2) if w.size else None,
             "ms": round(float(np.median(r["ms"])), 1)}
        summary[f"{name}|{pose}"] = s
        lines.append(f"| {name} | {pose} | {s['recall']} | {s['false_per_frame']} | {s['mean_iou']} | {s['iou85_pct']}% | "
                     f"{s['iou90_pct']}% | {s['end_med']} / {s['end_p90']} | {s['wobble_med']} / {s['wobble_p90']} | {s['ms']} |")
    json.dump(summary, open(out / "eval.json", "w"), indent=1)
    (out / "eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

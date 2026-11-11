"""
Labels for a static camera pose, made from many frames instead of one.

  consensus   python scripts/auto_label_static.py consensus --video V --frames 12:900:30 --out DIR --model name=weights@HxW ...
              Runs every detector on the sampled frames; a box / size_label / expiry_label object found by at least
              --min-models detectors in at least --min-share of the frames becomes one polygon = the median of the
              aligned corner sets (jitter of the teachers averages out; their shared bias does not: ends are corrected
              afterwards, see transfer). -> DIR/consensus.json + DIR/consensus_preview.jpg

The result is a PROPOSAL made by models; nothing here is human-reviewed (the owner checks the previews).
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

from src.box_detector import BoxDetector  # noqa: E402

CLASSES = ("box", "size_label", "expiry_label")


def parse_model(s):
    name, rest = s.split("=", 1)
    w, size = rest.rsplit("@", 1)
    h, wd = (int(v) for v in size.lower().split("x"))
    return name, w, [h, wd]


def make_detector(w, size):
    device = "cpu" if w.endswith(".pt") else "intel:gpu.0"
    return BoxDetector(w, "obb", size, device, {"box": 0.30, "size_label": 0.25, "expiry_label": 0.25}, 0.5,
                       {c: c for c in CLASSES}, fixed_input=True)


def align(poly, ref):
    q = np.asarray(poly, np.float32).reshape(4, 2)
    ch = [np.roll(o, k, axis=0) for o in (q, q[::-1]) for k in range(4)]
    return min(ch, key=lambda p: float(np.square(p - ref).sum()))


def iou(a, b):
    inter, _ = cv2.intersectConvexConvex(a, b)
    u = cv2.contourArea(a) + cv2.contourArea(b) - inter
    return inter / u if u > 0 else 0.0


def sample_frames(video, spec):
    a, b, step = (int(v) for v in spec.split(":"))
    cap = cv2.VideoCapture(video)
    out = []
    for i in range(a, b, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if ok:
            out.append((i, f))
    return out


def consensus(frames, models, min_models, min_share, iou_thr=0.5):
    """frames [(i, img)], models [(name, detector)] -> {class: [polygon (4, 2)]} + support."""
    found = {c: [] for c in CLASSES}            # class -> list of (frame, model, poly)
    for fi, img in frames:
        for name, det in models:
            dets, _ = det.detect(img)
            for d in dets:
                if d.cls in found:
                    found[d.cls].append((fi, name, np.asarray(d.poly, np.float32).reshape(4, 2)))
    result, support = {}, {}
    for cls, items in found.items():
        clusters = []
        for fi, name, q in items:
            for c in clusters:
                if iou(c["seed"], q) >= iou_thr:
                    c["m"].append((fi, name, align(q, c["seed"])))
                    break
            else:
                clusters.append({"seed": q, "m": [(fi, name, q)]})
        keep = []
        for c in clusters:
            nm = len({n for _, n, _ in c["m"]})
            share = len({f for f, _, _ in c["m"]}) / max(len(frames), 1)
            if nm >= min_models and share >= min_share:
                polys = np.stack([p for _, _, p in c["m"]])
                keep.append((np.median(polys, axis=0).astype(np.float32),
                             {"models": nm, "frame_share": round(share, 2), "spread_px": round(float(polys.std(0).mean()), 2)}))
        keep.sort(key=lambda t: float(t[0][:, 0].mean()))
        result[cls] = [p for p, _ in keep]
        support[cls] = [s for _, s in keep]
    return result, support


def preview(img, res, path, numbers=True):
    vis = img.copy()
    col = {"box": (0, 255, 0), "size_label": (0, 165, 255), "expiry_label": (255, 0, 255)}
    for cls, polys in res.items():
        for i, q in enumerate(polys):
            cv2.polylines(vis, [np.rint(q).astype(np.int32)], True, col[cls], 2, cv2.LINE_AA)
            if numbers and cls == "box":
                cv2.putText(vis, str(i), tuple(q.mean(0).astype(int)), 0, 0.8, (0, 0, 255), 2)
    cv2.imwrite(str(path), vis)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["consensus"])
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames", required=True, help="start:stop:step")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--min-models", type=int, default=3)
    ap.add_argument("--min-share", type=float, default=0.5)
    a = ap.parse_args()
    frames = sample_frames(a.video, a.frames)
    models = [(n, make_detector(w, s)) for n, w, s in map(parse_model, a.model)]
    res, sup = consensus(frames, models, a.min_models, a.min_share)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"video": a.video, "frames": a.frames, "models": a.model, "n_frames": len(frames),
               "made_by": "model consensus (proposal, owner check pending)",
               "polys": {c: [p.tolist() for p in v] for c, v in res.items()}, "support": sup},
              open(out / "consensus.json", "w"), indent=1)
    preview(frames[len(frames) // 2][1], res, out / "consensus_preview.jpg")
    print({c: len(v) for c, v in res.items()}, "from", len(frames), "frames x", len(models), "models")


if __name__ == "__main__" and not (len(sys.argv) > 1 and sys.argv[1] in ("ends", "zoom")):
    main()


# ---------------------------------------------------------------------------------------------------------------
# ends: set the top / bottom end of every box outline (read by eye on a zoomed view, +-2 px at 1080p), keep its sides
# ---------------------------------------------------------------------------------------------------------------
def with_ends(poly, top_y, bottom_y):
    q = np.asarray(poly, np.float32).reshape(4, 2)
    e = [q[1] - q[0], q[2] - q[1]]
    a = e[0] if np.linalg.norm(e[0]) >= np.linalg.norm(e[1]) else e[1]
    u = a / np.linalg.norm(a)
    if u[1] < 0:
        u = -u
    v = np.array([-u[1], u[0]])
    c = q.mean(0)
    b0, b1 = float((q @ v).min()), float((q @ v).max())
    a0 = float(c @ u + (top_y - c[1]) / u[1])
    a1 = float(c @ u + (bottom_y - c[1]) / u[1])
    return np.array([a0 * u + b0 * v, a1 * u + b0 * v, a1 * u + b1 * v, a0 * u + b1 * v], np.float32)


def ends_cmd(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--consensus", required=True)
    ap.add_argument("--tops", required=True, help="JSON list, one y per box, left to right")
    ap.add_argument("--bottoms", required=True)
    ap.add_argument("--drop-x-above", type=float, default=1e9, help="drop objects whose centre x is beyond this")
    ap.add_argument("--drop-box", default="", help="comma-separated box indices to drop before the ends are set")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    C = json.load(open(a.consensus))
    img = cv2.imread(a.image)
    drop = {int(v) for v in a.drop_box.split(",") if v}
    boxes = [np.array(q, np.float32) for i, q in enumerate(C["polys"]["box"]) if i not in drop]
    boxes = [q for q in boxes if q[:, 0].mean() < a.drop_x_above]
    tops, bottoms = json.loads(a.tops), json.loads(a.bottoms)
    assert len(boxes) == len(tops) == len(bottoms), (len(boxes), len(tops), len(bottoms))
    out = {"box": [with_ends(q, t, b) for q, t, b in zip(boxes, tops, bottoms)]}
    for cls in ("size_label", "expiry_label"):
        out[cls] = [np.array(q, np.float32) for q in C["polys"][cls] if np.mean(np.array(q)[:, 0]) < a.drop_x_above]
    H, W = img.shape[:2]
    for cls in out:
        out[cls] = [np.clip(q, [0, 0], [W - 1, H - 1]) for q in out[cls]]
    outp = pathlib.Path(a.out)
    outp.mkdir(parents=True, exist_ok=True)
    json.dump({"video": C["video"], "image_size": [W, H], "from": a.consensus,
               "made_by": "model consensus (sides, size / expiry labels) + top / bottom ends read by Claude on a zoomed "
                          "view (+-2 px); owner check pending",
               "tops": tops, "bottoms": bottoms,
               "polys": {c: [q.tolist() for q in v] for c, v in out.items()}}, open(outp / "template.json", "w"), indent=1)
    preview(img, out, outp / "template_preview.jpg")
    print({c: len(v) for c, v in out.items()})


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "ends":
    ends_cmd(sys.argv[2:])
    sys.exit(0)


# ---------------------------------------------------------------------------------------------------------------
# transfer: carry a template from its reference frame to another pose (feature homography of the whole scene)
# ---------------------------------------------------------------------------------------------------------------
def homography(ref_img, img, mask_ref=None, nfeat=6000):
    orb = cv2.ORB_create(nfeat)
    g0, g1 = (cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in (ref_img, img))
    k0, d0 = orb.detectAndCompute(g0, mask_ref)
    k1, d1 = orb.detectAndCompute(g1, None)
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d0, d1)
    if len(m) < 30:
        return None, 0
    p0 = np.float32([k0[x.queryIdx].pt for x in m])
    p1 = np.float32([k1[x.trainIdx].pt for x in m])
    h, inl = cv2.findHomography(p0, p1, cv2.RANSAC, 2.0)
    return h, (int(inl.sum()) if inl is not None else 0)


def shelf_mask(shape, boxes, pad=40):
    """Features only from the shelf row (the boxes and a margin): the plane the outlines live on."""
    m = np.zeros(shape[:2], np.uint8)
    pts = np.concatenate(boxes).astype(np.int32)
    x0, y0 = pts.min(0) - pad
    x1, y1 = pts.max(0) + pad
    m[max(y0, 0):y1, max(x0, 0):x1] = 255
    return m


def transfer(template, ref_img, img):
    boxes = [np.array(q, np.float32) for q in template["polys"]["box"]]
    h, n = homography(ref_img, img, shelf_mask(ref_img.shape, boxes))
    if h is None or n < 60:
        return None, n
    out = {c: [cv2.perspectiveTransform(np.array(q, np.float32).reshape(1, 4, 2), h)[0] for q in v]
           for c, v in template["polys"].items()}
    return out, n


def zoom_cmd(argv):
    """Zoomed top / bottom bands of the consensus outlines with a y ruler, to read the box ends by eye."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--consensus", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--drop-x-above", type=float, default=1e9)
    ap.add_argument("--drop-box", default="")
    a = ap.parse_args(argv)
    C = json.load(open(a.consensus))
    img = cv2.imread(a.image)
    drop = {int(v) for v in a.drop_box.split(",") if v}
    boxes = [np.array(q, np.float32) for i, q in enumerate(C["polys"]["box"]) if i not in drop]
    boxes = [q for q in boxes if q[:, 0].mean() < a.drop_x_above]
    xs = np.concatenate(boxes)[:, 0]
    x0, x1 = int(xs.min()) - 15, int(xs.max()) + 15
    mid = (x0 + x1) // 2 + 20
    tops = np.array([q[:, 1].min() for q in boxes])
    bots = np.array([q[:, 1].max() for q in boxes])
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print("boxes", len(boxes), "x", x0, x1, "top range", int(tops.min()), int(tops.max()), "bottom range", int(bots.min()), int(bots.max()))

    def render(name, ya, yb, xa, xb):
        ya, yb = max(ya, 0), min(yb, img.shape[0])
        fx = min(1500 / (xb - xa), 3.0)
        crop = cv2.resize(img[ya:yb, xa:xb].copy(), None, fx=fx, fy=fx, interpolation=cv2.INTER_CUBIC)
        for yy in range((ya // 10 + 1) * 10, yb, 10):
            y = int((yy - ya) * fx)
            cv2.line(crop, (0, y), (16 if yy % 50 else 40, y), (0, 0, 255), 1)
            if yy % 20 == 0:
                cv2.putText(crop, str(yy), (44, y + 5), 0, 0.45, (0, 0, 255), 1)
        for i, q in enumerate(boxes):
            cv2.polylines(crop, [((q - [xa, ya]) * fx).astype(np.int32)], True, (0, 255, 0), 1, cv2.LINE_AA)
            cx = q[:, 0].mean()
            if xa < cx < xb:
                cv2.putText(crop, str(i), (int((cx - xa) * fx) - 10, 20), 0, 0.8, (255, 0, 255), 2)
        cv2.imwrite(str(out / f"{name}.png"), crop)

    for nm, (xa, xb) in (("L", (x0, mid)), ("R", (mid - 20, x1))):
        render("top_" + nm, int(tops.min()) - 40, int(tops.max()) + 70, xa, xb)
        render("bot_" + nm, int(bots.min()) - 70, int(bots.max()) + 40, xa, xb)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "zoom":
    zoom_cmd(sys.argv[2:])
    sys.exit(0)

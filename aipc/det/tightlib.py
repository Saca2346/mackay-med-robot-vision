"""Shared outline-accuracy code (4 Nov 2026): truth labels, polygon IoU, matching, per-edge errors, summary, edge snap."""
import os

import cv2
import numpy as np

M = os.path.expanduser("~/Downloads/prelabel_size_label_full/export/cv/all/images/train")


def photos(session):
    """'test' = 28-Sep session (held out), 'dev' = 23-Sep + 1-Oct (training photos, for tuning only)."""
    fs = sorted(os.path.join(M, f) for f in os.listdir(M))
    test = [f for f in fs if os.path.basename(f).startswith("IMG_20260928")]
    return test if session == "test" else [f for f in fs if f not in test]


def truth(path, shape):
    lab = path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
    h, w = shape[:2]
    return [np.array(v[1:9], float).reshape(4, 2) * [w, h] for v in (l.split() for l in open(lab)) if v and v[0] == "0"]


def iou(a, b):
    a, b = cv2.convexHull(a.astype(np.float32)), cv2.convexHull(b.astype(np.float32))
    inter, _ = cv2.intersectConvexConvex(a, b)
    union = cv2.contourArea(a) + cv2.contourArea(b) - inter
    return inter / union if union > 0 else 0.0


def greedy(m, thr=0.5):
    pairs, ug, up = [], set(), set()
    for k in np.argsort(-m, axis=None):
        i, j = np.unravel_index(k, m.shape)
        if m[i, j] < thr:
            break
        if i not in ug and j not in up:
            pairs.append((i, j))
            ug.add(i)
            up.add(j)
    return pairs


def frame_of(poly):
    c = poly.mean(0)
    e1, e2 = poly[1] - poly[0], poly[2] - poly[1]
    lg = e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2
    u = lg / np.linalg.norm(lg)
    v = np.array([-u[1], u[0]])
    du, dv = (poly - c) @ u, (poly - c) @ v
    return c, u, v, (du.max() - du.min()) / 2, (dv.max() - dv.min()) / 2


def errors(gt, pr):
    c, u, v, L, W = frame_of(gt)
    pu, pv = (pr - c) @ u, (pr - c) @ v
    return [pu.max() - L, -pu.min() - L], [pv.max() - W, -pv.min() - W], 2 * L, 2 * W


def match_rows(gts, prs, photo=""):
    rows = []
    if gts and prs:
        m = np.array([[iou(g, q) for q in prs] for g in gts])
        for i, j in greedy(m):
            e, s, length, width = errors(gts[i], prs[j])
            rows.append({"photo": photo, "iou": float(m[i, j]), "ends": e, "sides": s, "length": float(length),
                         "width": float(width)})
    return rows


def summary(rows, n_gt, n_pr, exact=None):
    ends = np.array([r["ends"] for r in rows])
    sides = np.array([r["sides"] for r in rows])
    L = np.array([r["length"] for r in rows])[:, None]
    W = np.array([r["width"] for r in rows])[:, None]
    ious = np.array([r["iou"] for r in rows])
    pct = lambda x: round(100 * float(np.mean(x)), 1)  # noqa: E731
    out = {"truth_boxes": n_gt, "pred_boxes": n_pr, "matched": len(rows), "recall": round(len(rows) / max(n_gt, 1), 4),
           "precision": round(len(rows) / max(n_pr, 1), 4), "iou_median": round(float(np.median(ious)), 4),
           "iou_ge_090": pct(ious >= 0.90), "iou_ge_095": pct(ious >= 0.95),
           "end_abs_px_median": round(float(np.median(np.abs(ends))), 1),
           "end_abs_pct_median": round(float(np.median(np.abs(ends) / L)) * 100, 2),
           "end_over_1pct": pct((ends / L > 0.01).any(1)), "end_short_1pct": pct((ends / L < -0.01).any(1)),
           "side_abs_px_median": round(float(np.median(np.abs(sides))), 1),
           "side_over_5pct": pct((sides / W > 0.05).any(1)), "side_short_5pct": pct((sides / W < -0.05).any(1)),
           "tight_1_5": pct((np.abs(ends) / L <= 0.01).all(1) & (np.abs(sides) / W <= 0.05).all(1)),
           "tight_2_10": pct((np.abs(ends) / L <= 0.02).all(1) & (np.abs(sides) / W <= 0.10).all(1))}
    if exact is not None:
        out["photos_exact_count"] = exact
    return out


def gradients(img, blur=1.5):
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), blur)
    return cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)


def snap(poly, gx, gy, end_range=0.03, side_range=0.25, lam=0.0, samples=48, side_bias=0.0):
    """Move each of the 4 edges of a predicted box along its normal to the strongest straight edge in the photo.

    The box stays a rectangle (each edge only shifts parallel to itself). end_range / side_range: search distance as
    a fraction of the box length / width; lam: penalty per pixel of shift, as a fraction of the profile maximum;
    side_bias: each side moved by this fraction of the box width after the search (chosen on dev photos only)."""
    c, u, v, L, W = frame_of(poly)
    out = {}
    for axis, half, other, rng in ((u, L, W, end_range * 2 * L), (v, W, L, side_range * 2 * W)):
        perp = v if axis is u else u
        ts = np.linspace(-0.8 * other, 0.8 * other, samples)
        if rng <= 0:                                   # this pair of edges is left as predicted
            out[(id(axis), 1)] = out[(id(axis), -1)] = 0.0
            continue
        R = max(2, int(round(rng)))
        ds = np.arange(-R, R + 1, dtype=np.float32)
        for sign in (1, -1):
            n = axis * sign
            base = c + n * half
            pts = base[None, None, :] + ts[None, :, None] * perp[None, None, :] + ds[:, None, None] * n[None, None, :]
            mx, my = pts[..., 0].astype(np.float32), pts[..., 1].astype(np.float32)
            d = cv2.remap(gx, mx, my, cv2.INTER_LINEAR) * n[0] + cv2.remap(gy, mx, my, cv2.INTER_LINEAR) * n[1]
            prof = np.abs(d.mean(axis=1))
            prof = prof - lam * prof.max() * np.abs(ds) / max(R, 1)
            out[(id(axis), sign)] = float(ds[int(np.argmax(prof))])
    e_pos, e_neg = out[(id(u), 1)], out[(id(u), -1)]
    s_pos, s_neg = out[(id(v), 1)], out[(id(v), -1)]
    s_pos, s_neg = s_pos + side_bias * 2 * W, s_neg + side_bias * 2 * W     # constant side correction (- = tighter)
    a_u, b_u = -(L + e_neg), L + e_pos
    a_v, b_v = -(W + s_neg), W + s_pos
    return np.array([c + a_u * u + a_v * v, c + b_u * u + a_v * v, c + b_u * u + b_v * v, c + a_u * u + b_v * v])

"""Ends of the drawn PICK outline on webcam stills (8 Nov 2026): every green PICK of a Task 3 / live run log, its logged
outline (pick_poly) next to what the learned refiner and the edge snap would draw from it. No truth exists for webcam
frames (the static-pose labels are templates, not reviewed), so this writes zooms of both ends for a visual check and
the end offsets (px, outward positive, along the box's long axis) per candidate.

  cd aipc_20261104/tasks && PYTHONPATH=../det python webcam_ends.py OUT_DIR RUN_JSON [RUN_JSON ...]
      RUN_JSON = out/<tag>/task3.json or out/<tag>/live_<set>.json (the live set is found in ../livesets/<set>)
"""
import glob
import json
import os
import sys

import cv2
import numpy as np

import refine_lib as rl
import tightlib as tl

H = os.path.dirname(os.path.abspath(__file__))
SNAP = dict(blur=1.5, end=0.02, lam=0.6, bias=-0.02)          # aipc_tasks.SNAP
T3 = os.path.join(H, "../code/results/tasks_v2_20261029/task3/live_20261029_155853/cases")
COL = {"logged": (60, 200, 0), "refiner": (230, 40, 230), "snap": (0, 140, 255), "rule": (255, 255, 0)}


def snap(frame, poly):
    p = np.asarray(poly, np.float32)
    pad = int(0.05 * np.ptp(p, axis=0).max()) + 16
    x0, y0 = np.maximum(np.floor(p.min(0)) - pad, 0).astype(int)
    x1, y1 = np.minimum(np.ceil(p.max(0)) + pad, [frame.shape[1], frame.shape[0]]).astype(int)
    gx, gy = tl.gradients(frame[y0:y1, x0:x1], SNAP["blur"])
    return tl.snap(p - [x0, y0], gx, gy, SNAP["end"], 0.0, SNAP["lam"], side_bias=SNAP["bias"]) + [x0, y0]


def ends_by_colour(img, p, q):
    """--webcam-ends 1 (8 Nov, AI PC, drawing only): the 1280 outline p, each END moved to the refiner's end (q) only
    when the band between the two looks like the side the end joins. Extending: the band is closer in colour (mean Lab)
    to the box just inside the old end than to what lies just beyond the new end. Shrinking: the band is closer to what
    lies just beyond the old end than to the box just inside the new end. An end whose bands reach past the photo edge
    (a box cut by the frame) is kept. No threshold to set; the sides stay p's.
    Why: on webcam stills the 1280 outline stops short of a dark box bottom (Accuforce, 20-35 px) and the refiner, trained
    on phone photos only, runs the top end into the shelf bar or the blinds. Rule designed after seeing the Task 3 and
    recording stills (no truth exists for webcam frames)."""
    p = np.asarray(p, np.float64)
    c, u, v, L, W = tl.frame_of(p)
    off = rl.truth_offsets(p, np.asarray(q, np.float64))           # +u, -u, +v, -v, outward positive (px)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    def mean(sign, t0, t1):                                         # mean Lab of the band t0..t1 along sign*u
        ts = np.linspace(t0, t1, max(2, int(abs(t1 - t0)) + 1))
        ws = np.linspace(-0.7 * W, 0.7 * W, 15)
        pts = c + (sign * ts)[:, None, None] * u + ws[None, :, None] * v
        if pts[..., 0].min() < 0 or pts[..., 1].min() < 0 or pts[..., 0].max() > img.shape[1] - 1 or \
                pts[..., 1].max() > img.shape[0] - 1:
            return None                     # a band past the photo edge (a box cut by the frame): nothing to compare
        s = cv2.remap(lab, pts[..., 0].astype(np.float32), pts[..., 1].astype(np.float32), cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REPLICATE)
        return s.reshape(-1, 3).astype(np.float64).mean(0)

    new = [0.0, 0.0, 0.0, 0.0]
    for k, sign in ((0, 1.0), (1, -1.0)):
        o = float(off[k])
        if abs(o) < 2.0:
            continue
        a = abs(o)
        if o > 0:   # extend: band L..L+o; joins the box (inside L-a..L) rather than what lies beyond (L+o..L+o+a)
            band, side, away = mean(sign, L, L + o), mean(sign, L - a, L), mean(sign, L + o, L + o + a)
        else:       # shrink: band L+o..L; joins what lies beyond (L..L+a) rather than the box inside (L+o-a..L+o)
            band, side, away = mean(sign, L + o, L), mean(sign, L, L + a), mean(sign, L + o - a, L + o)
        if any(m is None for m in (band, side, away)):
            continue
        if np.linalg.norm(band - side) < np.linalg.norm(band - away):
            new[k] = o
    return rl.rect_from(p, new)


def ends_y(poly):
    """(top y, bottom y) of a near-vertical box: mean y of its two upper / two lower corners."""
    y = np.sort(np.asarray(poly, np.float64)[:, 1])
    return y[:2].mean(), y[2:].mean()


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    ref = rl.Refiner(os.path.join(H, "../det/weights/refiner.pt"), "cuda")
    rows, tiles = [], []
    for rj in sys.argv[2:]:
        d = json.load(open(rj))
        live = "set" in d["summary"]
        sd = os.path.join(H, "../livesets", d["summary"]["set"]) if live else None
        reqs = {x["id"]: x for x in json.load(open(os.path.join(sd, "set.json")))["requests"]} if live else {}
        for r in d["rows"]:
            if not r.get("pick_box") or not r.get("pick_poly"):
                continue
            if live:
                st = reqs[r["id"]].get("photos") or [reqs[r["id"]]["photo"]]
                path = os.path.join(sd, st[r.get("stills_read", 1) - 1])
            else:
                path = sorted(glob.glob(os.path.join(T3, r["id"], "take_*/photos/*_camera.png")))[0]
            img = cv2.imread(path)
            a = np.asarray(r["pick_poly"], np.float64)
            q = np.asarray(ref(img, a), np.float64)
            cand = {"logged": a, "refiner": q, "snap": snap(img, a), "rule": ends_by_colour(img, a, q)}
            name = (d["summary"]["set"] + ":" if live else "t3:") + r["id"]
            ya = ends_y(a)
            rows.append({"name": name, "photo": os.path.relpath(path, H),
                         **{k: [round(ya[0] - ends_y(p)[0], 1), round(ends_y(p)[1] - ya[1], 1)] for k, p in cand.items()},
                         "length": round(ya[1] - ya[0], 1)})
            vis = img.copy()
            for k, p in cand.items():
                cv2.polylines(vis, [np.rint(p).astype(np.int32)], True, COL[k], 1, cv2.LINE_AA)
            x0, x1 = int(a[:, 0].min()) - 30, int(a[:, 0].max()) + 30
            pair = []
            for yc in ya:
                y0 = int(yc) - 50
                t = vis[max(0, y0):y0 + 100, max(0, x0):x1]
                t = cv2.resize(t, (360, int(360 * t.shape[0] / max(1, t.shape[1]))), interpolation=cv2.INTER_CUBIC)
                pair.append(cv2.copyMakeBorder(t, 0, max(0, 200 - t.shape[0]), 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))[:200])
            tile = np.vstack(pair)
            cv2.putText(tile, name, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            tiles.append(cv2.copyMakeBorder(tile, 0, 0, 0, 8, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
    for k in range(0, len(tiles), 8):
        cv2.imwrite(os.path.join(out, f"ends_{k // 8}.jpg"), np.hstack(tiles[k:k + 8] + [np.full_like(tiles[0], 255)] * (8 - len(tiles[k:k + 8]))))
    json.dump(rows, open(os.path.join(out, "ends.json"), "w"), indent=1)
    for x in rows:
        print(x["name"], "top/bottom outward px: refiner", x["refiner"], "snap", x["snap"], "rule", x["rule"],
              "len", x["length"])


if __name__ == "__main__":
    main()

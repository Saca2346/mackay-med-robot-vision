"""
Training set from static-camera recordings: every sampled frame gets the template of its pose.

Spec JSON:
  poses:    name -> {template: template.json (polygons in the ref image), ref: ref image, day: tag, cap: max frames}
  segments: [{clip, video, start, end, pose, step}]
For each segment the template is carried from the pose's reference image to the segment's middle frame by a feature
homography (the camera was only nudged between takes), then every frame is kept only if the shelf region is still
the middle frame's (normalised cross-correlation >= --ncc: no box removed, no hand), so a label never describes a
box that is not there.  Labels: YOLO-OBB, classes 0 box / 1 size_label / 2 expiry_label.

  python scripts/build_static_dataset.py --spec SPEC.json --out DIR  -> DIR/images, DIR/labels, DIR/manifest.json
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

CLS = {"box": 0, "size_label": 1, "expiry_label": 2}


def gray_region(img, rect, scale=0.25):
    x0, y0, x1, y1 = rect
    return cv2.resize(cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_AREA).astype(np.float32)


def ncc(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))


def label_lines(polys, w, h):
    lines = []
    for cls, items in polys.items():
        for q in items:
            q = np.asarray(q, np.float32).reshape(4, 2)
            full = abs(cv2.contourArea(q))
            q = np.clip(q, [0, 0], [w - 1, h - 1]).astype(np.float32)
            area = abs(cv2.contourArea(q))
            if full < 50 or area < 0.25 * full or area < 30:
                continue
            lines.append(f"{CLS[cls]} " + " ".join(f"{v:.6f}" for v in (q / [w, h]).ravel()))
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ncc", type=float, default=0.99)
    ap.add_argument("--min-inliers", type=int, default=150)
    a = ap.parse_args()
    spec = json.load(open(a.spec))
    out = pathlib.Path(a.out)
    for part in ("images", "labels", "preview"):
        (out / part).mkdir(parents=True, exist_ok=True)
    manifest, per_pose = [], {}
    for si, seg in enumerate(spec["segments"]):
        pose = spec["poses"][seg["pose"]]
        tpl = json.load(open(pose["template"]))
        ref = cv2.imread(pose["ref"])
        cap = cv2.VideoCapture(seg["video"])
        mid_i = (seg["start"] + seg["end"]) // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_i)
        ok, mid = cap.read()
        if not ok:
            print("skip (no frame)", seg["clip"])
            continue
        h, w = mid.shape[:2]
        polys, n = al.transfer(tpl, ref, mid)
        if polys is None or n < a.min_inliers:
            print(f"SKIP {seg['clip']} {seg['start']}-{seg['end']}: homography inliers {n}")
            continue
        boxes = np.concatenate(polys["box"])
        shift = float(np.mean([np.linalg.norm(p.mean(0) - np.array(q, np.float32).mean(0))
                               for p, q in zip(polys["box"], tpl["polys"]["box"])]))
        x0, y0 = np.maximum(boxes.min(0).astype(int) - 10, 0)
        x1, y1 = np.minimum(boxes.max(0).astype(int) + 10, [w, h])
        rect = (int(x0), int(y0), int(x1), int(y1))
        base = gray_region(mid, rect)
        idx = list(range(seg["start"], seg["end"] + 1, seg.get("step", 10)))
        kept = []
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, f = cap.read()
            if not ok or f.shape[:2] != (h, w):
                continue
            c = ncc(base, gray_region(f, rect))
            if c >= a.ncc:
                kept.append((i, c))
        per_pose.setdefault(seg["pose"], []).append({"seg": si, "clip": seg["clip"], "polys": polys, "kept": kept,
                                                     "w": w, "h": h, "video": seg["video"]})
        print(f"{seg['clip']} {seg['start']}-{seg['end']} pose {seg['pose']}: inliers {n}, template moved "
              f"{shift:.1f} px, {len(kept)}/{len(idx)} frames unchanged", flush=True)
        cap.release()
    for pname, segs in per_pose.items():
        cap_n = spec["poses"][pname].get("cap", 200)
        quota = max(1, cap_n // max(len(segs), 1))        # spread the cap evenly: many short takes all count
        chosen = []
        for s in segs:
            ks = s["kept"]
            if len(ks) > quota:
                ks = [ks[int(j)] for j in np.linspace(0, len(ks) - 1, quota)]
            chosen += [(s, k) for k in ks]
        last_frame = None
        for s, (i, c) in chosen:
            cap = cv2.VideoCapture(s["video"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, f = cap.read()
            cap.release()
            if not ok:
                continue
            last_frame = (s, f)
            name = f"{spec['poses'][pname]['day']}_{pname}_s{s['seg']:02d}_{i:05d}"
            cv2.imwrite(str(out / "images" / f"{name}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 93])
            lines = label_lines(s["polys"], s["w"], s["h"])
            (out / "labels" / f"{name}.txt").write_text("\n".join(lines) + "\n")
            manifest.append({"name": name, "pose": pname, "day": spec["poses"][pname]["day"], "clip": s["clip"],
                             "frame": i, "ncc": round(c, 4), "labels": len(lines)})
        if last_frame:
            al.preview(last_frame[1], last_frame[0]["polys"], out / "preview" / f"{pname}.jpg")
        print(f"pose {pname}: {len(chosen)} frames written", flush=True)
    json.dump(manifest, open(out / "manifest.json", "w"), indent=1)
    print(len(manifest), "frames,", sum(m["labels"] for m in manifest), "labels")


if __name__ == "__main__":
    main()

"""Which owner-outlined boxes of the held-out 28-Sep photos a config misses (7 Nov 2026): pictures with the missed
truth outlines in red, the found ones in cyan, and per missed box its size, aspect and whether it touches the edge.
  python misses.py CONFIG OUT_DIR"""
import json, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tightlib as tl
from det_probe import detect
cfg, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
rep = []
for path in tl.photos("test"):
    img = cv2.imread(path)
    gts = tl.truth(path, img.shape)
    prs = [p for p, _ in detect(img, cfg)]
    m = np.array([[tl.iou(g, q) for q in prs] for g in gts]) if gts and prs else np.zeros((len(gts), len(prs)))
    hit = {i for i, _ in tl.greedy(m)} if m.size else set()
    miss = [i for i in range(len(gts)) if i not in hit]
    if not miss:
        continue
    vis = img.copy(); tk = max(3, img.shape[1] // 500)
    for q in prs:
        cv2.polylines(vis, [np.rint(q).astype(np.int32)], True, (255, 200, 0), tk)
    h, w = img.shape[:2]
    for i in miss:
        g = gts[i]
        cv2.polylines(vis, [np.rint(g).astype(np.int32)], True, (0, 0, 255), tk * 2)
        best = float(m[i].max()) if m.size else 0.0
        _, _, _, L, W = tl.frame_of(g)
        edge = bool((g[:, 0] < 5).any() or (g[:, 0] > w - 5).any() or (g[:, 1] < 5).any() or (g[:, 1] > h - 5).any())
        rep.append({"photo": os.path.basename(path), "best_iou": round(best, 2), "len_px": round(2 * L), "wid_px": round(2 * W),
                    "touches_edge": edge})
    s = 1400 / max(h, w)
    cv2.imwrite(os.path.join(out, os.path.basename(path)), cv2.resize(vis, None, fx=s, fy=s), [cv2.IMWRITE_JPEG_QUALITY, 85])
json.dump(rep, open(os.path.join(out, "misses.json"), "w"), indent=1)
print("missed", len(rep), "edge", sum(r["touches_edge"] for r in rep), "partial(best iou 0.2-0.5)", sum(0.2 <= r["best_iou"] < 0.5 for r in rep),
      "none(<0.2)", sum(r["best_iou"] < 0.2 for r in rep), "photos", len({r["photo"] for r in rep}))

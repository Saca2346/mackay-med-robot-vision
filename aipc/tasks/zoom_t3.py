"""Task 3 PICK outlines, top and bottom ends (7 Nov 2026, diagnosis): red = 544x960 detection (read), yellow = the
same box detected at 1280 px (drawn input), magenta = refiner on the 1280 outline (drawn now). From out/fq_t3."""
import sys
sys.argv = [sys.argv[0], "--draw-imgsz", "1280", "--refine", "1", "--refine-device", "cuda", "--snap", "1", "--tag", "zoom_t3"]
import aipc_tasks as A  # noqa: E402
import cv2, json, numpy as np, os  # noqa: E401,E402
rows = {r["id"]: r for r in json.load(open(os.path.join(A.H, "out", "fq_t3", "task3.json")))["rows"]}
reqs = {r["id"]: r for r in A.task3_requests()}
tiles = []
for k in ["K01", "K03", "K04", "K05", "K07", "K10", "K16", "K17", "K21"]:
    r = rows[k]
    frame = cv2.imread(reqs[k]["photo"])
    ids = {int(i): v for i, v in r["ids"].items()} if isinstance(r.get("ids"), dict) else None
    i = next(int(j) for j, v in r["ids"].items() if v == r["pick_box"])
    p544 = np.array(r["polys"][i], np.float32)
    p1280 = np.asarray(A.draw_poly(frame, p544), np.float32)
    pref = A.REFINER(frame, p1280).astype(np.float32)
    x0, y0 = np.maximum(p544.min(0) - 60, 0).astype(int)
    x1, y1 = np.minimum(p544.max(0) + 60, [frame.shape[1], frame.shape[0]]).astype(int)
    z = frame.copy()
    for poly, col in ((p544, (40, 40, 230)), (p1280, (0, 220, 255)), (pref, (220, 40, 220))):
        cv2.polylines(z, [np.rint(poly).astype(np.int32)], True, col, 1, cv2.LINE_AA)
    c = z[y0:y1, x0:x1]
    s = 700 / c.shape[0]
    c = cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    cv2.putText(c, k, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    tiles.append(c)
h = max(t.shape[0] for t in tiles)
cv2.imwrite(os.path.join(A.H, "out", "zoom_t3.jpg"), np.hstack([cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 6, cv2.BORDER_CONSTANT) for t in tiles]))
print("ok")

"""Both ends of the drawn PICK outline (pick_poly of a task3 run) on the original shelf still, zoomed x4: K04, K10."""
import json, sys, cv2, numpy as np
run, ids = sys.argv[1], sys.argv[2].split(",")
rows = {r["id"]: r for r in json.load(open(f"out/{run}/task3.json"))["rows"]}
sys.path.insert(0, "."); import glob
tiles = []
for k in ids:
    r = rows[k]
    photo = sorted(glob.glob(f"../code/results/tasks_v2_20261029/task3/live_20261029_155853/cases/{k}/take_*/photos/*_camera.png"))[0]
    im = cv2.imread(photo)
    p = np.array(r["pick_poly"], np.float32)
    cv2.polylines(im, [np.rint(p).astype(np.int32)], True, (60, 200, 0), 1, cv2.LINE_AA)
    x0, x1 = int(p[:, 0].min()) - 25, int(p[:, 0].max()) + 25
    for yc in (p[:, 1].min(), p[:, 1].max()):
        y0 = int(yc) - 40
        t = im[max(0, y0):y0 + 80, max(0, x0):x1]
        t = cv2.resize(t, (400, int(400 * t.shape[0] / t.shape[1])), interpolation=cv2.INTER_CUBIC)
        cv2.putText(t, f"{k} {'top' if yc == p[:, 1].min() else 'bottom'}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        tiles.append(t)
h = max(t.shape[0] for t in tiles)
tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]
cv2.imwrite(f"out/{run}/ends_{'_'.join(ids)}.jpg", np.hstack(tiles))
print("ok", [rows[k]["pick_box"] for k in ids])

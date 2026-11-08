"""What the reader actually read on given boxes of a task photo (7 Nov 2026, diagnosis only).
  python dbg_reads.py SET BOX... [aipc_tasks options]   (keys = this detector's box order of `boxes()`)"""
import sys
set_name = sys.argv[1]
cut = next((k for k, a in enumerate(sys.argv) if a.startswith("--")), len(sys.argv))
idx = [int(a) for a in sys.argv[2:cut]]
sys.argv = [sys.argv[0]] + sys.argv[cut:] + ["--tag", "dbg_reads"]
import aipc_tasks as A  # noqa: E402
import cv2  # noqa: E402
path, request = A.SETS[set_name]
frame = cv2.imread(path)
groups = A.boxes(frame)
keys = A.keys_of(set_name, groups)
gt = {}
for i, g in enumerate(groups):
    if idx and keys[i] not in [f"{set_name}:{k}" for k in idx]:
        continue
    req = A.parse_request(request)
    ev, *_ = A.read_target_evidence(*A.prepare_crops(frame, g, A.cfg), A.reader(), A.cfg, req, A.catalog, attempt=3,
                                    identity_verified=True)
    print("==", i, keys[i], "labels", len(g.labels))
    print("  spine:", (ev.spine or "")[:300])
    print("  text :", ev.text)
    print("  refs :", [(r.text, r.product, r.diameter, r.length) for r in ev.refs])

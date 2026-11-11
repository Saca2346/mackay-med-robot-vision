"""Leo's Drive photos, runs side by side: boxes found, boxes with diameter AND length read (any brand), boxes with
two sources, and the green request per catalogue photo. python3 drive_cmp.py TAG..."""
import json, sys
tags = sys.argv[1:]
D = {t: {r["photo"]: r for r in json.load(open(f"out/{t}/drive.json"))["rows"]} for t in tags}
print("photo".ljust(34), "  ".join(f"{t[:14]:>14}" for t in tags))
print("".ljust(34), "  ".join(f"{'box size 2src G':>14}" for t in tags))
tot = {t: [0, 0, 0, 0] for t in tags}
for p in sorted(D[tags[0]]):
    cells = []
    for t in tags:
        r = D[t].get(p)
        if r is None:
            cells.append(f"{'-':>14}")
            continue
        g = "G" if r["chosen"] else "."
        cells.append(f"{r['boxes']:>4}{r.get("size_read_boxes", 0):>4}{r['two_source_boxes']:>4} {g:>1}".rjust(14))
        tot[t][0] += r["boxes"]; tot[t][1] += r.get("size_read_boxes", 0); tot[t][2] += r["two_source_boxes"]; tot[t][3] += bool(r["chosen"])
    print(p[:34].ljust(34), "  ".join(cells))
print("TOTAL".ljust(34), "  ".join(f"{a:>4}{b:>4}{c:>4}{d:>2}".rjust(14) for a, b, c, d in tot.values()))

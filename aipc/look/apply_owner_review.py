"""Product-look labels from the OWNER's review (review_labels_owner.csv, 4 Nov 2026) instead of the file-name proxy.

ok -> label kept; wrong + a product of the list -> relabelled to it; wrong + OTHER_PRODUCT_NOT_IN_LIST / NOT_A_BOX /
blank / the same label -> dropped (no usable product). Crops not in the review keep their proxy label and are marked
reviewed=false. Writes crops/leo.jsonl here (crop images linked from the original crops folder) and a summary.
"""
import collections
import csv
import json
import os

H = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.expanduser("~/Downloads/claude_offline_20261101/crops")
review = {r["file"]: r for r in csv.DictReader(open(os.path.join(H, "review_labels_owner.csv"), encoding="utf-8"))}
rows = [json.loads(x) for x in open(os.path.join(SRC, "leo.jsonl"))]
labels = {r["label"] for r in rows}
out, why = [], collections.Counter()
for r in rows:
    v = review.get(r["file"])
    if v is None:
        out.append(dict(r, reviewed=False))
        why["not in review (proxy label kept)"] += 1
    elif v["verdict"] == "ok":
        out.append(dict(r, reviewed=True))
        why["ok"] += 1
    elif v["right_label"] in labels and v["right_label"] != r["label"]:
        out.append(dict(r, label=v["right_label"], proxy_label=r["label"], reviewed=True))
        why["relabelled"] += 1
    else:
        why[f"dropped: {v['right_label'] or 'blank'}"] += 1
os.makedirs(os.path.join(H, "crops"), exist_ok=True)
with open(os.path.join(H, "crops", "leo.jsonl"), "w") as f:
    f.writelines(json.dumps(r) + "\n" for r in out)
for name in ("leo", "shelf", "shelf.jsonl"):
    dst = os.path.join(H, "crops", name)
    if not os.path.lexists(dst):
        os.symlink(os.path.join(SRC, name), dst)
summary = {"crops_before": len(rows), "crops_after": len(out), "reviewed_rows": len(review), "counts": dict(why)}
json.dump(summary, open(os.path.join(H, "owner_labels_summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))

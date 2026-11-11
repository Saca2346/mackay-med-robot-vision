"""Task 3 shelf positions: centre x of every box that was VERIFIED in the 29 Oct live run or its 2 Nov replays."""
import collections
import glob
import json
import os

os.chdir(r"C:\Users\sacab\Downloads\mackay_vision\code\results")
pos = collections.defaultdict(list)
srcs = glob.glob("tasks_v2_20261029/task3/live_20261029_155853/cases/K*/take_*/target_view.jsonl") + \
    glob.glob("tasks_final_20261102/replays/*_live/K*/cases/K*/take_*/target_view.jsonl")
for f in srcs:
    k = [p for p in f.replace("\\", "/").split("/") if p.startswith("K")][0]
    ver = [r for r in (json.loads(line) for line in open(f)) if r.get("identity_verified") and r.get("outline_px")]
    if ver:
        o = ver[-1]["outline_px"]
        pos[k].append(round(sum(p[0] for p in o) / 4, 1))
for k in sorted(pos):
    print(k, pos[k])
print(len(pos), "boxes with a verified position;", len(srcs), "takes")
json.dump({k: v for k, v in sorted(pos.items())}, open(r"C:\Users\sacab\AppData\Local\Temp\claude\c--Users-sacab-Downloads-mackay-vision\d18b26c2-f946-4b99-b27a-ca032af00c01\scratchpad\t3pos.json", "w"), indent=1)

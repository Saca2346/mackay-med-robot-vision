"""The held asked boxes (FN) of a score run and a task3 run, with the reason the rule gave."""
import json, sys
s, t3 = sys.argv[1], sys.argv[2]
for name in ["t1_drive", "t2_drive", "t1_orig"]:
    for p in json.load(open(f"out/{s}/{name}_score.json"))["pairs"]:
        if p["cell"] == "FN":
            print(name, p["key"], p["printed"], "|", p["decision"], "|", p["reason"][:110])
for r in json.load(open(f"out/{t3}/task3.json"))["rows"]:
    if r["target"] and r["target"] not in r["green_boxes"]:
        own = [x for x in r["decisions"] if x[0] == r["target"]]
        print("task3", r["id"], r["request"], "|", own[0][1] if own else "-", "|", own[0][2][:110] if own else "not detected")

"""Task 3 runs side by side: PICK time of each right-box green, median, under 2 s, wrong greens."""
import json, sys, statistics as st
T = sys.argv[1:]
D = {t: {r["id"]: r for r in json.load(open(f"out/{t}/task3.json"))["rows"]} for t in T}
S = {t: json.load(open(f"out/{t}/task3.json"))["summary"] for t in T}
print("id   ", "  ".join(t[:11].rjust(11) for t in T))
for k in sorted(D[T[0]]):
    if not k.startswith("K"):
        continue
    print(k.ljust(5), "  ".join(("%.2f" % D[t][k]["pick_s"] if D[t][k]["pick_s"] and D[t][k]["pick_box"] == k else "-").rjust(11) for t in T))
for t in T:
    v = [r["pick_s"] for r in D[t].values() if r["pick_s"] and r["pick_box"] == r["target"]]
    print(t, "green", S[t]["on_shelf_green_right_box"], "wrong", S[t]["wrong_box_green"], "absent", S[t].get("absent_any_green"),
          "median %.2f" % st.median(v), "under2", sum(x < 2 for x in v), "max %.2f" % max(v))

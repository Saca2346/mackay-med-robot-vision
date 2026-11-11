"""Product-look model: machine labels (2 Nov runs v2_*_b0e30_s0-2) vs the OWNER-reviewed labels (4 Nov runs owner_*_b0_s0-2).

  venv/Scripts/python.exe results/id_model_20261102/score_owner.py

Photo split (train Leo main photos, test Leo dup photos): both are scored against the OWNER's label of each test crop
(review_labels_owner.csv), on the crops both runs tested; crops the owner dropped (not a box / other product) are left
out. Shelf split (sealed: our 1 Nov webcam crops, never trained on): truth = the inventory product of the box (laptop
only), for the 8 shelf products that have Leo photos. Writes owner_summary.md / .json next to this file.
"""
import collections
import csv
import json
import pathlib
import statistics

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = ROOT / "results/id_model_20261102"            # runs/, review_labels_owner.csv and the outputs (not in git)
SHELF = {"accuforce": "Terumo_Accuforce", "ryurei": "Terumo_Ryurei", "ultimaster nagomi": "Terumo_UltimasterNagomi",
         "nc emerge": "BostonScientific_NCEmerge", "agent": "BostonScientific_Agent",
         "conqueror nc pro": "APTMedical_ConquerorNCPro", "scoreflex": "OrbusNeich_Scoreflex",
         "sapphire": "OrbusNeich_Sapphire3"}
inv = {r["box_id"]: r["product"] for r in csv.DictReader(open(ROOT / "data/rak/inventaris.csv", encoding="utf-8"))}
review = {r["file"]: r for r in csv.DictReader(open(HERE / "review_labels_owner.csv", encoding="utf-8"))}


def owner_label(file, proxy):
    v = review.get(file)
    if v is None:
        return proxy
    if v["verdict"] == "ok":
        return proxy
    return v["right_label"] if v["right_label"] not in ("", "OTHER_PRODUCT_NOT_IN_LIST", "NOT_A_BOX") else None


def preds(run):
    return [json.loads(x) for x in open(HERE / "runs" / run / "preds.jsonl")]


def photo(run, files):
    ps = [p for p in preds(run) if p["file"] in files]
    ok = [p["pred"] == files[p["file"]] for p in ps]
    sure = [o for p, o in zip(ps, ok) if p["p"] >= 0.7]
    return {"crops": len(ps), "acc": 100 * sum(ok) / len(ok), "wrong_at_0.7": len(sure) - sum(sure),
            "coverage_0.7": 100 * len(sure) / len(ps)}


def shelf(run):
    out = {}
    by = collections.defaultdict(list)
    for p in preds(run):
        t = SHELF.get(inv.get(p["box"], ""))
        if t:
            by[p["run"]].append((p["pred"] == t, p["p"]))
    for r, v in sorted(by.items()):
        sure = [o for o, pr in v if pr >= 0.7]
        out[r] = {"crops": len(v), "acc": 100 * sum(o for o, _ in v) / len(v), "wrong_at_0.7": len(sure) - sum(sure)}
    return out


# the photo test crops both kinds of run saw, with the owner's label
proxy = {p["file"]: p["label"] for p in preds("v2_photo_b0e30_s0")}
common = set(proxy) & {p["file"] for p in preds("owner_photo_b0_s0")}
files = {f: owner_label(f, proxy[f]) for f in common}
files = {f: l for f, l in files.items() if l}
res = {"photo_test_crops_scored": len(files), "photo": {}, "shelf": {}}
for kind, fmt in (("machine labels (2 Nov)", "v2_{}_b0e30_s{}"), ("owner labels (4 Nov)", "owner_{}_b0_s{}")):
    ph = [photo(fmt.format("photo", s), files) for s in range(3)]
    sh = [shelf(fmt.format("shelf", s)) for s in range(3)]
    res["photo"][kind] = {"acc_median": statistics.median(x["acc"] for x in ph), "acc_runs": [x["acc"] for x in ph],
                          "wrong_at_0.7": [x["wrong_at_0.7"] for x in ph]}
    res["shelf"][kind] = {r: {"acc_runs": [x[r]["acc"] for x in sh], "wrong_at_0.7": [x[r]["wrong_at_0.7"] for x in sh],
                              "crops": sh[0][r]["crops"]} for r in sh[0]}
lines = ["# Product-look model: machine labels vs owner-reviewed labels (4 Nov 2026)", "",
         f"Photo split: {len(files)} test crops (Leo dup photos) scored against the owner's label; 3 seeds each.", "",
         "| Labels used for training | Photo test accuracy (3 seeds) | Wrong at p >= 0.7 |", "|---|---|---|"]
for k, v in res["photo"].items():
    lines.append(f"| {k} | {' / '.join(f'{a:.1f}%' for a in v['acc_runs'])} | {v['wrong_at_0.7']} |")
lines += ["", "Shelf split (sealed, never trained on; truth = inventory product):", "",
          "| Labels | Recording | Crops | Accuracy (3 seeds) | Wrong at p >= 0.7 |", "|---|---|---|---|---|"]
for k, v in res["shelf"].items():
    for r, x in v.items():
        lines.append(f"| {k} | {r} | {x['crops']} | {' / '.join(f'{a:.1f}%' for a in x['acc_runs'])} | {x['wrong_at_0.7']} |")
(HERE / "owner_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
(HERE / "owner_summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print("\n".join(lines))

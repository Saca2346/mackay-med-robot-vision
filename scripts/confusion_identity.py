"""
Confusion matrices of the box SELECTION (mentor's request 15-10: false positive / false negative), from logged results.

  1. identity evaluation, out-of-fold (14 Oct, 116 detected labelled boxes, 12 identities, one photo per decision):
     every box asked for its own identity (target) and for every other identity (not the target)
  2. live stress test on the shelf (D455, 20 boxes): each request vs the box that was picked, from the stress CSV
     and its verify log

  python scripts/confusion_identity.py --out results/meeting_YYYYMMDD --stress results/uji_realsense/hasil_X.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# safety_evaluation.log of 14 Oct (server audit/codex_20261014), catalogue = verified entries only
EVAL_20261014 = {
    "rows": ["target (right box asked)", "not the target (other identity asked)"],
    "cols": ["PICK", "candidate\n(1 source)", "needs\nconfirmation", "not target\n(rejected)"],
    "m": [[13, 42, 61, 0], [0, 41, 660, 575]],
    "note": "116 detected boxes x 12 identities; same product, other size: 87 requests, 0 PICK",
}


def plot(m, rows, cols, title, path, note="", bad_cell=(1, 0)):
    m = np.asarray(m, float)
    share = m / m.sum(axis=1, keepdims=True).clip(1)
    fig, ax = plt.subplots(figsize=(1.9 * len(cols) + 2.6, 1.15 * len(rows) + 2.0), dpi=160)
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(rows)):
        for j in range(len(cols)):
            bad = (i, j) == bad_cell                  # wrong box picked = the only unsafe cell
            ax.text(j, i, f"{int(m[i, j])}\n{share[i, j]:.0%}", ha="center", va="center", fontsize=12,
                    color="#B00020" if bad else ("white" if share[i, j] > 0.55 else "#14213D"),
                    fontweight="bold" if bad else "normal")
    ax.set_xticks(range(len(cols)), cols, fontsize=10)
    ax.set_yticks(range(len(rows)), rows, fontsize=10)
    ax.set_xlabel("decision of the system", fontsize=10)
    ax.set_title(title, fontsize=11)
    if note:
        fig.text(0.01, 0.01, note, fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def stress_matrix(csv_path):
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    tp = sum(r["hasil"] in ("PICK", "TAKEN", "PICK NOT TAKEN") for r in rows)
    fn = len(rows) - tp
    return rows, [[tp, fn], [0, 0]]


BRAND_WORDS = {"boston", "ivascular", "terumo", "apt", "orbusneich", "biosensors", "b"}


def picks_in_log(log, t_from, t_to, extra_text=None):
    """PICK rows of a verify log in [t_from, t_to] -> [(request, text read, consistent?)]. Consistent = the text read
    on the picked box holds the requested diameter and length (a wrong-box PICK would show other numbers)."""
    import re
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from src.box_ref import parse_refs
    out = []
    for r in csv.DictReader(open(log, encoding="utf-8")):
        if r["state"] != "PICK" or not (t_from <= r["time"] <= t_to):
            continue
        m = re.search(r"(\d+(?:\.\d+)?) x (\d+(?:\.\d+)?)", r["request"])
        # the verify log keeps 60 characters of the text read; the stress CSV keeps 120 of the same reading
        text = r["text"] + " | " + (extra_text or {}).get(r["request"].replace(" mm", ""), "")
        nums = {float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)}
        want = (float(m.group(1)), float(m.group(2))) if m else None
        refs = {(h.diameter, h.length) for h in parse_refs(text)}
        said = {(float(d), float(ln)) for d, ln in re.findall(r"(\d+(?:\.\d+)?) x (\d+(?:\.\d+)?)", text)}
        if want is None or (want[0] in nums and want[1] in nums) or want in refs:
            verdict = "right"
        elif (refs | said) - {want}:
            verdict = "wrong"                 # another size of the product was read on the picked box
        else:
            verdict = "not checkable"         # log text too short to show the second number
        out.append((r["request"], r["text"][:60], verdict))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stress", nargs="*", default=[], help="stress CSV(s) (scripts/test_webcam_box.py --log-stress)")
    ap.add_argument("--verify-log", nargs="*", default=[],
                    help='per stress CSV: "verify_log.csv|from|to" (ISO times) to check every PICK was the asked box')
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    e = EVAL_20261014
    plot(e["m"], e["rows"], e["cols"], "Box selection, out-of-fold photos (14 Oct): 0 wrong PICK of 1,276",
         out / "confusion_identity_oof.png", e["note"])
    summary = {"identity_oof_20261014": e}
    m = np.asarray(e["m"])
    tp, fn, fp, tn = int(m[0, 0]), int(m[0, 1:].sum()), int(m[1, 0]), int(m[1, 1:].sum())
    summary["identity_oof_20261014"]["binary_pick"] = {"TP": tp, "FN": fn, "FP": fp, "TN": tn,
                                                       "precision": round(tp / max(tp + fp, 1), 3),
                                                       "recall": round(tp / max(tp + fn, 1), 3)}
    for k, p in enumerate(a.stress):
        rows, _ = stress_matrix(p)
        picked = [r for r in rows if r["hasil"] in ("PICK", "TAKEN", "PICK NOT TAKEN")]
        # the stress test asks each box once; a PICK on another box would be a wrong pick: checked on the text that
        # was read on the picked box (verify log of the same run)
        extra = {}
        for r in rows:                                  # request text without the brand -> text read (stress CSV)
            words = r["permintaan"].split()
            key = " ".join(words[2:] if words[:2] == ["boston", "scientific"] or words[:2] == ["apt", "medical"]
                           else words[1:] if words[0] in BRAND_WORDS else words)
            extra[key] = r.get("teks_terbaca", "")
        checked = picks_in_log(*a.verify_log[k].split("|"), extra) if k < len(a.verify_log) else []
        wrong = [c for c in checked if c[2] == "wrong"]
        unsure = [c for c in checked if c[2] == "not checkable"]
        name = pathlib.Path(p).stem if pathlib.Path(p).stem != "hasil" else pathlib.Path(p).parent.name
        summary[name] = {"requests": len(rows), "PICK": len(picked), "no_pick": len(rows) - len(picked),
                         "picks_checked_in_verify_log": len(checked), "wrong_box": wrong,
                         "not_checkable_from_log": unsure,
                         "median_s": float(np.median([float(r["detik"]) for r in picked])) if picked else None}
        right = len(picked) - len(wrong) - len(unsure)
        plot([[right, len(unsure), len(wrong), len(rows) - len(picked)], [0, 0, 0, 0]],
             ["requested box", "any other box"],
             ["PICK\n(right box)", "PICK\n(log too short)", "PICK\n(wrong box)", "no PICK\n(timeout)"],
             f"Shelf replay: {len(picked)}/{len(rows)} PICK, {len(wrong)} wrong box", out / f"confusion_{name}.png",
             "D455 at shelf distance; no PICK = 2nd source not readable in time; 'log too short' = the 60-character log "
             "text does not show the 2nd number", bad_cell=(0, 2))
    (out / "confusion_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: (v.get("binary_pick") if isinstance(v, dict) and "binary_pick" in v else v)
                      for k, v in summary.items()}, indent=1))


if __name__ == "__main__":
    main()

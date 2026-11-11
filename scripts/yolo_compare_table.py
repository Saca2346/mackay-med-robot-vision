"""
YOLO comparison table for the mentor (meeting 15-10: YOLOv8 / YOLO11 / YOLO12 / YOLO26 - accuracy, FPS, parameters,
far detection). Inputs, all produced by the comparison experiment (server ~/Downloads/exp_yolo_compare_20261016):

  results.json         per model and fold: clean / degraded mAP, per class, confusion matrix, params, GFLOPs, speed
  far_eval.json        far detection MATCHED per box (far_eval.py): every detection of the D455 shelf recording (20 boxes,
                       never trained on) matched to 20 reference spine polygons (IoU >= 0.5) -> found / missed / false
                       boxes per frame. Keys "<model>/fold<k>" are the CV models; other keys are final / live models.
  far_detection.json   older box COUNT per frame, used only when far_eval.json is missing (a missed box plus a false
                       box still counts as 20: the 17-Oct check found exactly that for YOLO11 at 960)
  laptop_fps.json      laptop speed of the OpenVINO export at the demo size 544 x 960 (scripts/yolo_laptop_fps.py)

  python scripts/yolo_compare_table.py --dir results/meeting_next/yolo_compare
-> yolo_compare.md, yolo_compare.csv, yolo_compare.png, confusion_<model>.png (detection classes, summed over folds)
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ORDER = ["yolov8s-obb", "yolo11s-obb", "yolo12s-obb", "yolo26s-obb", "yolo11s-obb-coco"]
NAMES = {"yolov8s-obb": "YOLOv8s-OBB", "yolo11s-obb": "YOLO11s-OBB", "yolo12s-obb": "YOLO12s-OBB",
         "yolo26s-obb": "YOLO26s-OBB", "yolo11s-obb-coco": "YOLO11s-OBB, COCO start (control)"}
SIZES = ("544x960", "960")                    # far detection at the laptop's input (fixed 544 x 960), else 960


def ms(vals):
    return statistics.mean(vals), (statistics.pstdev(vals) if len(vals) > 1 else 0.0)


def far_matched(rows) -> dict:
    """[(found, missed, false), ...] -> rates and the share of frames per outcome."""
    a = np.array(rows)
    miss, false = a[:, 1] > 0, a[:, 2] > 0
    return {"recall": round(float(a[:, 0].sum() / (a[:, 0] + a[:, 1]).sum()), 4),
            "missed_per_frame": round(float(a[:, 1].mean()), 2), "false_per_frame": round(float(a[:, 2].mean()), 2),
            "perfect": round(float((~miss & ~false).mean() * 100), 1),
            "missed_only": round(float((miss & ~false).mean() * 100), 1),
            "false_only": round(float((~miss & false).mean() * 100), 1),
            "both": round(float((miss & false).mean() * 100), 1), "n": int(len(a))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    d = pathlib.Path(a.dir)
    res = json.load(open(d / "results.json"))
    far, far_kind, finals, size = {}, "", {}, "960"
    if (d / "far_eval.json").exists():
        raw = json.load(open(d / "far_eval.json"))["raw"]
        by_model: dict = {}
        size = next((z for z in SIZES if any(z in by for by in raw.values())), SIZES[-1])
        for key, by in raw.items():
            if "/fold" in key:
                by_model.setdefault(key.split("/")[0], []).extend(by.get(size, []))
            elif by.get(size):
                finals[key] = far_matched(by[size])
        far = {m: far_matched(r) for m, r in by_model.items() if r}
        far_kind = "matched"
    elif (d / "far_detection.json").exists():
        far_json = json.load(open(d / "far_detection.json"))
        for m, by in far_json.get("raw", {}).items():
            c = np.array(by.get("960", []))
            if c.size:
                far[m] = {"perfect": round(float((c == 20).mean() * 100), 1)}
        far_kind = "count"
    fps = json.load(open(d / "laptop_fps.json")) if (d / "laptop_fps.json").exists() else {}
    rows = []
    for m in ORDER:
        runs = [res[k] for k in sorted(res) if k.split("/")[0] == m]
        if not runs:
            continue
        clean = ms([r["clean"]["mAP50_95"] for r in runs])
        deg = ms([r["degraded"]["mAP50_95"] for r in runs])
        per = {c: statistics.mean(r["clean"]["per_class"].get(c, {}).get("mAP50_95", 0) for r in runs)
               for c in ("box", "size_label", "expiry_label")}
        rec = statistics.mean(r["clean"]["per_class"].get("box", {}).get("R", 0) for r in runs)
        cm = np.sum([np.array(r["confusion_matrix"]) for r in runs], axis=0)
        f = far.get(m, {})
        lf = fps.get(m, {})
        rows.append({"model": NAMES[m], "folds": len(runs), "mAP50_95_clean": f"{clean[0]:.3f} ± {clean[1]:.3f}",
                     "mAP50_95_webcam_like": f"{deg[0]:.3f} ± {deg[1]:.3f}", "box": f"{per['box']:.3f}",
                     "size_label": f"{per['size_label']:.3f}", "expiry_label": f"{per['expiry_label']:.3f}",
                     "box_recall": f"{rec:.3f}", "params_M": runs[0]["params_M"], "GFLOPs_640": runs[0]["gflops_640"],
                     "val_false_boxes": int(cm[:-1, -1].sum()), "val_missed": int(cm[-1, :-1].sum()),
                     "laptop_ms_960": lf.get("ms", ""), "laptop_fps_960": lf.get("fps", ""),
                     "far_recall": f.get("recall", ""), "far_false_per_frame": f.get("false_per_frame", ""),
                     "far_perfect_pct": f.get("perfect", ""), "far_missed_only_pct": f.get("missed_only", ""),
                     "far_false_only_pct": f.get("false_only", ""), "far_both_pct": f.get("both", ""),
                     "note": runs[0]["note"] + (f"; batch {runs[0].get('batch')}" if runs[0].get("batch", 8) != 8 else "")})
    keys = list(rows[0])
    with open(d / "yolo_compare.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    far_size = size if far_kind == "matched" else "960"
    far_head = (f"D455 shelf ({far_size}): recall / false per frame / perfect frames %" if far_kind == "matched"
                else "D455 shelf (960): frames with exactly 20 boxes %")
    head = ["Model", "mAP50-95 (clean)", "mAP50-95 (webcam-like)", "box", "size label", "expiry label", "box recall",
            "val false / missed (3 folds)", "Params (M)", "GFLOPs", "Laptop ms (FPS)", far_head]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        farcell = (f"{r['far_recall']} / {r['far_false_per_frame']} / {r['far_perfect_pct']}"
                   if far_kind == "matched" and r["far_recall"] != "" else str(r["far_perfect_pct"]))
        lines.append("| " + " | ".join(str(x) for x in (
            r["model"], r["mAP50_95_clean"], r["mAP50_95_webcam_like"], r["box"], r["size_label"], r["expiry_label"],
            r["box_recall"], f"{r['val_false_boxes']} / {r['val_missed']}", r["params_M"], r["GFLOPs_640"],
            f"{r['laptop_ms_960']} ({r['laptop_fps_960']})" if r["laptop_ms_960"] != "" else "", farcell)) + " |")
    notes = [f"* {r['model']}: {r['note']}" for r in rows]
    fin = []
    if finals:
        fin = ["", f"## Final / live models on the same shelf frames ({far_size})", "",
               "| Model | recall | missed / frame | false / frame | perfect frames % |", "|---|---|---|---|---|"]
        fin += [f"| {k} | {v['recall']} | {v['missed_per_frame']} | {v['false_per_frame']} | {v['perfect']} |"
                for k, v in finals.items()]
    (d / "yolo_compare.md").write_text(
        "# YOLO comparison (same 3 folds, same recipe)\n\n" + "\n".join(lines) + "\n\n" + "\n".join(notes) + "\n"
        + "\n".join(fin) + ("\n" if fin else ""), encoding="utf-8")
    # chart: accuracy, laptop speed, far detection (the control row only in the table)
    cr = [r for r in rows if "control" not in r["model"]]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4), dpi=150)
    names = [r["model"] for r in cr]
    vals = [float(r["mAP50_95_clean"].split(" ")[0]) for r in cr]
    errs = [float(r["mAP50_95_clean"].split(" ")[-1]) for r in cr]
    ax[0].bar(names, vals, yerr=errs, color="#2a6f97", capsize=4)
    ax[0].set_ylim(0.5, 0.85)
    ax[0].set_title("mAP50-95, 3-fold CV (mean ± sd)")
    for i, v in enumerate(vals):
        ax[0].text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=9)
    sp = [float(r["laptop_fps_960"]) if r["laptop_fps_960"] != "" else 0 for r in cr]
    ax[1].bar(names, sp, color="#c06a00")
    ax[1].set_title("Laptop detector speed, OpenVINO 544x960 (FPS)")
    for i, v in enumerate(sp):
        ax[1].text(i, v + 0.2, f"{v:.1f}" if v else "n/a", ha="center", fontsize=9)
    if far_kind == "matched":
        parts = [("far_perfect_pct", "#2e7d32", "all 20 found, no false box"),
                 ("far_missed_only_pct", "#c06a00", "box missed"),
                 ("far_false_only_pct", "#9e9e9e", "false box"),
                 ("far_both_pct", "#6d4c41", "missed + false")]
        bottom = np.zeros(len(cr))
        for k, col, lab in parts:
            v = np.array([float(r[k] or 0) for r in cr])
            ax[2].bar(names, v, bottom=bottom, color=col, label=lab)
            bottom += v
        ax[2].set_title(f"D455 shelf, 20 boxes, input {far_size}: frames (%)")
        ax[2].legend(fontsize=7, loc="upper center", ncol=2, frameon=False)
        ax[2].set_ylim(0, 125)
        ax[2].set_yticks(range(0, 101, 20))
        for i, r in enumerate(cr):
            v = float(r["far_perfect_pct"] or 0)
            ax[2].text(i, max(v - 8, 2), f"{v:.0f}%", ha="center", fontsize=9, color="white")
    for x in ax:
        x.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(d / "yolo_compare.png")
    plt.close(fig)
    # detection confusion matrices (summed over folds)
    for m in ORDER:
        runs = [res[k] for k in sorted(res) if k.split("/")[0] == m]
        if not runs:
            continue
        cm = np.sum([np.array(r["confusion_matrix"]) for r in runs], axis=0)
        names_cm = runs[0]["cm_names"]
        col = cm / cm.sum(axis=0, keepdims=True).clip(1)          # share of each true class (column = ground truth)
        fig, ax = plt.subplots(figsize=(5.6, 4.6), dpi=150)
        ax.imshow(col, cmap="Blues", vmin=0, vmax=1)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center", fontsize=9,
                        color="white" if col[i, j] > 0.55 else "#14213D")
        ax.set_xticks(range(len(names_cm)), names_cm, fontsize=8, rotation=20)
        ax.set_yticks(range(len(names_cm)), names_cm, fontsize=8)
        ax.set_xlabel("true class")
        ax.set_ylabel("predicted class")
        ax.set_title(f"{NAMES[m]}: detection confusion matrix (3 folds)", fontsize=9)
        fig.tight_layout()
        fig.savefig(d / f"confusion_{m}.png")
        plt.close(fig)
    print("\n".join(lines + fin))


if __name__ == "__main__":
    main()

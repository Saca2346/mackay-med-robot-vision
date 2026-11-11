"""Export and inspect a detector candidate without changing the live configuration.

The two task photos are diagnostic checks, not a held-out accuracy benchmark.
Reference outlines were made by Claude and still require owner review. Timings
printed by the underlying pipeline are not collected as benchmark results.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TASKS = (
    ("task1", "braun_sequent", "Braun, DCB, 2.75mm, 30mm", "sequent please neo|2.75|30|"),
    ("task2", "chroma", "Chroma, 2.75, 33", "chroma|2.75|33|"),
)


def sha256(path):
    with pathlib.Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fresh_output(path, root=ROOT):
    """Keep all candidate artifacts in a new, ignored results subdirectory."""
    path = pathlib.Path(path).resolve()
    results = (pathlib.Path(root) / "results").resolve()
    if not path.is_relative_to(results) or path == results:
        raise ValueError("--out must be a new subdirectory of this repository's results/")
    if any(part.lower().startswith("bukti_") for part in path.relative_to(results).parts):
        raise ValueError("Historical results/bukti_* evidence cannot hold candidate outputs")
    if path.exists():
        raise ValueError("--out already exists; choose a fresh directory to preserve previous runs")
    return path


def match_detections(left, right, iou_min=0.5):
    """Match same-class polygons one to one; counts alone can conceal a wrong box."""
    from src.box_geometry import poly_iou

    pairs = sorted(
        ((poly_iou(a.poly, b.poly), i, j) for i, a in enumerate(left)
         for j, b in enumerate(right) if a.cls == b.cls), reverse=True
    )
    used_left, used_right, matches = set(), set(), []
    for iou, i, j in pairs:
        if iou < iou_min or i in used_left or j in used_right:
            continue
        used_left.add(i)
        used_right.add(j)
        matches.append({"class": left[i].cls, "left": i, "right": j, "iou": round(iou, 6)})
    return {"matches": matches, "unmatched_left": sorted(set(range(len(left))) - used_left),
            "unmatched_right": sorted(set(range(len(right))) - used_right)}


def photo_log_summary(folder, identity_prefix):
    folder = pathlib.Path(folder)
    verify = folder / "verify_log.csv"
    with verify.open(encoding="utf-8-sig", newline="") if verify.exists() else _empty_csv() as stream:
        picks = [row for row in csv.DictReader(stream) if row.get("state") == "PICK"]
    readings = []
    for path in folder.glob("bacaan_*.csv"):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            readings.extend(row for row in csv.DictReader(stream)
                            if row.get("identity", "").startswith(identity_prefix))
    return {"pick_rows": picks, "requested_identity_readings": readings,
            "result_images": [str(p) for p in (folder / "screens").glob("*_result.png")]}


def _empty_csv():
    import io
    return io.StringIO("")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--config", type=pathlib.Path, default=ROOT / "config/box_pipeline_config_realsense.yaml")
    parser.add_argument("--task-root", type=pathlib.Path, default=ROOT / "results/task_20261015")
    parser.add_argument("--openvino-device", default="intel:cpu")
    parser.add_argument("--skip-pipeline", action="store_true", help="Only export and compare detector polygons")
    args = parser.parse_args()
    try:
        out = fresh_output(args.out)
    except ValueError as error:
        parser.error(str(error))
    weights = args.weights.resolve(strict=True)
    if weights.suffix.lower() != ".pt":
        parser.error("--weights must be a local .pt file")

    import cv2
    import numpy as np
    import yaml
    from ultralytics import YOLO
    from src.box_detector import BoxDetector
    from src.box_geometry import draw_poly, poly_iou

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    tasks = []
    for task, reference, request, identity in TASKS:
        reference_path = args.task_root / "training_data/eval_gt" / f"{reference}.json"
        gt = json.loads(reference_path.read_text(encoding="utf-8"))
        photo = args.task_root / "input" / pathlib.Path(gt["photo"]).name
        frame = cv2.imread(str(photo))
        if frame is None:
            raise ValueError(f"Cannot read photo: {photo}")
        tasks.append((task, request, identity, photo.resolve(), frame, gt, reference_path))

    out.mkdir(parents=True)
    model_dir = out / "model"
    model_dir.mkdir()
    candidate = model_dir / "candidate.pt"
    shutil.copy2(weights, candidate)
    report = {"source_sha256": sha256(weights), "copied_sha256": sha256(candidate),
              "versions": {name: importlib.metadata.version(name)
                           for name in ("ultralytics", "torch", "openvino")},
              "live_configuration_changed": False, "timing_benchmark": False,
              "reference_review": "Claude outlines; owner review pending",
              "config_sha256": sha256(args.config), "detections": [], "photo_pipeline": []}
    if report["source_sha256"] != report["copied_sha256"]:
        raise RuntimeError("Candidate copy hash mismatch")
    summary = out / "summary.json"

    def save():
        summary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    save()
    exported = pathlib.Path(YOLO(str(candidate)).export(
        format="openvino", imgsz=[544, 960], half=False, dynamic=False, device="cpu"
    )).resolve()
    if not exported.is_relative_to(out):
        raise RuntimeError("Exporter returned a path outside candidate scratch")
    import openvino as ov
    xml = next(exported.glob("*.xml"))
    ov_model = ov.Core().read_model(str(xml))
    report["openvino"] = {"path": str(exported), "input_shape": list(ov_model.input(0).shape),
                          "artifact_sha256": {p.name: sha256(p) for p in exported.iterdir() if p.is_file()}}
    if report["openvino"]["input_shape"] != [1, 3, 544, 960]:
        raise RuntimeError("Unexpected export input shape")
    save()
    profiles = (("pt_photo_960", candidate, 960, "cpu", False),
                ("pt_fixed_544x960", candidate, [544, 960], "cpu", True),
                ("openvino_544x960", exported, [544, 960], args.openvino_device, False))
    by_task = {}
    for name, model, size, device, fixed in profiles:
        print(f"Checking detector profile {name}", flush=True)
        detector = BoxDetector(str(model), "obb", size, device, config["model"]["conf"],
                               config["model"]["iou"], config["classes"], fixed_input=fixed)
        profile_dir = out / name
        profile_dir.mkdir()
        for task, _, _, photo, frame, gt, reference_path in tasks:
            dets, _ = detector.detect(frame)
            by_task[(task, name)] = dets
            preview = frame.copy()
            for det in dets:
                draw_poly(preview, det.poly, (255, 160, 0) if det.cls == "box" else (180, 0, 180))
            draw_poly(preview, np.asarray(gt["target"]["box"]), (0, 255, 255), 3)
            cv2.imwrite(str(profile_dir / f"{task}_detections.jpg"), preview)
            report["detections"].append({
                "profile": name, "task": task, "photo_sha256": sha256(photo),
                "reference_sha256": sha256(reference_path), "frame_shape": list(frame.shape),
                "target_iou": {cls: round(max((poly_iou(d.poly, gt["target"][cls]) for d in dets
                                              if d.cls == cls), default=0.0), 6)
                               for cls in ("box", "size_label")},
                "counts": {cls: sum(d.cls == cls for d in dets) for cls in ("box", "size_label")},
                "polygons": [{"class": d.cls, "confidence": d.conf, "poly": d.poly.tolist()} for d in dets],
            })
            save()
        del detector
    report["fixed_input_export_comparison"] = {
        task: match_detections(by_task[(task, "pt_fixed_544x960")], by_task[(task, "openvino_544x960")])
        for task, *_ in tasks
    }
    save()
    if not args.skip_pipeline:
        for name, model, size, device, _ in (profiles[0], profiles[2]):
            for task, request, identity, photo, *_ in tasks:
                run_dir = out / name / task
                run_dir.mkdir()
                command = [sys.executable, str(ROOT / "scripts/demo_foto_target.py"), str(photo),
                           "--request", request, "--out", str(run_dir), "--config", str(args.config.resolve()),
                           "--weights", str(model), "--imgsz", ",".join(map(str, size)) if isinstance(size, list) else str(size),
                           "--device", device]
                (run_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
                print(f"Checking photo pipeline {name}/{task}", flush=True)
                with (run_dir / "run.log").open("w", encoding="utf-8") as log:
                    completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
                report["photo_pipeline"].append({"profile": name, "task": task,
                                                "returncode": completed.returncode,
                                                **photo_log_summary(run_dir, identity)})
                save()
                if completed.returncode:
                    raise RuntimeError(f"Photo pipeline failed; inspect {run_dir / 'run.log'}")
    print(f"Candidate artifacts and diagnostic summary: {summary}")


if __name__ == "__main__":
    main()

"""
Request-level confusion matrix of the target verification on a recorded shelf (mentor's request 15-10: false positive /
false negative), one matrix per camera, same code and profile as the live view.

  rows    : the requested box IS on the shelf (positive) / is NOT on the shelf (negative)
  columns : what the system decided for that request in a fresh process (45 frames of the recording)
              VERIFIED         green, two independent sources
              1 source only    blue: a candidate (text or REF alone) or TEXT READ (e.g. Sled Bag name + date); the robot waits
              nothing matched  no evidence for the request

Positives = the 11 inventory requests of the 11-box shelf (scripts/validate_box_targets.py). Negatives = requests that are
not on that shelf: same product in another size (the dangerous ones), Sled Bag with a wrong date, products of other shelves.

  python scripts/confusion_shelf_requests.py --out results/weekly_20261023/confusion

The recordings overlap the training data and a replay is not a live test; ground truth = the 11 boxes of the request menu
(owner review of the box identities is pending).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
CAMERAS = {
    "webcam": dict(source="results/calibrated_eleven_20261021/capture/camera_raw.mp4",
                   profile="config/box_pipeline_config_webcam_eleven.yaml"),
    "realsense": dict(source="results/target_realsense_20261022_085253_734152/take_085402_552845/camera_raw.mp4",
                      profile="config/box_pipeline_config_realsense_eleven.yaml"),
}
NEGATIVES = [
    ("size", "angiolite 4 x 29"), ("size", "angiolite 2.5 x 19"), ("size", "angiolite 4 x 24"), ("size", "angiolite 3 x 19"),
    ("size", "essential pro 3 x 30"), ("size", "essential pro 3.5 x 40"),
    ("size", "xperience pro 2.5 x 20"), ("size", "xperience pro 3 x 15"),
    ("size", "accuforce 2.5 x 15"), ("size", "accuforce 2.75 x 12"), ("size", "accuforce 3 x 15"), ("size", "accuforce 3.5 x 12"),
    ("date", "permanent sled bag 2029-03-11"), ("date", "permanent sled bag 2029-03-25"),
    ("date", "permanent sled bag 2029-04-01"), ("date", "permanent sled bag 2030-01-01"),
    ("other shelf", "ryurei 2 x 10"), ("other shelf", "ultimaster nagomi 2.75 x 33"),
    ("other shelf", "nc emerge 5.5 x 12 2028-01-03"), ("other shelf", "scoreflex 3 x 15"),
    ("other shelf", "conqueror nc pro 3.5 x 15"), ("other shelf", "sapphire 3 x 15"),
]


def classify(take: pathlib.Path) -> dict:
    """Decision of one trial from its target_view.jsonl: VERIFIED > 1 source only > nothing matched."""
    verified = text_ready = one_source = None
    for line in (take / "target_view.jsonl").read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        state = d.get("state")
        if state == "VERIFIED" and verified is None:
            verified = dict(seconds=d.get("seconds"), track=d.get("track"), sources=d.get("sources"),
                            outline=d.get("outline_px"))
        elif state == "TEXT_READY" and text_ready is None:
            text_ready = d.get("source_seconds")
        elif state == "CHECKING" and d.get("sources") and one_source is None:
            one_source = d.get("sources")
    decision = "VERIFIED" if verified else "ONE_SOURCE" if (text_ready is not None or one_source) else "NOTHING"
    return dict(decision=decision, verified=verified, text_ready=text_ready, one_source=one_source)


def run_negative(camera: str, request: str, folder: pathlib.Path) -> int:
    cfg = CAMERAS[camera]
    cmd = [sys.executable, str(ROOT / "scripts/run_target.py"), "--camera", camera, "--config", str(ROOT / cfg["profile"]),
           "--source", str(ROOT / cfg["source"]), "--request", request, "--max-frames", "45", "--no-window",
           "--output", str(folder)]
    return subprocess.call(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600)


def power() -> dict:
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "Add-Type -A System.Windows.Forms; [System.Windows.Forms.SystemInformation]::PowerStatus.PowerLineStatus"],
                         capture_output=True, text=True).stdout.strip()
    return {"power_line": out}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--cameras", default="webcam,realsense")
    p.add_argument("--skip-positives", action="store_true")
    p.add_argument("--skip-negatives", action="store_true")
    p.add_argument("--profile", action="append", default=[], metavar="CAMERA=PATH",
                   help="use another profile for a camera, e.g. webcam=config/x.yaml")
    args = p.parse_args()
    for item in args.profile:
        camera, path = item.split("=", 1)
        CAMERAS[camera]["profile"] = path
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for camera in args.cameras.split(","):
        cfg = CAMERAS[camera]
        rows = []
        before = power()
        pos_dir = out / camera / "positives"
        if not args.skip_positives:
            subprocess.call([sys.executable, str(ROOT / "scripts/validate_box_targets.py"), "--source", str(ROOT / cfg["source"]),
                             "--config", str(ROOT / cfg["profile"]), "--max-frames", "45", "--output", str(pos_dir)],
                            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3600)
        for case in sorted((pos_dir / "cases").glob("K*")):
            take = next(case.glob("take_*"))
            row = classify(take)
            meta = next(r for r in json.loads((pos_dir / "results.json").read_text(encoding="utf-8")) if r["box_id"] == case.name)
            rows.append(dict(kind="present", group=case.name, request=meta["request"], take=str(take), **row))
        for i, (group, request) in enumerate([] if args.skip_negatives else NEGATIVES, 1):
            folder = out / camera / "negatives" / f"N{i:02d}"
            code = run_negative(camera, request, folder)
            take = next(folder.glob("take_*"), None)
            row = classify(take) if take else dict(decision="ERROR", verified=None, text_ready=None, one_source=None)
            rows.append(dict(kind="absent", group=group, request=request, take=str(take) if take else None, exit_code=code, **row))
            print(camera, f"N{i:02d}", request, row["decision"], flush=True)
        results[camera] = dict(rows=rows, power_before=before, power_after=power())
        (out / "confusion_results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

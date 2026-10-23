"""Same verified target view and evidence files for webcam and RealSense."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.sesi_config import create_session_config
from src.box_calib import parse_region


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--camera", choices=("webcam", "realsense"))
    p.add_argument("--config", type=Path, help="optional camera profile for a separately evaluated model")
    p.add_argument("--request", default="angiolite 4 x 19")
    p.add_argument("--index", type=int, help="webcam device index; current laptop default is 1")
    p.add_argument("--workers", type=int, choices=(1, 2), default=2)
    p.add_argument("--target-seconds", type=float, default=2.0, help="measured latency target; never changes verification rules")
    p.add_argument("--output", type=Path)
    p.add_argument("--source", help="optional recorded video for a replay instead of live capture")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--roi", type=parse_region, help="optional target area x1,y1,x2,y2 as fractions of the original frame")
    p.add_argument("--no-window", action="store_true")
    p.add_argument("--no-barcode", action="store_true", help="skip barcode/QR decoding; OCR/REF and independent-source verification rules remain active")
    p.add_argument("--show-all", action="store_true", help="show all visible detections while reading the selected target")
    p.add_argument("--expected", type=int, help="expected shelf count for diagnostics; never caps detections")
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--prepare-only", action="store_true", help="create isolated config/inventory without opening a camera")
    p.add_argument("--calibrate-first", action="store_true", help="open live framing calibration before detection; cancel/failure stops the launch")
    return p


def prepare_session(args):
    if args.index is not None and args.camera != "webcam":
        raise ValueError("--index is only for a webcam; RealSense uses its SDK")
    if args.target_seconds <= 0 or args.max_frames < 0:
        raise ValueError("target seconds must be positive; max frames must not be negative")
    if args.expected is not None and args.expected < 1:
        raise ValueError("expected count must be positive")
    if args.calibrate_first and args.source:
        raise ValueError("--calibrate-first needs a live camera; omit --source")
    profile = "box_pipeline_config_focus.yaml" if args.camera == "webcam" else "box_pipeline_config_focus_realsense.yaml"
    source = (args.config or ROOT / "config" / profile).resolve()
    base = yaml.safe_load(source.read_text(encoding="utf-8"))
    is_realsense = base.get("camera", {}).get("backend") == "realsense"
    if is_realsense != (args.camera == "realsense"):
        raise ValueError("profile camera backend does not match --camera")
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    folder = (args.output or ROOT / "results" / f"target_{args.camera}_{stamp}").resolve()
    folder.mkdir(parents=True, exist_ok=False)
    config = create_session_config(source, folder)
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    cfg["visual"].update(workers=args.workers, target_seconds=args.target_seconds)
    cfg["visual"].setdefault("transparent_header", True)
    if args.no_barcode:
        cfg.setdefault("verify", {})["decode_barcodes"] = False
    if args.show_all:
        cfg["visual"]["show_all"] = True
    if args.expected is not None:
        cfg.setdefault("selection", {})["expected_count"] = args.expected
    if args.index is not None:
        cfg["camera"]["index"] = args.index
    config.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    inventory = folder / "inventory_copy.csv"
    shutil.copy2(ROOT / "data/rak/inventaris.csv", inventory)
    (folder / "session.json").write_text(json.dumps({"camera": args.camera, "profile": source.name, "profile_path": str(source), "camera_settings": cfg["camera"],
                              "model": cfg["model"], "ocr_workers": args.workers,
                              "latency_target_seconds": args.target_seconds, "source": args.source or "live camera",
                              "roi": args.roi,
                              "barcode_decoding": cfg.get("verify", {}).get("decode_barcodes", True),
                              "show_all": cfg["visual"].get("show_all", False),
                              "expected_count": cfg.get("selection", {}).get("expected_count", 0),
                              "transparent_header": cfg["visual"]["transparent_header"],
                              "calibration_first": args.calibrate_first,
                              "note": "Latency is measured after model startup; verification always needs independent evidence."}, indent=2))
    print(f"[session] {folder}", flush=True)
    print(f"[camera] {args.camera}: {cfg['camera']}", flush=True)
    print(f"[target] {args.target_seconds:g}s to VERIFIED; {args.workers} OCR workers; startup excluded", flush=True)
    if not cfg.get("verify", {}).get("decode_barcodes", True):
        print("[target] Barcode/QR decoding OFF; OCR/printed REF remain active. Text alone cannot verify identity.", flush=True)
    return folder, config, inventory


def run_calibration(args, folder, config):
    """Finish same-profile live framing before the detector opens a new camera stream."""
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    expected = cfg.get("selection", {}).get("expected_count") or 1
    output = folder / "calibration"
    name = folder.name
    card = output / (name + "_kalibrasi.json")
    command = [sys.executable, str(ROOT / "scripts/run_logged.py"), "--log", str(folder / "cmd_calibration.txt"),
               "--", sys.executable, "-u", str(ROOT / "scripts/kalibrasi_kamera.py"), "--sesi", name,
               "--config", str(config), "--out", str(output), "--min-boxes", str(expected)]
    if args.roi is not None:
        command += ["--roi", ",".join(str(v) for v in args.roi)]
    if args.no_window:
        command += ["--no-window"]
    print("[calibration] Adjust framing, then Enter to measure. ESC cancels; detection starts after completion.", flush=True)
    code = subprocess.call(command, cwd=ROOT)
    result = json.loads(card.read_text(encoding="utf-8")) if card.exists() else None
    if code == 0 and (result is None or result.get("verdict") not in ("PASS", "WARN")):
        code = 1
    session = folder / "session.json"
    metadata = json.loads(session.read_text(encoding="utf-8"))
    metadata["calibration"] = {"exit_code": code, "card": str(card) if card.exists() else None,
                               "verdict": result.get("verdict") if result else "CANCELLED_OR_FAILED",
                               "scope": "Framing/image checks, not identity, lens intrinsics or latency certification"}
    session.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if code:
        print("[calibration] Detection not started; calibration failed or was cancelled. Evidence retained.", flush=True)
    else:
        print(f"[calibration] {result['verdict']}: {card}; starting a fresh detection stream", flush=True)
    return code


def run_request(args, folder, config, inventory, request):
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%H%M%S_%f")
    take = folder / ("take_" + stamp)
    take.mkdir()
    config = create_session_config(config, take)
    settings = yaml.safe_load(config.read_text(encoding="utf-8"))
    expected = settings.get("selection", {}).get("expected_count") or 20
    command = [sys.executable, "-u", str(ROOT / "scripts/test_webcam_box.py"), "--verify-target", "--config", str(config),
               "--inventaris", str(inventory), "--request", request, "--expected", str(expected), "--record-vis", str(take / "view.mp4"),
               "--evidence-photos", str(take / "photos"), "--debug-readings", str(take / "readings.csv")]
    if args.source:
        command += ["--source", args.source]
    else:
        command += ["--record", str(take / "camera_raw.mp4")]
    if args.no_window:
        command += ["--no-window"]
    if args.max_frames:
        command += ["--max-frames", str(args.max_frames)]
    if args.roi is not None:
        command += ["--roi", ",".join(str(v) for v in args.roi)]
    print("[view] S: extra photo; Q: stop. Type another target in CMD to keep models loaded.", flush=True)
    code = subprocess.call([sys.executable, str(ROOT / "scripts/run_logged.py"), "--log", str(take / "cmd.txt"), "--", *command], cwd=ROOT)
    print(f"[session] {'Saved' if code == 0 else 'Failed/interrupted; inspect CMD transcript'}: {take}", flush=True)
    return code


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if not args.camera and args.interactive:
        choice = input("Camera: 1 external webcam, 2 RealSense, 0 exit: ").strip()
        if choice == "0":
            return 0
        args.camera = {"1": "webcam", "2": "realsense"}.get(choice)
    if not args.camera:
        p.error("choose --camera webcam or --camera realsense")
    try:
        folder, config, inventory = prepare_session(args)
    except (OSError, ValueError) as exc:
        p.error(str(exc))
    if args.prepare_only:
        return 0
    if args.calibrate_first:
        code = run_calibration(args, folder, config)
        if code:
            return code
    if not args.interactive:
        return run_request(args, folder, config, inventory, args.request)
    while True:
        print("\n1 Preflight | 2 Angiolite 2.5 x 29 | 3 Angiolite 4 x 19 | 4 Other request | 0 Exit", flush=True)
        choice = input("Choose: ").strip()
        if choice == "0":
            return 0
        if choice == "1":
            subprocess.call([sys.executable, str(ROOT / "scripts/run_logged.py"), "--log", str(folder / "cmd_preflight.txt"), "--",
                             sys.executable, "-u", str(ROOT / "scripts/demo_preflight.py"), "--config", str(config), "--inventaris", str(inventory)], cwd=ROOT)
            continue
        request = {"2": "angiolite 2.5 x 29", "3": "angiolite 4 x 19"}.get(choice)
        if choice == "4":
            request = input("Product and exact size: ").strip()
        if request:
            run_request(args, folder, config, inventory, request)


if __name__ == "__main__":
    raise SystemExit(main())

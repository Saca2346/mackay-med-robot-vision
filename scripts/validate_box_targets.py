"""Sequential live or recorded target trials with evidence and separate owner review.

No inventory writes or physical retrieval. An inventory row defines a request,
never supplies an observed field or a machine/owner verification decision.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def power_status():
    if os.name != "nt":
        return {"ac": None, "note": "Power status unavailable"}
    import ctypes
    class Power(ctypes.Structure):
        _fields_ = [("ac", ctypes.c_ubyte), ("flags", ctypes.c_ubyte),
                    ("percent", ctypes.c_ubyte), ("reserved", ctypes.c_ubyte),
                    ("life", ctypes.c_uint32), ("full", ctypes.c_uint32)]
    status = Power()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
        return {"ac": None, "note": "Power status could not be read"}
    return {"ac": int(status.ac), "battery_percent": int(status.percent)}


def inventory_plan(path):
    """Keep every row separate, including explicitly requested printed dates."""
    from src.box_verifier import parse_request
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    plan = []
    seen = set()
    for row in rows:
        if row.get("status") != "terverifikasi":
            continue
        box_id = row["box_id"]
        if box_id in seen or not box_id.isalnum():
            raise ValueError(f"Duplicate or invalid inventory id: {box_id!r}")
        seen.add(box_id)
        request = row["product"].strip()
        d, length = row.get("diameter", ""), row.get("length", "")
        if bool(d) != bool(length):
            raise ValueError(f"Incomplete reference size: {box_id}")
        if d:
            request += f" {float(d):g} x {float(length):g}"
        expiry = row.get("expiry", "").strip()
        if expiry:
            datetime.strptime(expiry, "%Y-%m-%d")
            request += " " + expiry
        parsed = parse_request(request)
        if parsed.product != row["product"] or (expiry and str(parsed.expiry) != expiry):
            raise ValueError(f"Request did not retain the reference identity/date: {box_id}")
        thumb = Path(path).parent / row.get("thumb", "")
        plan.append({"box_id": box_id, "request": request, "product": row["product"],
                     "reference_diameter": float(d) if d else None,
                     "reference_length": float(length) if length else None,
                     "required_printed_date": expiry or None,
                     "reference_photo": str(thumb.resolve()) if thumb.is_file() else None,
                     "owner_validation": "PENDING"})
    if not plan:
        raise ValueError("No reference boxes available")
    # Identical products/sizes must remain individually addressable.
    keys = [(p["product"], p["reference_diameter"], p["reference_length"], p["required_printed_date"])
            for p in plan]
    if len(set(keys)) != len(keys):
        raise ValueError("Indistinguishable inventory rows need unique printed dates before individual validation")
    for item in plan:
        siblings = [p for p in plan if (p["product"], p["reference_diameter"], p["reference_length"])
                    == (item["product"], item["reference_diameter"], item["reference_length"])]
        if len(siblings) > 1 and not item["required_printed_date"]:
            raise ValueError("Indistinguishable inventory rows need unique printed dates before individual validation")
    return sorted(plan, key=lambda r: r["box_id"])


def summarize(case, folder, exit_code, before, after, has_real_times):
    takes = sorted(Path(folder).glob("take_*"))
    if not takes:
        return {**case, "status": "PROCESS_FAILED", "exit_code": exit_code,
                "owner_validation": "PENDING", "take": None}
    take = takes[-1]
    log = take / "target_view.jsonl"
    events = [json.loads(s) for s in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []
    states = [r for r in events if r["state"] in ("VERIFIED", "CHECKING", "SEARCHING")]
    verified = [r for r in states if r["state"] == "VERIFIED"]
    displays = [r for r in events if r["state"] == "DISPLAY_READY"]
    exports = [r for r in events if r["state"] == "EVIDENCE_SAVED"]
    text_ready = [r for r in events if r["state"] == "TEXT_READY"]
    from src.box_verifier import parse_request
    requested = str(parse_request(case["request"]))
    changed_request = any(r.get("request", requested) != requested for r in events)
    restarted = any(r.get("request_generation", 0) != 0 for r in events)
    first_index = next((i for i, r in enumerate(states) if r["state"] == "VERIFIED"), None)
    losses = sum(r["state"] != "VERIFIED" for r in states[first_index+1:]) if first_index is not None else 0
    green = displays[0]["wall_seconds"] if displays else None
    eligible = (before.get("ac") == after.get("ac") == 1 and has_real_times
                and not changed_request and not restarted and not exit_code)
    status = ("PROCESS_FAILED" if exit_code else "REQUEST_CHANGED" if changed_request else
              "REQUEST_RESTARTED" if restarted else
              "TEXT_READ_NOT_VERIFIED" if not verified and text_ready else "NOT_VERIFIED" if not verified else
              "VERIFIED_THEN_LOST" if losses else "VERIFIED_TIMING_INELIGIBLE" if not eligible else
              "VERIFIED_WITHIN_2S" if green is not None and green <= 2 else
              "VERIFIED_OVER_2S")
    latest = None
    if exports:
        path = Path(exports[-1]["path"]) / "result.json"
        if path.exists():
            latest = json.loads(path.read_text(encoding="utf-8"))
    photos_log = take / "photos/index.jsonl"
    photos = [json.loads(s) for s in photos_log.read_text(encoding="utf-8").splitlines()] if photos_log.exists() else []
    first_photo = next((r for r in photos if r["event"] == "VERIFIED"), None)
    final_photo = photos[-1] if photos else None
    fields = latest.get("fields", {}) if latest else {}
    verified_export = next((r for r in exports if r["identity_verified"]), None)
    return {**case, "status": status, "exit_code": exit_code, "owner_validation": "PENDING",
            "take": str(take), "power_before": before, "power_after": after,
            "timing_eligible": eligible, "actual_capture_times": has_real_times,
            "request_changed": changed_request,
            "request_restarted": restarted,
            "text_frame_seconds": text_ready[0]["wall_seconds"] if text_ready else None,
            "machine_verified_once": bool(verified),
            "machine_verified_at_end": bool(states and states[-1]["state"] == "VERIFIED"),
            "verification_seconds": verified[0]["wall_seconds"] if verified else None,
            "green_frame_seconds": green, "later_verification_losses": losses,
            "first_verified_export_seconds": verified_export["wall_seconds"] if verified_export else None,
            "verified_sources": verified[0].get("sources", "") if verified else "",
            "last_observed_sources": latest.get("sources", "") if latest else "",
            "observed_fields": fields, "all_fields_read": bool(latest and latest["all_fields_read"]),
            "missing_fields": [k for k, v in fields.items() if v["status"] != "read"],
            "latest_result": str(Path(exports[-1]["path"]) / "result.json") if exports else None,
            "first_verified_result": str(Path(verified_export["path"]) / "result.json") if verified_export else None,
            "first_verified_photo": str(take / "photos" / first_photo["photos"]["view"]) if first_photo else None,
            "final_photo": str(take / "photos" / final_photo["photos"]["view"]) if final_photo else None}


def fingerprint(profile, inventory, source=None):
    import yaml
    cfg = yaml.safe_load(Path(profile).read_text(encoding="utf-8"))
    weights = Path(cfg["model"]["weights"])
    weights = weights if weights.is_absolute() else ROOT / weights
    model_files = sorted(p for p in weights.rglob("*") if p.is_file()) if weights.is_dir() else [weights]
    code = sorted((ROOT / "src").glob("*.py")) + [ROOT / p for p in (
        "scripts/test_webcam_box.py", "scripts/run_target.py", "scripts/run_logged.py",
        "scripts/sesi_config.py", "scripts/validate_box_targets.py", "scripts/box_target_review.py",
        "scripts/kalibrasi_kamera.py")]
    paths = [Path(profile), Path(inventory), *model_files, *code]
    if source is not None:
        paths.append(Path(source))
    catalog = Path(cfg.get("verify", {}).get("catalog_csv", "data/gtin_catalog.csv"))
    paths.append(catalog if catalog.is_absolute() else ROOT / catalog)
    timing = Path(str(source) + ".times.csv") if source is not None else None
    if timing is not None and timing.exists():
        paths.append(timing)
    return {str(p.resolve()): digest(p) for p in paths}


def check_inputs(expected, output, stage):
    """Stop on a shared-tree change; never certify a trial against the wrong source version."""
    changed = [path for path, sha in expected.items() if not Path(path).is_file() or digest(path) != sha]
    if changed:
        (Path(output) / "input_drift.json").write_text(json.dumps({"stage": stage, "changed": changed,
            "note": "Stopped; retain partial evidence, use a fresh folder for the changed inputs"}, indent=2))
        raise SystemExit("Input/code drift during batch; see input_drift.json. Partial evidence retained.")


def run_case(command, log_path, timeout):
    """A timeout stops only the owned trial process tree before another trial starts."""
    with Path(log_path).open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if proc.poll() is None:
                if os.name == "nt":
                    subprocess.run(["taskkill.exe", "/PID", str(proc.pid), "/T", "/F"],
                                   stdout=log, stderr=subprocess.STDOUT, check=False)
                else:
                    proc.kill()
                proc.wait(timeout=30)
            log.write("\n[batch] Owned trial timed out; partial evidence retained.\n")
            return 124


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--source", type=Path, help="existing full-shelf recording")
    mode.add_argument("--live", action="store_true", help="record a fresh camera trial for each target")
    p.add_argument("--camera", choices=("webcam", "realsense"), help="defaults to the profile camera backend")
    p.add_argument("--config", type=Path)
    p.add_argument("--calibrate-first", action="store_true", help="live framing calibration once before all trials")
    p.add_argument("--no-window", action="store_true", help="headless capture; calibration measures automatically")
    p.add_argument("--inventory", type=Path, default=ROOT / "data/rak/inventaris.csv")
    p.add_argument("--output", type=Path)
    p.add_argument("--only", default="", help="comma-separated inventory ids")
    p.add_argument("--max-frames", type=int, default=80)
    p.add_argument("--timeout", type=float, default=180)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.max_frames < 1 or args.timeout <= 0 or (args.source and not args.source.is_file()):
        raise SystemExit("Need an existing recording (or --live), positive frame count and timeout")
    if args.calibrate_first and not args.live:
        raise SystemExit("--calibrate-first requires --live")
    if args.live and args.resume:
        raise SystemExit("Live trials require a fresh session; use a new output folder")
    import yaml
    profile = (args.config or ROOT / "config" / (
        f"box_pipeline_config_{args.camera or 'webcam'}_eleven.yaml" if args.live else
        "box_pipeline_config_focus_stable.yaml")).resolve()
    cfg = yaml.safe_load(profile.read_text(encoding="utf-8"))
    camera = "realsense" if cfg.get("camera", {}).get("backend") == "realsense" else "webcam"
    if args.camera and args.camera != camera:
        raise SystemExit("Profile camera backend does not match --camera")
    source = args.source.resolve() if args.source else None
    inventory = args.inventory.resolve()
    if inventory != (ROOT / "data/rak/inventaris.csv").resolve():
        raise SystemExit("The current runner uses the master inventory as its read-only reference; use that path")
    plan = inventory_plan(inventory)
    profile_ids = cfg.get("visual", {}).get("request_ids")
    if profile_ids:
        if len(set(profile_ids)) != len(profile_ids) or set(profile_ids) - {r["box_id"] for r in plan}:
            raise SystemExit("Profile contains duplicate/unknown request ids")
        plan = [r for r in plan if r["box_id"] in profile_ids]
    if args.only:
        selected = set(args.only.split(","))
        if selected - {r["box_id"] for r in plan}:
            raise SystemExit("Unknown inventory ids")
        plan = [r for r in plan if r["box_id"] in selected]
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    out = (args.output or ROOT / "results" / ("all_boxes_review_" + stamp)).resolve()
    inputs = {"fingerprint": fingerprint(profile, inventory, source), "plan": plan,
              "max_frames": args.max_frames, "source": str(source) if source else None, "profile": str(profile),
              "camera": camera, "live": args.live, "calibration_first": args.calibrate_first,
              "no_window": args.no_window,
              "training_overlap": "Not established for this capture; model has seen earlier shelf imagery",
              "owner_validation": "PENDING", "mode": "live capture" if args.live else "recorded replay",
              "scope": "Fresh process/observations per target; initialization excluded; owner approval separate"}
    if out.exists():
        if not args.resume or not (out / "inputs.json").exists():
            raise SystemExit("Output exists; choose a new folder or --resume with matching inputs")
        if json.loads((out / "inputs.json").read_text()) != inputs:
            raise SystemExit("Resume refused: input/model/code/plan changed; create a new evidence folder")
    else:
        out.mkdir(parents=True)
        (out / "inputs.json").write_text(json.dumps(inputs, indent=2), encoding="utf-8")
        shutil.copy2(inventory, out / "inventory_reference.csv")
        for filename, expected in inputs["fingerprint"].items():
            original = Path(filename)
            if original.suffix == ".py" or original == profile:
                snapshot = out / "source_snapshot" / original.relative_to(ROOT)
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original, snapshot)
                if digest(snapshot) != expected:
                    raise SystemExit("Source changed while snapshotting; choose a fresh output folder")
    (out / "cases").mkdir(exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    rows_path = out / "results.json"
    rows = json.loads(rows_path.read_text()) if rows_path.exists() else []
    from scripts.box_target_review import write_review
    write_review(out, plan, rows)
    print(f"[batch] {len(plan)} boxes; output {out}; owner validation remains PENDING", flush=True)
    if args.prepare_only:
        return 0
    if args.calibrate_first and not rows:
        from scripts.run_target import parser as target_parser, prepare_session, run_calibration
        calibration_args = target_parser().parse_args(["--camera", camera, "--config", str(profile),
            "--calibrate-first", "--output", str(out / "calibration_session")]
            + (["--no-window"] if args.no_window else []))
        folder, config, _ = prepare_session(calibration_args)
        code = run_calibration(calibration_args, folder, config)
        if code:
            return code
    for i, case in enumerate(plan, 1):
        if any(r["box_id"] == case["box_id"] for r in rows):
            continue
        check_inputs(inputs["fingerprint"], out, "before " + case["box_id"])
        before = power_status()
        folder = out / "cases" / case["box_id"]
        command = [sys.executable, str(ROOT / "scripts/run_target.py"), "--camera", camera,
                   "--config", str(profile), "--request", case["request"],
                   "--max-frames", str(args.max_frames), "--output", str(folder)]
        if source is not None:
            command += ["--source", str(source), "--no-window"]
        elif args.no_window:
            command += ["--no-window"]
        print(f"[batch] {i}/{len(plan)} {case['box_id']} {case['request']} | AC={before.get('ac')}", flush=True)
        exit_code = run_case(command, out / "logs" / (case["box_id"] + ".txt"), args.timeout)
        after = power_status()
        check_inputs(inputs["fingerprint"], out, "after " + case["box_id"])
        actual_times = (Path(str(source) + ".times.csv").exists() if source else
                        any(folder.glob("take_*/camera_raw.mp4.times.csv")))
        row = summarize(case, folder, exit_code, before, after, actual_times)
        row["mode"] = inputs["mode"]
        takes = sorted(folder.glob("take_*"))
        if takes:
            shelf = takes[-1] / "shelf_detections.jsonl"
            records = [json.loads(s) for s in shelf.read_text().splitlines()] if shelf.exists() else []
            row["shelf_frames"] = len(records)
            row["shelf_count_histogram"] = dict(Counter(r["detected"] for r in records))
        rows.append(row)
        rows_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        write_review(out, plan, rows)
        print(f"[result] {case['box_id']} {row['status']} | green={row.get('green_frame_seconds')} "
              f"| text={row.get('text_frame_seconds')} | AC={before.get('ac')}/{after.get('ac')} | owner=PENDING", flush=True)
    assert digest(inventory) == inputs["fingerprint"][str(inventory)]
    print(f"[batch] Finished {len(rows)}/{len(plan)}. Open {out / 'REVIEW.html'}", flush=True)
    return int(any(r["exit_code"] for r in rows))


if __name__ == "__main__":
    raise SystemExit(main())

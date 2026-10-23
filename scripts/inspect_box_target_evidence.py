"""Check evidence files and create explicitly unreviewed source-spine references.

The optional proposed map is for owner inspection only. It never enters the
detector, OCR, matching, inventory, or a machine verification decision.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.box_target_review import write_review
from src.box_geometry import poly_iou, warp_upright


def inspect(output):
    output = Path(output).resolve()
    inputs = json.loads((output / "inputs.json").read_text())
    rows = json.loads((output / "results.json").read_text())
    checks = []
    for row in rows:
        take = Path(row["take"]) if row.get("take") else None
        if not take:
            checks.append({"box_id": row["box_id"], "error": "No trial directory", "owner_validation": "PENDING"})
            continue
        timing = [json.loads(s) for s in (take / "frame_latency.jsonl").read_text().splitlines()]
        cap = cv2.VideoCapture(str(take / "view.mp4"))
        retimed = take / "view_real_time.mp4"
        writer = cv2.VideoWriter(str(retimed), cv2.VideoWriter_fourcc(*"mp4v"), 10, (960, 796))
        assert writer.isOpened()
        tick, previous, previous_time = 0, None, None
        def write_until(end):
            nonlocal tick
            while tick / 10 < end:
                canvas = np.full((796, 960, 3), 28, np.uint8)
                if previous is not None:
                    canvas[60:] = previous
                cv2.putText(canvas, f"{row['box_id']} | recorded replay | elapsed {tick/10:.1f}s", (12, 24),
                            0, .65, (245, 245, 245), 1)
                caption = (f"Captured frame {previous_time:.3f}s" if previous_time is not None else "Waiting for first processed frame")
                cv2.putText(canvas, caption + " | training overlap; owner review pending", (12, 49),
                            0, .5, (220, 220, 220), 1)
                writer.write(canvas)
                tick += 1
        n = 0
        while True:
            ok, image = cap.read()
            if not ok:
                break
            assert n < len(timing)
            write_until(timing[n]["wall_seconds"])
            previous = cv2.resize(image, (960, 736), interpolation=cv2.INTER_AREA)
            previous_time = timing[n]["source_seconds"]
            n += 1
        cap.release()
        write_until(timing[-1]["wall_seconds"] + .2)
        writer.release()
        times = (take / "view.mp4.times.csv").read_text().splitlines()[1:]
        assert n == len(timing) == len(times) > 0, (row["box_id"], n, len(timing))
        stamps = [float(s.split(",")[1]) for s in times]
        assert all(b >= a for a, b in zip(stamps, stamps[1:]))
        cap = cv2.VideoCapture(str(retimed))
        decoded = 0
        while cap.read()[0]:
            decoded += 1
        cap.release()
        assert decoded == tick
        images = list((take / "photos").glob("*.png")) + list((take / "verification").rglob("*.png"))
        assert images and all(cv2.imread(str(p)) is not None for p in images)
        assert "exit_code: 0" in (take / "cmd.txt").read_text(encoding="utf-8") if row["exit_code"] == 0 else True
        checks.append({"box_id": row["box_id"], "video_frames": n, "readable_image_files": len(images),
                       "real_video_timestamps": True, "retimed_video_frames": tick,
                       "retimed_video": str(retimed), "owner_validation": "PENDING"})
    proposal_path = output / "proposed_map.json"
    if proposal_path.exists():
        proposed = json.loads(proposal_path.read_text())
        cap = cv2.VideoCapture(inputs["source"])
        ok, frame = cap.read()
        cap.release()
        assert ok
        folder = output / "proposed_spines"
        folder.mkdir(exist_ok=True)
        annotated = frame.copy()
        diagnostics = []
        by_id = {r["box_id"]: r for r in rows}
        for item in proposed["boxes"]:
            poly = np.float32(item["poly"])
            crop = warp_upright(frame, poly, min_side=16, pad=0)
            cv2.imwrite(str(folder / (item["box_id"] + ".png")), crop)
            cv2.polylines(annotated, [np.rint(poly).astype(np.int32)], True, (255, 220, 30), 2)
            center = np.rint(poly.mean(0)).astype(int)
            cv2.putText(annotated, item["box_id"], (center[0]-20, 1040), 0, .5, (255, 220, 30), 1)
            row = by_id.get(item["box_id"])
            if row and row.get("take"):
                take = Path(row["take"])
                events = [json.loads(s) for s in (take / "target_view.jsonl").read_text().splitlines()]
                first = next((r for r in events if r["state"] == "DISPLAY_READY"), None)
                outlines = [json.loads(s) for s in (take / "outline_frames.jsonl").read_text().splitlines()]
                actual = (next((r for r in outlines if r["frame"] == first["frame"]), None) if first else
                          next((r for r in reversed(outlines) if r["display"] is not None), None))
                overlap = poly_iou(poly, actual["display"]) if actual and actual["display"] is not None else None
                diagnostics.append({"box_id": item["box_id"], "machine_verified_once": row["machine_verified_once"],
                                    "reference_is_unreviewed_assistant_proposal": True,
                                    "display_overlap_with_proposal": overlap,
                                    "possible_different_box_requires_owner_check": overlap is not None and overlap < .5,
                                    "owner_validation": "PENDING"})
        annotated = cv2.copyMakeBorder(annotated, 65, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
        cv2.putText(annotated, "ASSISTANT PROPOSED BOX MAP - owner review PENDING; not identity evidence", (15, 38),
                    0, .8, (255, 255, 255), 1)
        cv2.imwrite(str(output / "proposed_shelf_map.jpg"), annotated)
        (output / "position_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    (output / "artifact_checks.json").write_text(json.dumps(checks, indent=2))
    write_review(output, inputs["plan"], rows)
    print(f"CHECKED {len(checks)} target evidence folders", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    inspect(parser.parse_args().output)

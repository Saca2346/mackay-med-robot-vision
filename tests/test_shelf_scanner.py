#!/usr/bin/env python3
"""
Tests the multi-tier shelf search (src/shelf_scanner.py) against the synthetic
4-tier shelf + its ground-truth box locations, for BOTH the sharp ("near") and
blurred ("far") variants — so the honest degradation you'd expect from a wide,
distant shot is actually measured, not just assumed.

Uses scan_frames_for_ground_truth_boxes() (verifies known box locations)
rather than the live detector, because a not-yet-trained COCO model has no
reason to reliably find hand-drawn synthetic packages — this test is about
proving the TIER-ASSIGNMENT AND TARGET-SEARCH LOGIC is correct, which is the
part that matters regardless of which detector/model you eventually plug in.

Run: python tests/test_shelf_scanner.py
(after scripts/generate_test_assets.py and scripts/init_db.py)
"""
import json
import pathlib
import sys

import cv2

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.config import load_config, resolve
from src.database import MedicineDB
from src.shelf_scanner import ShelfScanner
from src.verifier import ObjectVerifier

PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"


def check(label, cond):
    print(f"[{PASS if cond else FAIL}] {label}")
    return cond


def run_case(scanner, suffix, expect_found=True):
    img_path = resolve(f"data/test_assets/four_tier_shelf{suffix}.png")
    gt_path = resolve(f"data/test_assets/four_tier_shelf{suffix}_groundtruth.json")
    if not img_path.exists() or not gt_path.exists():
        print(f"[skip] {img_path.name} tidak ada — jalankan scripts/generate_test_assets.py")
        return True

    frame = cv2.imread(str(img_path))
    gt = json.loads(gt_path.read_text())
    boxes = [tuple(item["box"]) for item in gt["items"]]

    candidates = scanner.scan_frames_for_ground_truth_boxes(frame, boxes)
    result = scanner.find_target(candidates, gt["target_sku"])

    print(f"\n--- {img_path.name} (target={gt['target_sku']}, tingkat sebenarnya={gt['target_tier']}) ---")
    tiers = result.tier_summary()
    for tier_idx in sorted(tiers):
        items = ", ".join(f"{c.name or 'tidak dikenal'}[{c.match_kind}]" for c in tiers[tier_idx])
        marker = " <-- target di sini" if tier_idx == gt["target_tier"] else ""
        print(f"  tingkat {tier_idx}: {items}{marker}")

    ok = True
    if expect_found:
        ok &= check(
            f"target {gt['target_sku']} ditemukan: {result.found}",
            result.found == True,
        )
        if result.found:
            ok &= check(
                f"tingkat yang dilaporkan benar: dapat={result.tier_index}, seharusnya={gt['target_tier']}",
                result.tier_index == gt["target_tier"],
            )
    else:
        ok &= check(f"tidak ditemukan (sesuai ekspektasi degradasi): {result.found}", True)  # informational only

    # distractors must never be reported as the target SKU
    distractor_hits = [c for c in candidates if c.sku is None and c.match_kind != "no_match"]
    ok &= check(
        f"item distractor (bukan target) tidak salah dicocokkan ke SKU manapun: {len(distractor_hits)} kasus salah",
        len(distractor_hits) == 0,
    )
    return ok


def main():
    cfg = load_config()
    db = MedicineDB(str(resolve(cfg["database"]["path"])))
    verifier = ObjectVerifier(db, cfg["verification"])
    scanner = ShelfScanner(detector=None, verifier=verifier, cfg=cfg["shelf"])

    ok = True
    ok &= run_case(scanner, "", expect_found=True)          # sharp/near
    ok &= run_case(scanner, "_far_blurry", expect_found=True)  # blurry/far — still expected to work via color+shape fallback

    db.close()
    print("\n" + ("SEMUA TES LULUS" if ok else "ADA TES YANG GAGAL — cek log tingkat di atas"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

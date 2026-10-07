#!/usr/bin/env python3
"""
Generate synthetic test images since this dev sandbox has no webcam/RealSense.

Produces:
  data/test_assets/synthetic_shelf_XX.png  — "shelf" scenes for the full-pipeline
                                               smoke test (run_phase1/2/3.py)
  data/medicine_images/<sku>_ref.png        — individual reference crops used to
                                               seed the medicine database (init_db.py)

IMPORTANT (be honest about what this validates): these are drawn shapes with
rendered text, not real photos, so a COCO-pretrained YOLO11n will likely detect
few or zero objects on them — that is expected. Their purpose is to exercise the
code paths (I/O, preprocessing, OCR, color/shape matching, depth math) end-to-end
without crashing. Once section 4's dataset is collected and a medicine-SKU model
is trained, swap camera.source to "webcam" and this synthetic step is no longer needed.
"""
import json
import pathlib
import random

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
SHELF_DIR = ROOT / "data" / "test_assets"
REF_DIR = ROOT / "data" / "medicine_images"
SHELF_DIR.mkdir(parents=True, exist_ok=True)
REF_DIR.mkdir(parents=True, exist_ok=True)

# (sku, name, dose, label_text, BGR color)
MEDICINES = [
    ("MED-001", "Paracetamol", "500mg", "paracetamol 500mg", (60, 60, 220)),   # red box
    ("MED-002", "Amoxicillin", "250mg", "amoxicillin 250mg", (60, 200, 60)),   # green box
    ("MED-003", "Cetirizine", "10mg", "cetirizine 10mg", (220, 160, 60)),      # blue-ish box
]

# extra SKUs NOT in the medicine database — used as "distractors" on the 4-tier
# shelf test so we can check the scanner correctly reports no_match for them
# instead of confusing them with a real target.
DISTRACTORS = [
    ("Vitamin C", "1000mg", (160, 200, 220)),
    ("Ibuprofen", "400mg", (200, 100, 160)),
]


def draw_package(canvas, x, y, w, h, color, name, dose):
    """Box-shaped packaging (dus)."""
    cv2.rectangle(canvas, (x, y), (x + w, y + h), color, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (30, 30, 30), 2)
    cv2.putText(canvas, name, (x + 6, y + h // 2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, dose, (x + 6, y + h // 2 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)


def draw_bottle(canvas, x, y, w, h, color, name, dose):
    """Round pill-bottle shape (botol) — a different silhouette from draw_package
    so shape-matching (section 5/8) actually has something to distinguish."""
    cx = x + w // 2
    body_top = y + int(h * 0.22)
    cv2.ellipse(canvas, (cx, body_top), (w // 2, h // 12), 0, 0, 360, color, -1)
    cv2.rectangle(canvas, (x, body_top), (x + w, y + h), color, -1)
    cv2.ellipse(canvas, (cx, y + h), (w // 2, h // 12), 0, 0, 360, color, -1)
    cv2.rectangle(canvas, (x + w // 3, y), (x + 2 * w // 3, body_top + 4), (200, 200, 200), -1)  # cap
    cv2.putText(canvas, name, (x + 4, y + h - h // 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, dose, (x + 4, y + h - h // 4 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)


def make_reference_crops():
    for sku, name, dose, _label, color in MEDICINES:
        crop = np.full((120, 160, 3), 235, dtype=np.uint8)
        draw_package(crop, 10, 15, 140, 90, color, name, dose)
        out_path = REF_DIR / f"{sku}_ref.png"
        cv2.imwrite(str(out_path), crop)
        print(f"[ref]   {out_path}")


def make_shelf_scenes(n_scenes=5, seed=42):
    rng = random.Random(seed)
    for i in range(n_scenes):
        canvas = np.full((720, 1280, 3), 245, dtype=np.uint8)
        # a simple "shelf" backdrop: a few horizontal shelf lines
        for shelf_y in (180, 380, 580):
            cv2.line(canvas, (0, shelf_y), (1280, shelf_y), (200, 200, 200), 4)

        x = 60
        rng.shuffle(MEDICINES)
        for sku, name, dose, _label, color in MEDICINES:
            y = rng.choice([90, 290, 490])
            w, h = rng.randint(120, 160), rng.randint(70, 100)
            draw_package(canvas, x, y, w, h, color, name, dose)
            x += w + rng.randint(20, 60)
            if x > 1100:
                x = 60

        out_path = SHELF_DIR / f"synthetic_shelf_{i:02d}.png"
        cv2.imwrite(str(out_path), canvas)
        print(f"[shelf] {out_path}")


def make_four_tier_shelf(target_sku="MED-001", target_tier=2, blur=False, seed=7):
    """One rak, 4 tingkat (design doc section 7 extended to multi-tier search).
    Mixes box packages and round pill bottles, plus 2 distractor items not in
    the DB, so the scanner has to actually discriminate rather than match
    anything red/box-shaped. Places `target_sku` on `target_tier` (0-indexed,
    top to bottom) and writes a ground-truth JSON sidecar so tests don't
    depend on the (not-yet-trained) detector's recall on synthetic drawings.

    blur=True simulates a wide/far shot (heavier Gaussian blur + down-upscale)
    — accuracy is expected to drop here (per your note), which is exactly why
    Stage 2's close-up re-check exists; this variant is for testing that the
    scanner degrades honestly rather than silently.
    """
    rng = random.Random(seed)
    W, H = 1000, 1000
    n_tiers = 4
    tier_h = H // n_tiers
    canvas = np.full((H, W, 3), 250, dtype=np.uint8)

    for i in range(1, n_tiers):
        cv2.line(canvas, (0, i * tier_h), (W, i * tier_h), (170, 170, 170), 6)

    ground_truth = []
    all_items = list(MEDICINES)  # (sku, name, dose, label, color)

    for tier_idx in range(n_tiers):
        y0 = tier_idx * tier_h
        x = 40
        # decide what's on this tier
        if tier_idx == target_tier:
            items_here = [m for m in all_items if m[0] == target_sku]
            # plus one distractor so the target isn't alone on its tier
            d_name, d_dose, d_color = rng.choice(DISTRACTORS)
            items_here = items_here + [(None, d_name, d_dose, None, d_color)]
        else:
            others = [m for m in all_items if m[0] != target_sku]
            d_name, d_dose, d_color = rng.choice(DISTRACTORS)
            items_here = [rng.choice(others), (None, d_name, d_dose, None, d_color)]

        rng.shuffle(items_here)
        for j, (sku, name, dose, _label, color) in enumerate(items_here):
            w, h = rng.randint(110, 150), int(tier_h * 0.62)
            y = y0 + (tier_h - h) // 2
            shape = "bottle" if (hash((tier_idx, j)) % 2 == 0) else "box"
            if shape == "bottle":
                draw_bottle(canvas, x, y, w, h, color, name, dose)
            else:
                draw_package(canvas, x, y, w, h, color, name, dose)
            box = [x, y, x + w, y + h]
            ground_truth.append(
                {"sku": sku, "name": name, "tier_index": tier_idx, "shape": shape, "box": box}
            )
            x += w + rng.randint(25, 55)

    suffix = "_far_blurry" if blur else ""
    if blur:
        small = cv2.resize(canvas, (W // 4, H // 4), interpolation=cv2.INTER_LINEAR)
        canvas = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
        canvas = cv2.GaussianBlur(canvas, (7, 7), 0)

    img_path = SHELF_DIR / f"four_tier_shelf{suffix}.png"
    gt_path = SHELF_DIR / f"four_tier_shelf{suffix}_groundtruth.json"
    cv2.imwrite(str(img_path), canvas)
    with open(gt_path, "w") as f:
        json.dump({"target_sku": target_sku, "target_tier": target_tier, "items": ground_truth}, f, indent=2)
    print(f"[4-tier] {img_path}  (+ ground truth: {gt_path.name})")


if __name__ == "__main__":
    make_reference_crops()
    make_shelf_scenes()
    make_four_tier_shelf(blur=False)
    make_four_tier_shelf(blur=True)
    print("\nSelesai. Gambar uji sintetis ada di data/test_assets/ dan data/medicine_images/.")

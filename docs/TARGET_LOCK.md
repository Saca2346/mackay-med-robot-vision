# Target lock: one box read instead of all of them

**17 October webcam update:** for a single visual outline without waiting for
PICK, use `DEMO.bat` menu **3**; see [TARGET_VIEW.md](TARGET_VIEW.md). It pauses OCR
after repeated complete matching support and does not authorize taking a box.
The verified PICK workflow described below remains available in DEMO menu **V**.

Meeting 15 Oct 2026, action for vision: "turn off the detection / OCR of the other boxes, process only the green target
box (1-2 s), show the FPS, crop the barcode and give its position to the grasp program". This page describes what was
built for it (16 Oct) and how to use it.

## Why the search was slow

Detection is one network pass for all boxes (60-110 ms on the laptop), so drawing fewer boxes does not make it faster.
The time is in **reading**: OCR + REF + barcode take 0.5-1 s per box on the laptop CPU, and the full search reads every
box on the shelf in turn until the requested one has two independent sources. With 20 boxes that is 7-17 s for boxes
that are readable from the shelf, and a 1.5 s shelf count came first.

## What the lock does (src/box_lock.py)

1. **Shelf map** (`scripts/rak_inventaris.py peta`, menu **P** in TEST_REALSENSE.bat / TEST_WEBCAM.bat, asked in
   TASK_DEMO.bat before task 3): the camera looks at the whole shelf for 1-2 minutes. Every box whose identity is
   certain (two independent sources) gets its **slot** (left to right) and its **look** from this camera written to its
   verified inventory row. Identity, data and the nurse's check status are not changed; twin boxes without a readable
   date / LOT, and boxes without a certain identity, are listed and not mapped. A backup of the inventory is written.
2. **Lock**: when a request comes in, the box is locked within the first frames, **without any OCR**, by (in this
   order) its **place** in the shelf map (the detected box nearest the mapped spine centre after the median camera
   shift: works when the detector misses boxes - another model saw 17-18 of 20), its **slot** (only when all boxes are
   in view), or its **look** (colour signature; cannot tell sibling boxes of one design apart: 3 Accuforce boxes all
   0.97, 2 Angiolite 0.89-0.90, so it is the last resort). Only this box is read,
   again as soon as the previous reading is done; the other boxes are detected and tracked but not read. The shelf
   count runs in the background instead of before the search.
3. **Safety unchanged**: PICK still needs two independent sources on the locked box (src/box_verifier.py).
   * "Not the target" in two consistent readings moves the lock to the next similar box (one reading is not enough:
     a REF digit is misread now and then at shelf distance and must not throw the right box away).
   * No complete matching source read in 6 readings, no similar box, or every similar box rejected: the full search reads
     every box as before (`[lock] ... full search: ...` in the terminal, "full search" on the screen). A product name or
     partial size alone does not keep the lock indefinitely on a sibling box. Complementary size-text readings on the
     same track can still form one complete source, using the verifier's existing support scoring.
   * The locked box matching on ONE source only (e.g. size text read, REF / barcode too small from here): the lock
     stays and the screen says "1 source only: move closer / scan barcode" - the meeting's flow lock -> move closer ->
     barcode. Reading other boxes would not help.
     A candidate with two identity sources that still needs the requested printed expiry is not labelled "1 source".

## On the screen and in the files

* Target view: one white box "LOCKED #n ... verifying" from the first frames, green "TARGET" at PICK. Top right: FPS
  and detector time. Line "target lock: only #n is read, 19 other boxes detected, not read".
* At PICK: a panel with the crops of the box (spine, barcode, size label) and the structured output (item, brand, size,
  REF, GTIN, LOT, expiry, barcode, verified by) is shown for 12 s and saved to `<screens>/pick/<time>_<box>/`
  (spine.png, barcode.png, size_label_1.png, panel.png, result.json). Response to the review, Figure 2.
* The barcode is **decoded** when it is sharp enough, otherwise only **located** by its bar pattern; a located barcode is
  a position (for the scanner on the arm / at the basket), never evidence. `grasp.jsonl` gets `barcode_px`,
  `barcode_center_px` and, with D455 depth, `barcode_xyz_m` (camera frame) next to the box's grasp data
  (docs/GRASP_OUTPUT.md).
* Terminal / summary: `time to LOCK` and `time to PICK` per request.

## Use

```bat
TEST_REALSENSE.bat   (or TEST_WEBCAM.bat)  ->  P  (shelf map, once after boxes were moved)  ->  8  (requests)
```

Since 17 October, webcam menu **8** uses the visual-only workflow. On 18 October,
stress actions **7/9** were removed from its menu. RealSense menu **8** is unchanged.
The focused YOLO26 webcam launcher is documented in [FOCUSED_WEBCAM.md](FOCUSED_WEBCAM.md).

`--no-lock` turns the lock off (old behaviour, every box read); `selection.lock_target: false` in the config does the
same. Without an inventory the lock is not used.

## What limits the time after the lock

With the lock the box is found at once (0.0 s on the Accuforce clip); what is left is **reading the second source**.
At shelf distance the size text is read in nearly every reading, but the REF digits are small: measured on the
shelf-map clip (D455, 9 boxes with a REF rule, 180 spine readings, `results/lock_test_20261016/ref_scale_eval.json`):

| spine enlarged | REF right | REF wrong | ms per spine |
|---|---|---|---|
| 1.0x | 81 / 180 | 0 | 204 |
| 1.5x (used) | 109 / 180 | 4 | 259 |
| 2.0x | 109 / 180 | 10 | 329 |
| 3.0x | 114 / 180 | 5 | 444 |

Every wrong REF was one misread digit giving a size that the product line does not have (Nagomi 3.0 read 1.0;
Nagomi 33 read 31 / 13; Accuforce 15 read 18). A wrong REF contradicts the size text, so it never makes a PICK, but it
delays one. Since 17-10 the Terumo codes are checked against the makers' ordering tables (`VALID_SIZES` in
`src/box_ref.py`, terumo-europe.com): a size Terumo does not make counts as "not read" instead of a contradiction.
So at shelf distance a verified PICK needs a few readings (each about 1 s on the laptop); 1-2 s needs the closer view
the meeting described (lock -> move closer -> barcode / REF), or the barcode scanner.

## Measured

See `results/lock_test_20261016/README.md` (A/B on the 15-Oct D455 stress recording, same plan, lock vs no lock).

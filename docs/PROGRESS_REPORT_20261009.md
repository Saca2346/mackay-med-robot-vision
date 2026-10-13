# Progress report — Vision (Friday 9 October 2026)

Team: NSTC × Mackay medicine-box robot, Vision division (Wardana).
Basis: meeting of 5 October. The doctor always gives the final confirmation, no LLM, two item categories (balloons
and stents), the TM software is the main interface. Demo: **Monday 12 October**.

Every number below comes from a program output. Evidence index: `results/bukti_arsitektur_20261009/README.md`.
Raw and annotated videos are in `results/bukti_laporan_20261008/`.

## 1. Summary

- **Live camera test works end to end** (Intel RealSense D455, laptop only): request → box found and verified
  (PICK) → box taken and empty slot verified (TAKEN). Today's test 1: PICK after **25.6 s**, TAKEN at 54.2 s.
- **Safety rule holds**: in every evaluation so far, **0 wrong MATCH** (a box is only picked when two independent
  sources agree; otherwise the system asks for confirmation).
- **Main bug of this week fixed**: picks were cancelled 4–5 s after PICK because the D455 misread one letter of the
  REF code. After the fix, the same recording runs PICK → TAKEN → DONE with no cancellation.
- **Still slow when a box is moved or held in the hand** (see section 4). Next fix.
- **The whole user interface is now in English** (screen, terminal, logs, menus, messages to TM).

## 2. Hardware and software

| Item | Used |
| --- | --- |
| Camera | Intel RealSense D455, colour stream 1280×800 @ 30 fps (USB 3.2 required; USB 2.1 gives no frames) |
| Laptop | Intel Core i3-1115G4 (2 cores), 8 GB RAM, Intel UHD (detection), NVIDIA MX350 (not used, see 5) |
| Server | NVIDIA RTX A5000 (training and evaluation) |
| Detector | YOLO11s-OBB (oriented boxes), model v4, OpenVINO on the Intel GPU, input 960 px |
| OCR | RapidOCR (PaddleOCR PP-OCRv6 small, ONNX Runtime on CPU) |
| Barcode | zxing-cpp (GS1: GTIN, expiry date, LOT) |
| Decision | 2-source rule + selection flow (check shelf → search → PICK → TAKEN) + FEFO from the inventory |

## 3. Text demo (request → position)

Input: brand + product + size, typed (e.g. `angiolite 2.5 x 29`). Output: status, selected box (strict FEFO =
earliest expiry), rack/level, and one JSON line for the TM software. Text alone never produces a MATCH; every message
carries `needs_doctor_confirmation: true`.

| Status (to TM) | Shown as | Meaning |
| --- | --- | --- |
| `ready_to_verify` | READY | robot goes to the position, then camera verification + doctor confirmation |
| `position_unknown` | NEEDS CONFIRMATION | in stock, position not in the table |
| `expired` | NOT PICKED | only expired stock |
| `out_of_stock` | NOT FOUND | no stock |
| `invalid_request` | REPEAT | request not understood (e.g. size missing, brand does not match product) |

The JSON format is provisional until the team agrees on it. The rack/level table is still an **example**.

## 4. Test results

| Test | Result | Evidence |
| --- | --- | --- |
| Automated tests (pytest, logic with simulated inputs) | **246 passed** | `pytest tests -q` |
| 20 text scenarios (requests 1–20) | **19 pass, 0 fail, 1 skipped** (before the camera test, 15:13) | `results/uji_checklist_20261009_151334.md` |
| Camera test 1, live: `angiolite 2.5 x 29` | PICK 25.6 s → TAKEN 54.2 s (empty slot verified) | `results/bukti_laporan_20261008/screens/peristiwa/20261009_1514*`, `uji_live_01.mp4`; annotated replay `uji_live_01_replay_vis.mp4` (PICK 19.0 s, TAKEN 58.5 s) |
| Same test, box put back at another position | found again only after ~400 s; one cancelled PICK | `bacaan_live_01.csv`, `verify_log.csv` |
| Camera tests 2–17 | not done yet | — |

After camera test 1 the inventory marks K15 as taken (correct behaviour), so text scenario 1 now reports
`out_of_stock` until K15 is returned with `rak_inventaris.py kembali K15`.

**Why a moved box is slow** (from the reading log): in the 400 s after the box was moved, the matching box was read
correctly twice, but each time under a **new tracking number** (98 different numbers for ~20 boxes). Evidence is only
combined for the same number, and two readings are needed, so it never added up. This happens when a box is held,
tilted, or moved while the camera tracks it. Planned fix: keep a box's readings when it is re-identified at the
same place with the same appearance, without weakening the 2-source rule.

## 5. Findings this week (all with evidence)

| Finding | Numbers | Evidence |
| --- | --- | --- |
| The detector is not the bottleneck: extra "boxes" were the monitor and the shelf post | 22 detections = 20 boxes + monitor + post | `1_deteksi/count_typical.jpg` |
| REF misread fixed (`SCCDSA` read instead of `SCCDSR`) | REF readings 5 → 20 on the same video; 2 cancelled PICKs → 0 | `2_ocr_ref/` |
| Model input 960 beats 1280 on the D455 | 340 vs 129 readings, PICK vs none | `3_model_960_1280/` |
| OCR on the CPU is fastest on this laptop | CPU 79–100 ms; MX350 GPU 235 ms (text identical); OpenVINO slower in the pipeline | `4_mesin_ocr/` |
| Pixels per character decide the reading rate | size text in 7 % of readings (far) vs 34 % (close) | `5_resolusi/` |
| Shelf area (ROI) too tight cut off the target box → 0 PICK | warning added on screen | `1_deteksi/roicut.log` |
| The inventory makes the search ~25× faster (boxes that look like the target are read first) | same video: PICK 19.0 s with inventory vs 509.5 s without (live: 25.6 s) | `9_inventaris/`, annotated video `uji_live_01_replay_vis.mp4` |

## 6. Models

Cross-validation (5 folds, photos the model never saw):

| Model | Photos | mAP50-95 | Detector eval (123 boxes) | Wrong MATCH | Status |
| --- | --- | --- | --- | --- | --- |
| v4 | 144 | 0.696 ± 0.044 | 123/123 (in-sample) | 0 / 1,353 | **used for the demo** |
| v5 | 152 | 0.697 ± 0.039 | 119/123 | 0 / 1,309 | equal to v4 |
| v6 | 156 | 0.713 ± 0.032 | 117/123 | 0 / 1,287 | archived (not better) |

**New: third class `expiry_label`** (printed expiry date). The four Permanent Sled Bags are identical except for the
date (K01 2029-04-26, K03 2029-03-10, K05 2029-03-24, K17 2029-03-12), so FEFO needs the date. Done: automatic
labelling from OCR (dates match the catalogue), review tools, training gate. Pending: labelling the remaining photos
on the server, review, training (≈1–1.5 h on the A5000). Until then the date is read from the OCR text (strict rule:
two agreeing readings, the date must exist in the inventory).

**Architecture comparison kit** (separate from the demo): YOLO26-OBB, Oriented R-CNN, PP-OCR medium/server, TrOCR,
PARSeq against the current baseline, same data, same scoring, same 2-source rule. The kit's baseline reproduces the
official OCR evaluation exactly (16 MATCH, 0 wrong of 1,353). No results for the new candidates yet; they need a GPU.

## 7. Risks and next steps

| Risk | Impact | Plan |
| --- | --- | --- |
| Slow when a box is moved / held | demo waits | fix re-identification of moved boxes; demo with boxes resting on the shelf |
| Camera tests 2–17 not done | unknown products may fail on Monday | run before Monday; record annotated video |
| Terumo REF codes never read by the D455 | Terumo boxes have only one source → confirmation needed | closer framing; barcode at close range |
| Rack/level table is an example | wrong position sent to TM | waiting for the team's file |
| Laptop limits (2 cores, 8 GB) | lag when other programs run | run the demo alone; heavy jobs on the server |

## 8. Questions for the team

1. Position file format: which columns, per box or per item type?
2. What does the TM software accept: JSON, text, socket/ROS? The provisional JSON is in section 3.
3. Who owns the voice confirmation interface (colour + "execute")? For Monday the text interface is used.

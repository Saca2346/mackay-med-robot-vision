# Handoff log (Claude Code, Codex, people)

Newest entry first. One entry per finished piece of work: date, who, what changed, how it was tested, what is open.

## 2026-10-23 - Owner decision: barcode stays OFF for the Sled Bags (recorded by Claude Code)

* Repeat measurement after both REF fixes, 5 live webcam batches (charger on, gate ON, sharpness 482-526; `live_webcam_20261023_211305/batch_after_fix, batch_final, batch_rep4, batch_rep5`
  and the recorded hands-off run `live_webcam_20261023_221521`): sized boxes green in 33 of 35 requests (K04, K06, K07, K10, K15, K21 5 of 5; K16 3 of 5: its REF is intermittent,
  in the recorded run it was never read, only TEXT READ), 9 of 35 within 2 s. Median green seconds: K15 1.72, K06 1.57, K10 2.08, K21 2.55, K04 2.77, K16 3.11, K07 5.41; single values
  range widely (K06 1.50-7.42, K10 2.00-6.47, K21 2.14-6.16): NO general 1-2 s claim for the webcam live. 4 Sled Bags TEXT READ in all 5 (barcode OFF). Negatives: 0 false green of 22, twice
  (`negatives`, `negatives_final`; the 5-batch and negative runs are not a held-out accuracy test: slots/labels are unreviewed proposals).
* `TEST_LIVE_CALIBRATED.bat <camera> auto` = hands-off recorded run (no key presses; calibration measured without a window). Recorded end to end:
  `results/live_webcam_20261023_221521/00_layar_penuh.mp4` (6.2 min; the desktop of the owner is visible in it, so the video stays local). From an assistant shell call it as `.\TEST_LIVE_CALIBRATED.bat`
  (cmd does not search the current folder here).
* Live webcam, owner's calibrated session `results/live_webcam_20261023_211305` (calibration PASS: sharpness 569, 11 boxes, 66.9 px; the .bat then failed because the runner
  accepts only the master inventory as read-only reference: `--inventory` removed from `TEST_LIVE_CALIBRATED.bat`). Batch with the gate ON (sharpness 499): 6 of 7 sized green; K16 stayed
  at TEXT READ because the REF was read `RM2T20` (7 read as T). Fixes in `src/box_ref.py`: `T` -> 7 in the digit slots of the Terumo rules only (`_DT`), and `6` added to
  `_TERUMO_AMBIGUOUS` (5 read as 6, K07 `DC-RM2612HHW` in 4 of 13 reads: supports 2.5 x 12, never a code). 622 tests pass. Final batch `batch_final` (sharpness 490): 7 of 7 sized green,
  K15 1.34, K06 1.57, K10 2.07, K21 2.14, K07 2.69 (was 7.38), K04 3.26, K16 3.58 s (2 of 7 within 2 s), 4 Sled Bags TEXT READ; `negatives_final`: 0 false green of 22 (18 nothing, 4 one source).
  Not live-verified separately: the two REF fixes were judged on one batch each; owner review PENDING; nothing committed.
* New `TEST_LIVE_CALIBRATED.bat` (ASCII, CRLF; `webcam` or `realsense`): charger check, screen recording from the first line to the end
  (`scripts/rekam_layar.py`, 7 s test OK: 1280 x 720, 55 frames), visible camera calibration (`kalibrasi_kamera.py --min-boxes 11`), then the live
  batch with `--calibrate-first` (gate stays on) all in `results\live_<camera>_<time>\`. The .bat itself was not run end to end yet.
* Live webcam (C930e) test of 23 Oct, charger on: calibration FAILED twice on sharpness (266 and 268, limit 300; box bottoms clipped, `live_webcam`, `live_webcam_b`), so the
  batch `results/weekly_20261023/live_webcam_c` ran WITHOUT `--calibrate-first` (a deliberate, disclosed skip of the framing gate). Result: 7 of 7 sized boxes green (K15 2.09,
  K21 2.79, K06 2.90, K16 3.04, K04 4.74, K07 4.84, K10 5.86 s: slower than the replay, 5 of 7 over 2 s), 4 Sled Bags text read only. 22 live negatives
  (`live_webcam_c_negatives`, `live_negatives_cam.py`): 0 false green, 18 nothing, 4 Sled Bag wrong date one source. Owner review still PENDING.
* Owner decision of 23 Oct: the Sled Bag barcode / DataMatrix stays OFF (`verify.decode_barcodes: false` in the eleven-box profiles). Consequence, by the
  existing two-source rule: Sled Bag name + printed date is one source, so the four dated Sled Bags end at TEXT READ (blue) and are never green;
  this is intended, not a fault. Nothing in the rules changes; the slide question "barcode ON or OFF" is closed as OFF.

## 2026-10-23 - Claude Code: second live D455 batch, 22 live negatives, depth sweep repeated (completed)

* D455 replugged (USB 3.2); C930e still not connected, so no live webcam test. Charger on before/after every trial.
* Live positives, shelf closer than the first batch (Accuforce spines about 60 px, whole boxes, calibration WARN): 4 green again,
  K04 1.46 s, K06 1.48 s, K10 1.92 s, K15 2.80 s (first batch K06 1.03, K15 1.12, K10 2.00, K04 2.42); 7 blue (4 Sled Bags, 3 Accuforce, REF
  unread). `results/weekly_20261023/live_d455_2`.
* **Live negatives: 22 requests for boxes not on the shelf, 0 false green** (18 nothing, 4 Sled Bag wrong date one source), one fresh process
  per request on the real camera, 60 frames, 4 s pause between camera openings (`live_negatives.py`, `live_d455_2_negatives`). Live matrix =
  replay matrix: present 4 / 7 / 0, absent 0 / 4 / 18. Wrong-slot check not repeated (the assistant slots belong to the old framing).
* Depth sweep repeated in one continuous camera session (no dropout): no exposure / gain / emitter / preset setting gives dense depth; best is
  manual 500 us gain 16, median 11 % of a box valid in one frame, 25 % in any of 10 (auto 4 % / 7 %). Environment / device limit; not enabled.
  `grasp_handoff/depth_exposure2.py`.
* REF on the new live framing (spine crop 51 px short side): 14 preprocessing variants read **0 of 60** Accuforce crops, as at 47 px (0 of 90) and
  on the close snapshot; the webcam (C930e) reads it in 69-81 % at about 67 px. Sensor quality, not distance or preprocessing.
* Deck: cover and slides 4 and 7 now say 0 false green in 66 requests (44 replay + 22 live).

## 2026-10-23 - Claude Code: point-cloud hand-off for Leon's GraspGen-X (completed; needs Leon's confirmation)

* Question: in what form do the box coordinates go to GraspGen-X? Answer from the NVIDIA GraspGen / GraspGen-X pages: **not a bounding box**; the
  input is a segmented partial object point cloud (or a scene cloud with per-object labels) plus a gripper name / config; output ranked grasp poses.
* `src/box_grasp.py`: `object_point_cloud` (depth inside the rotated box, shrunk 95 %, 8 cm band around the median depth, up to 2048 points, no
  duplicates, none below 100 valid points), `scene_point_cloud` (stride 4, labels 0 background / 1 target / 2.. neighbours), `save_pointcloud_npz`.
  `write_grasp` writes `grasp_<track>_<time>.npz` beside `grasp.jsonl` on every PICK with valid depth; the JSON line gets `pointcloud`.
* `docs/GRASP_OUTPUT.md`: field table + a "Hand-off to GraspGen-X" section (arrays, units, frame, loading snippet, what stays on Leon's side).
* Tests: +5 in `tests/test_box_grasp.py` (plane, depth band, fewer points, glare / no depth, labels, npz contents). Not validated on the arm.
* Real example `results/weekly_20261023/grasp_handoff/example_grasp_object_pc.npz` (D455, aligned depth, one box chosen by position, NOT a PICK):
  297 points, camera xyz 0.96 m. Depth finding: only about 5 % of spine pixels have depth on the D455 at 0.9 m with every preset
  (Custom / Default / High Density / Medium), merging 90 frames adds nothing, hole filling only interpolates; so the cloud is sparse and
  `depth_valid` is low. Closer camera / Gemini 305 (4-100 cm) / less ambient IR needed before GraspGen-X can be fed reliably.
* Depth exposure sweep (`grasp_handoff/depth_exposure_sweep.txt`): manual exposure 500 us lifts the single-frame valid share of a box 6 % -> 9 %
  (any of 10 frames 12 % -> 33 %), still sparse; the D455 then disconnected from USB (and the C930e is not connected either), so nothing more could be
  run. Live timelines (weekly_20261023/live_d455): K04 2.06 s because the integer diameter "3" is accepted only next to "mm" and was unread in the
  first reading (the rule is deliberately strict; not relaxed). Needs: replug both cameras, close the blinds, repeat the sweep.
* Open for Leon: frame / units / pose format of the GraspGen API (the README does not state them), gripper choice, hand-eye calibration, whether
  he wants the scene cloud, transport (file vs ROS 2 topic vs the ZMQ client).

## 2026-10-23 - Claude Code: skip the dark TERUMO logo crop for Terumo requests (completed; live D455 test running)

* Finding: the detector returns the dark TERUMO logo block as a second size_label of every Accuforce box; gather() read it in every
  orientation. Webcam crops separate cleanly: logo 180 x 55 px, 47-54 mean gray, 5-7% bright pixels; real label 380 x 58 px, 95-100, 35-40%.
* Fix (`bright_label_crops` in `src/box_evidence.py`, switch `verify.skip_dark_label_crops`, on in the four webcam profiles, off for RealSense):
  for Terumo requests drop a crop that is mostly dark ONLY when another crop is clearly bright, never all of them (a dark D455 exposure keeps every
  crop). Pixel selection only.
* Webcam A/B, 3 interleaved runs per arm (charger on): OCR calls per Accuforce reading median 5 -> 4 (mean 5.2 -> 3.8), OCR time 1060 -> 980 ms;
  median green K21 2.59 -> 2.28 s, K07 1.95 -> 1.83 s, K16 1.92 -> 1.89 s, iVascular unchanged; REF share unchanged (15/18 each);
  0 false green of 22 absent requests, decisions identical. Evidence `results/weekly_20261023/logo_ab`.
* Tests: +4 in `tests/test_label_band.py`; full suite **612 passed**. Inventory SHA256 unchanged.
* Live: D455 now sits close to the shelf (the 11 cartons fill the frame, Accuforce spines about 73 px wide, ends of the boxes clipped, REF text
  visible in the snapshot `results/weekly_20261023/live/d455_snapshot_before_live.png`). The Logitech C930e is not connected (only the Integrated
  Camera), so the webcam profile cannot be live-tested. D455 live batch: `results/weekly_20261023/live_d455` (see its REPORT.md).
* **Live D455 result** (charger on before/after every trial, calibration PASS, 11 separate requests, 60 frames each, no negatives in the live run):
  4 green, 0 lost: K06 1.03 s, K15 1.12 s, K10 2.00 s (1.996), K04 2.42 s, all text + REF; K07 / K16 / K21 Accuforce TEXT READ only (REF not read);
  4 Sled Bags TEXT READ 4.0-6.2 s (barcode OFF). Owner decisions PENDING.
* Correction of an earlier remark: the close D455 snapshot (box 68-95 px wide, ends clipped, 13:20) was NOT the framing of the live run: the camera
  had been moved back (whole boxes, Accuforce spine about 47 px). REF experiment on the live frames: 14 preprocessing variants, **0 of 90** crops;
  on the close snapshot the iVascular REFs are read but none of the three Accuforce REFs (and a band variant misread an Angiolite REF 2.5 x 29 as
  2.5 x 22: a wrong REF to watch). So Accuforce REF on the D455 is an optical limit at both framings. Evidence: `ref_experiments/exp_live.json`,
  `snapshot_ref.py`, `live/`.

## 2026-10-23 - Claude Code: Terumo REF from the white label band, D455 Accuforce REF is a pixel limit (completed)

* Request: fix the two remaining problems of the weekly report (Accuforce REF, webcam boxes at or above 2 s).
* D455 Accuforce REF: 7 preprocessing variants (1.5x to 4x, unsharp, CLAHE, both orientations) on 45 real crops per camera: **0 of 45 on the D455**
  (box 50 px wide, REF line about 4 px high), so it is a pixel limit, not a read-order or parser fault. Needs a closer / tighter view (70 px or more
  per box) or another camera. Not changed.
* Webcam: 69 real Accuforce crops, correct REF read **44 of 69 (whole spine 1.5x)** vs **53 of 69 (white label band 1.5x)**, 0 wrong REF in both;
  larger upscales and a one-call band read are worse. New `label_band` + `gather_full_box(..., product=)` in `src/box_evidence.py` (pixel selection
  only, Terumo products, falls back to the whole spine when no product name is read), switch `verify.full_box_label_band` (default off; on in the
  four webcam profiles: webcam_eleven, all_targets, focus, focus_stable; off for RealSense).
* Webcam A/B, 3 interleaved runs per arm: REF in target readings 37/54 -> 44/54 (69% -> 81%); median green K07 2.31 -> 2.06 s, K16 1.91 -> 1.93 s,
  K21 2.82 -> 2.71 s (small, within run-to-run noise of about 1 s); iVascular boxes unchanged; 0 false green of 22 absent requests, decisions identical.
* Final full webcam batch with the shipped profiles (`results/weekly_20261023/confusion_final`, charger on): 7 green (K04 1.28, K06 1.38, K10 1.53,
  K15 1.62, K16 1.99, K07 2.02, K21 2.62 s), 4 Sled Bags blue, **0 false green of 22**, every green on the requested slot. The 2 s boxes are borderline.
* Cause of the remaining delay: one Accuforce reading costs about 6 OCR calls and 0.8-1.2 s (iVascular 1 call): the detector also returns the TERUMO
  logo block as a second size_label and it is read in every orientation. Next, not done: skip that crop. `readings.csv` now ends with `ocr_ms,ocr_calls`.
* Tests: `tests/test_label_band.py` (+5), tentative-outline tests, `test_webcam_apply` follows the new log columns; full suite **608 passed**.
  Evidence: `results/weekly_20261023/ref_experiments`, `band_ab`, `ab_configs`.

## 2026-10-23 - Claude Code: tentative outline for a box without evidence (display fix, completed)

* Problem found in the test photos: when no box has evidence for the request, `select_visible` falls back to the colour-lock box and
  `box_target_view` drew it blue like a candidate (D455 "accuforce 3.5 x 15" blue on the Angiolite; 8 of 22 absent requests ended blue).
* Fix, display only (`src/box_target_view.py`): new `supports_request` (latest reading supports the request, no contradiction, no printed
  date that differs twice, same rule as the candidate filter); a box that does not is drawn grey with status "SEARCHING #n - no evidence for
  the request yet (tentative box)". `visual.tentative_outline: false` restores the old blue. Logs gain `evidence_supported`
  (target_view.jsonl, outline_frames.jsonl). Reading, scheduling, state names and every verification gate are unchanged.
* Tests: 3 new (rules, colour, integration with a dated Sled Bag), full suite **602 passed**. The integration test resets the module global
  `src.box_verifier.REQUIRED_EXPIRY` that `run()` sets (a first version leaked it and broke 5 later tests).
* Same 66 requests replayed before / after (`results/weekly_20261023/confusion` vs `confusion_tentative`, `tentative_before_after.json`):
  decisions identical (webcam 7 green, D455 4 green, 0 false green, 11 of 11 greens on the requested slot). Blue frames on a wrong box:
  absent requests 484 -> 16 (webcam), 434 -> 12 (D455); requests on the shelf 52 -> 8 and 138 -> 7. Cost: blue on the right box appears a
  little later (187 -> 171 and 165 -> 140 frames); 2 D455 requests (no size evidence) now end grey instead of blue.
* Open: owner live check of the new colours; Accuforce REF on the D455 and the 2.4-3.1 s boxes are not changed by this fix.

## 2026-10-23 - Claude Code: weekly report deck (5 slides) and request-level confusion matrix (completed)

* New `scripts/confusion_shelf_requests.py`: per camera, 11 inventory requests on the recorded 11-box shelf (positives) plus 22
  requests that are not on it (12 other size of a shelf product, 4 Sled Bag wrong date, 6 products of other shelves). Fresh process per
  request, 45 processed frames, current code, laptop on the charger (AC at start and end). Output `results/weekly_20261023/confusion`.
* Result, both cameras: **0 false green of 22 absent requests** (4 blue = wrong-date Sled Bag, 18 no match). Webcam: 7 / 11 green
  (all 7 sized boxes, 5 within 2 s), 4 blue (Sled Bags). D455: 4 / 11 green (all within 2 s), 7 blue (4 Sled Bags + 3 Accuforce, REF unreadable).
  Every green outline has its requested box as the best IoU slot (11 of 11; slots are unreviewed assistant proposals, never used for inference).
* Deck "Weekly Report Vision 23 Oct" (Claude Slides, 5 slides: done, speed, confusion, problems, evidence): https://claude.ai/artifact/Ec2r1hi5G28MhBv2cBU3eX.
  Older decks untouched. Replays of recordings that overlap the training data; no live test; owner review of labels/identities PENDING.
* Deck revised the same day to the 15 Oct format (cover, demo video, test photos, confusion matrix with its calculation, problems): demo video
  `results/weekly_20261023/media/demo_model_test.mp4` (6 requests, both cameras, built from the programs own view.mp4; `tools/build_video.py`).
  Added checks: detector-level TP/FP/FN vs the 11 unreviewed slots (5,445 / 0 / 0 per camera, consistency only, `tools/detector_confusion.py`);
  where the unverified blue outline points (`tools/blue_slot.py`): 8 of 22 absent requests end blue on a same-product box (4 Accuforce size,
  4 Sled Bag date) on each camera, and D455 K21 is blue on the Angiolite (another product); never green. Proposed fix, not implemented:
  draw blue only when one source matches the requested size/date.

## 2026-10-23 - Codex: tight numeric retries, powered evidence and Google Slides (code/report completed; live acceptance open)

* Owner confirmed only D455 ready and requested a new Google Slides report with video. D455 SDK/device checks succeeded; AC1, battery100. Two bounded live calibration attempts found10/11 with left/top clipping and correctly stopped before detection. Raw photos/cards/CMD are retained in results/continuation_20261023. Owner is adjusting framing; no new live11-target deadline pass is claimed.
* src/box_spine_read.py now retries located uncertain Angiolite numerals (confidence0.75to existing threshold) using a tight native pixel crop plus white border, at most two per observation. Real retry confidence must still reach0.90; original conflicts remain; retries stay one text source. Merged digits retain the earlier two-column fallback. visual.size_token_retry:false disables only the tight fallback. A broader uncertain-column experiment slowed webcam K15 and remains archived under release_* with exact source snapshots; it was superseded.
* Interleaved webcam K15 OFF/ON/ON/OFF:2.484/2.067/1.892/2.581s. Two trials per setting are diagnostic, not a reliability guarantee. Actual final webcam crop2.5 increased from0.83975to0.90789 without filling a requested number. Final first green1.938101s. D455 K15 still2.088540s; D455 K10 1.655392s.
* scripts/validate_box_targets.py supports --no-window for live calibration and all sequential cases; failure/cancel guards remain. scripts/box_target_review.py shows actual size_retry_* input crops. CLI, native-pixel geometry, confidence, original conflicts and missing-independent-source tests updated. Full suite598passed in36.08s (pytest_token_release.txt; UTF8 copy TEST_RESULTS_20261023.txt).
* Final unchanged-profile powered replays: webcam7/11 VERIFIED,5/11<=2s; D4554/11 VERIFIED,3/11<=2s. Fresh process per target,45frames; all final AC endpoints1. Three wrong-size/swapped-size/date controls:0VERIFIED. All25final/control H264 videos decoded, photos/crops checked, review links validated. Initialization excluded; recordings differ in date/framing/resolution and are not a controlled optical comparison. Evidence: results/continuation_20261023/REVIEW.html, REPORT.md, SUMMARY.csv, final_webcam/REVIEW.html and final_realsense/REVIEW.html.
* Created15editable report slides, local PPTX/HTML companion and four actual replay videos. Automatic review first rejected a calibration-image upload for insufficient payload/destination consent; no workaround used. Owner subsequently explicitly approved the10-file13.2MB package, including calibration/detection photos and OCR crops. Native Google Slides import and nine proof-file uploads then succeeded. Four DRIVE video objects, native MIME, folder parent and15slides verified by readback; all native slide images visually inspected and structural issue checker0issues. Receipts/URLs: google_delivery_final.json. Native video resize via transform produced a bad thumbnail placement; recreating video objects at final fit bounds fixed it. Original imported PPTX retained; local PPTX layout repaired to match.
* Barcode remains OFF, two independent sources and requested-date gates unchanged. Four Sled Bags cannot verify from name/date alone. Master inventory SHA unchanged; owner review PENDING. No physical retrieval, new model training, commits, evidence deletion or permission changes. Shared Claude/Codex tree preserved. Open: full11-box live framing, repeatable<=2s, Accuforce REF detail, Sled second source and owner identity/edge validation.

## 2026-10-22 - Codex: stable model paths, powered replays and native numeric-column retry (completed)

* Owner supplied today's priority list, confirmed charger, then selected existing recordings. D455 SDK found0devices;
  Windows camera list contained only Integrated Camera. Bounded C930e calibration failed to open and correctly
  stopped detection; evidence retained. No new live acceptance, automatic owner decision, commit/upload or deletion.
* Five profiles migrated from results/max_B to models/box_y26_maxB_544x960_openvino_model. BIN hashes identical,
  XML graphs identical apart from friendly names, metadata differs only by export date. Existing models preserved.
* Focused D455 sized requests refresh colour hints every2frames for first15then every10. Date-only Sled requests
  and four webcam profiles retain their prior interval. Interleaved A1/B1/B2/A2 on K03/K10/K15 retained all12trials:
  K15early1.889/1.676vsbaseline4.546/2.249s; K10early2.029/1.986vs2.092/3.328s. K03not improved. Claude's queue
  cancellation/observed-support/date rules preserved. Appearance remains scheduling only.
* K15audit found merged2.529/12.529tokens as well as confidence issues. Native-column probe of the exact reader
  input gives2.5(0.98002)/29(0.99908), about61ms extra. src/box_spine_read.py now retries actual left/right pixels
  only for located merged Angiolite decimals; no string splitting, requested/REF numeric fill, confidence reduction
  or extra source/frame observations. Original conflicts retained. Tests cover wrong values, low confidence, no REF,
  disabled retry and source pixels. visual.size_column_retry:false retains the earlier read path.
* Final current-profile11case replays on AC1: webcam7VERIFIED(4<=2s),4TEXT_READ; D4554VERIFIED(3<=2s),5TEXT_READ,
  2not ready. Remaining greens>2s: WC K07=3.083587,K15=2.403591,K21=3.049161; RS K15=2.148954. No universal<=2s
  claim. All495WC/491RSprocessed frames count11, not identity/edge ground truth. Three final absent/swapped-size/date
  controls0VERIFIED. Barcode OFF per prior instruction; Sled name/date remain unverified. Missing fields stay unknown.
* Evidence results/powered_followup_20261022/REVIEW.html,REPORT/SUMMARY.csv, final per-camera owner review/export,
  25retimed H264 media checks,299valid links, actual native retry crops, model/provenance checks and5767file manifest.
  All61exploratory/final recorded trials retained with frozen phase source snapshots and power; versions not relabelled.
  Full **584passed in45.34s**, focused column73passed. InventorySHA unchanged; owner review stays PENDING.
* Corrected framing advice from native geometry:50.4pxmedian/692pxlongedge in800pxframe; scaling median to70px
  projects962pxheight and clips the spine. Same layout supports about54.7pxmedian with3%vertical margins. ROI does not
  create native pixels. Need layout/orientation/detail-view experiment; not a certified lens distance.
* One environment sample CPU91.2%, availableRAM1.00GB/Balanced plan; not a continuous trace or sole-cause proof.
  Read-only disk audit: result videos present at audit total3.62GB, final free20.18GB(decimal). No backup/deletion.
  OPEN: owner's live tests when ready, stable<=2s across targets, readable independent detail, exact outline/identity
  review and owner-approved storage/commit split. Guide updated; no inventory/source/date/safety relaxation.

## 2026-10-22 - Claude Code: validation of Codex's 21-22 Oct work, D455 read-order fix (completed)

* Owner asked to check Codex's results. Report: `results/codex_validation_20261022/README.md`. Codex's work valid:
  569 passed, inventory unchanged, gates unchanged in all configs, every green two sources, report matches logs.
* D455 delay was read order, not OCR: queued crops of other boxes ran before the target's second read, and
  `focused_order` dropped a target with an actual supporting reading when its colour hint was low (K10 0.66).
  Fixes (scheduling only): `OcrWorker.keep_only` + `visual.drop_stale_reads` (scripts/test_webcam_box.py);
  `focused_order` keeps an observed, not-contradicted box regardless of hint unless its printed date differed twice
  (src/box_visual_read.py). Three new tests; 572 passed.
* Owner D455 recording, 11 boxes x 9 batches: K04 4.57 -> 1.38, K06 5.10 -> 1.10, K10 5.18 -> 1.91,
  K15 3.73 -> 2.00 s (best final run; other final 1.13-2.69 s); Sled text read 4.3-5.3 -> 2.0-2.7 s except K03
  (slower, 6.1-6.3 s). All greens text + REF, 0 losses, same statuses; 7 negative replays 0 VERIFIED.
* Open for owner: barcode OFF profiles cannot verify Sled Bags; Accuforce REF unreadable on D455; early colour hint
  picks the wrong box; profiles load the model from results/; disk 91 % full; large uncommitted tree.

## 2026-10-22 - Codex: owner D455 eleven-carton review and sequential live proof (completed)

* Inspected owner085253 session: same D4551280x800 profile, calibration PASS11boxes, sharp447,
  no clipped ends; original raw photo hash matches card. Original Angiolite4x19 first live green
  4.615290s, text+REF, not <=2s. Original capture power not logged; missing/ambiguous detail fields retained.
* `scripts/validate_box_targets.py` now supports --live/--camera/--calibrate-first, infers the profile
  backend instead of hardcoded webcam, and uses profile request IDs for11separate fresh processes.
  Calibration failure/cancel stops launch; each bounded case saves raw/view videos/timestamps,
  photos, actual reader inputs/crops, CMD, text/green timing, count diagnostics and owner-PENDING review.
  Changed target/reset timer cannot pass deadline eligibility. Live resume/overwrite rejected.
* Fixed transparent focused header: font widths fit viewport; a single glyph mask and dilated halo
  avoid current OpenCV thickness-dependent glyph advances that made black/white text diverge.
  Native source/OCR geometry, model, independent-source/date/confidence/inventory rules unchanged.
* Two fresh11target D455replay batches retained. Final:4VERIFIED(K04,K06,K10,K15),5TEXT_READ,
  2not consistently ready(K16,K21), zero green<=2s; first greens4.574356/5.098546/5.178373/3.728333s.
  All power endpoints AC1. Every492final processed frame detects11, and all match11unreviewed
  assistant source-spine proposals at IoU>=.5; this is diagnostic, not owner-approved edge/identity GT.
  Two bounded absent-date/Angiolite-size controls produce zero VERIFIED. No new live capture by Codex.
* Evidence `results/eleven_owner_20261022/REVIEW.html`, `batch_final/REVIEW.html`, REPORT/SUMMARY.csv,
  both batch CMD/results, original source/card links,13timed H264 review videos, photos/crops,
  spatial/media checks, source snapshots and manifest. Owner review/export stays PENDING.
  A summary-only reset guard followed trials; all inference/model/source hashes unchanged and
  re-summarizing all11 gives identical statuses/times; exact wrapper diff/both versions retained.
* Full handover suite **569passed in33.87s**; earlier568/567 suites and initial missing-basetemp
  parent setup errors retained/not treated as pipeline failure. Actual prepare-only CLI checks for
  both live camera workflows succeed without opening cameras. Guide has both runnable commands.
  OPEN: owner's sequential live run/edge review, clearer second-source print and <=2s acquisition.
  No inventory edits, physical retrieval, new training/classes, commits or uploads.

## 2026-10-22 - Codex: transparent focused header and calibration-first camera launch (completed)

* Owner confirms all11outlines/readselectedtarget,requests transparent top and calibration before detection
  for both cameras. Focused renderer now82%source/18%dark overlay with outlined text,retaining native viewport
  and source/OCRpixels. No extraopaque90pxbar; `visual.transparent_header:false` retains previous header.
* `scripts/run_target.py --calibrate-first` launches same isolated config/model/ROI/expectedcount calibration
  before fresh detection. PASS/WARNcontinues;FAIL/cancel/nonzero/missing/invalidcard stops. WARN remains image
  advice,notidentityspeedapproval. Auto sessionname and calibration/outputs avoid oldv3overwriteerrors;
  `cmd_calibration.txt` andsession.jsonretain result. Recorded--source rejects livecalibration before output.
* New `config/box_pipeline_config_webcam_eleven.yaml` adds the same11requestmenu/showall/count/barcodeOFF as
  RSeleven profile,retaining webcammodel/settings. Both updatedcommands/guidance in `docs/REALSENSE_ELEVEN.md`.
  Rawgeometry/identity/source/date/confidence/model rules unchanged; masterinventoryunchanged,no commits/uploads.
* UIproof `results/transparent_calibration_20261022/REVIEW.html`,REPORT.md: native historical before/afterPNG
  forWC11/RSsingle,sourcehashes/blending/viewportschecked. Two bounded20frame actual UIreplays with annotated/
  browservideos,pairedphotos/OCRinputs/CMD; no new live calibration or timingcertificate. Workflow tests check
  bothcameras/configROIcount/evidencerouting/successordering/cancelfailure/missingcard/replayguard.
* Focused75pass; finalfull **561passed in35.32s**. Retained firstfull3failures from old90pxoutline-testoffset;
  testnowcompares identical nativeviewport/headerbackgrounds to retain inside-carton ink/sourcegeometry assertion.
  OwnerPENDING. OPEN: user's live camera positioning/calibration and actual11carton detection/latencyvalidation.

## 2026-10-22 - Codex: faster RealSense OCR and eleven-carton target view (completed; live D455shelf validation open)

* Owner requests further RealSense optimization and11boxversion,barcode/QRomitted. Small native date-token
  region retry in `src/box_target_evidence.py` reduces fresh RS011727recorded textread baseline2.702227s to
  **1.873389/1.963103/1.621588s** in3poweredprocesses. Token geometry only selects pixels; actual OCR supplies
  date at unchanged confidence; both orientations remain one frame/source. Whole-spine fallback retained.
  No VERIFIEDclaim for Sledname/datealone; `visual.date_region_retry:false` retains baseline acquisition.
* `scripts/run_target.py --show-all --expected11` and new
  `config/box_pipeline_config_realsense_eleven.yaml`: D4551280x800@30,existingY26,barcodeOFF,all visible detections
  drawn neutrally,selectedtargetrefined/read,observed/expectedcount never caps predictions. Focused view exports
  `shelf_detections.jsonl`. Eleven dated/sized reference requests numbered from inventorycopy forCMDselection;
  menu requests never assign trackidentity. Default choice targets one box for speed; optional layout question
  had no response,assumption stated. Counts/context are not verified identities or owner-accepted boundaries.
* New eleven-target replay uses existing **webcam-origin1920x1080**shelfvideo,not actualD455shelf footage.
  Separate replay config explicitly preserves source camera.11/11trials45frameseach:7VERIFIEDatend,text+REF,
  zero laterlosses;4SledTEXTREADonly. FirstgreensK04=1.339119,K06=1.349824,K10=1.994209;other4greens2.188-3.598s.
  Sledtextread1.339/1.476/2.909/3.073s. Wrongdate2029-03-11/wrongsizeAngio4x24controls zeroVERIFIED.
  Seventeen total fresh trials retained with source/runtime/model/catalogue hashes and power; no all-box<=2sclaim.
* All13shelf/controlcases11detections45/45each; separate one-to-oneIoU>=.5diagnostic matches all11previous
  assistant-proposed source spines in585/585frames. References remainunreviewed and never enter inference.
  Media checks decode17timedH264videos/689casePNGs. `results/realsense_eleven_20261022/REVIEW.html`,REPORT.md,
  `owner_review/REVIEW.html`(11separatechoices/CSVJSON),allCMD/photos/readerinputs/crops/timings/spatialdiagnostics,
  source snapshots andmanifest. Full **552passed in32.51s**; no threshold/model/classes/inventorychanges.
* Guide `docs/REALSENSE_ELEVEN.md` plus single-camera link: detection/calibrationCMD,menu,reset and limits.
  OwnerPENDING,no physicalretrieval/commits/uploads. OPEN: actualD455fullshelfcalibration/recording,live timings,
  remaining>2sboxes andowner edge/identityreview. Full-field completion remains separate from text/VERIFIED.

## 2026-10-22 - Codex: explicit focused OCR/REF mode without barcode/QR decoding (completed)

* Owner asks to ignore barcode/QR. `scripts/run_target.py --no-barcode` sets session-only
  `verify.decode_barcodes:false` for either camera and records it in metadata/CMD; original profiles untouched.
  `src/box_evidence.py`, `src/box_spine_read.py`, `src/box_target_evidence.py` skip decoders on fast/full paths.
  Export skips barcode location/candidate crops; `src/box_crops.py` hides disabled barcode tile. OCR/REF/GTINtext,
  actual crops/video/photos/CMD retained; no source/date/confidence/identity/outline/model thresholds changed.
* Focused display logs TEXT_READY and shows TEXT READ for repeated correct-date text without an independent
  identity source. Sledname/datealone remains blue/unverified. Printed size+REFstillverifies(test). When no code
  date exists and no confident text date is found, try actual180degreeOCRinput immediately; both orientations
  remain one frame/read/source and actualinputs/tokens are exported. Existing periodic full/detail reader retained.
* Eight recorded fresh-process trials retained(initial/final45frameseach). Final disabled mode textready:
  webcam1.097761s,RS2.611693s(initialRS4.021598s); neither greens. Wrong requested2029-03-11nevertextready/green.
  DefaultcodeONcontrol green1.151795s,barcode+text; initialdefaultcontrolnotverified,retained. SourceRS011727video
  predates horizontal20cmphoto; no new live20cm/25cm or universal<=2sclaim. Source/phase hashes retained.
* Focused82pass; full **547passed in46.20s**. Tests check disabled decoders across fast/full paths, source profile
  isolation, unchanged independent-source rules, no locator/crops, real orientation retry with no read inflation.
  Proof `results/no_barcode_20261022/REVIEW.html`,REPORT.md: all outcomes,actual photos/reader crops/CMD,timedH264
  videos,source snapshots/hashes,media/manifest checks. Updated single-camera guide and directCMD option.
* Masterinventoryunchanged,no commits/uploads/retrieval. OwnerPENDING. OPEN: live OCRtiming on currenthorizontal
  view; RSrecordedtextstill>2s; SledcannotbeVERIFIEDfromname/datealoneunderexistingindependentsourcerule.

## 2026-10-22 - Codex: owner 20cm RealSense calibration review (completed diagnostic; code still unreadable)

* New `realsense_20cm_20261022_v1`02:05:34original1280x800photo hash matches card. One horizontal carton,
  width68.8px,globalsharp958,shift0.87px,scenechange5.5%WARN. Name/date OCR0.99983/0.99990(native regions),
  productionspine0.99972/0.99988; observeddate2029-03-24. Globalsharp755to958 does not prove improved codefocus.
* Thirteen bounded fullphoto/label/code1x2x3x4x/productionspine probes all zero validGS1. NativeDataMatrix
  remains unresolved,smaller than prior16cmphoto. No VERIFIED/<=2sclaim from this still or borrowed code.
* Evidence `results/realsense_20cm_review_20261022/REVIEW.html`,REPORT.md,originalhash/OCR/decodeJSON,CMD,
  native/4xnearestcrops,normalproductionwarp,manifest. Prior16cmevidence unchanged. Directmedia/hash/outcome
  assertions; production/model/options/tests unchanged,prior539passapplicable. No inventorychanges/commits/uploads.
  OPEN:25cmcomparison with nativecodeevaluation,clear second source,and fresh actual green timing; ownerPENDING.

## 2026-10-22 - Codex: RealSense horizontal detail-focus audit (completed diagnostic; physical distance sweep open)

* New owner `realsense_detail_20261022_v3` original1280x800photo/hash matches01:26:37card. Horizontal carton
  improves label size; nameOCR0.96023(nativepatch)/0.98493(productionspine),printed2029-03-24at0.99996.
  Ten bounded decode probes(fullphoto/label/code1x2x3x/productionspine) all zero validGS1; no VERIFIED claim.
  Native code remains blurred. Global central-half sharpness755 does not certify the local code; shift0.55px,
  moving7.4%WARN cannot identify a unique optical/exposure cause from one still. Card's second detection is
  a background bottle-label region,not another real carton. Native crops/4xnearest diagnostic zooms retained.
* Read-only connected D455RGBoptions: no focus control,USB3.2,AE/AWB1,exposure156,gain64,sharpness50 at probe
  time,not photo metadata. No stream opened or settings changed. SDK sandbox Windowsknownfolderlookupfailed;
  outer-context read-only probe succeeded. Manufacturer D400 datasheet3.6.3/table3-19supports fixed RGB focus.
  Do not confuse depthMinZ with RGBfocusdistance or promise an unmeasured minimum/workingdistance.
* Evidence `results/realsense_focus_20261022/REVIEW.html`,REPORT.md,sourcehash/nativeOCR/decode outcomes,
  source crops and device-option transcripts. Assertions check ten failures/no focuscontrol; original/input
  and masterinventory hashes retained. Production/tests/model unchanged; prior suite539pass remains applicable.
  Guide adds stepwise retreat/light/nativecode comparison. Owner subsequently reports lens-label distance16cm
  for this photo(user estimate,not SDKdepthmeasurement). Proposed20/25/30cmcomparisons are experiments,not
  a certified RGBfocusrange; increasing distance also reduces codepixels,so evaluate actual decode.
  OPEN: measured physical distance sweep,new sharper same-box code capture and fresh green<=2s live trial.
  No identity/source/date relaxation,inventory changes,commits or uploads; owner decisionsPENDING.

## 2026-10-22 - Codex: Sled Bag identity acquisition below two seconds in recorded webcam trials (completed; live/RS limits open)

* Owner requests faster green verification with current outline fixes. `src/box_target_evidence.py` now reads
  the complete spine once for NO_SIZE requests (currently Sled Bag) instead of several size-label orientations
  plus tiled spine OCR. Actual name/date tokens and decoded original-spine code feed unchanged verifier gates.
  Every fourth attempt/post-verification detail search uses the full reader; unreadable names fall back.
  Sized products and all confidence/date/source/tracking/outline/model rules remain unchanged.
* Three fresh-process powered webcam replays first rendered green in **1.306422/1.058592/1.135472 seconds**,
  barcode+text, correct Sled2029-03-24, no later loss in45processed frames each. Old acquisition baseline on
  identical recording NOT VERIFIED during its45frames; retained, not omitted. Earlier successful runs2.46-2.48s.
  Wrong-date2029-03-11control zero green. Latest owner RS010340recording selected at0.758s but remained
  CHECKING without readable independent code. Never report that selection time as green verification.
* Six new tests: one-pass observed independent sources, single-source candidate, observed name/code conflict,
  periodic full fallback, post-verification full detail search and low-confidence date rejection. Focused45pass;
  full **539passed in36.51s**. `visual.fast_no_size_spine:false` reproduces prior acquisition path.
* Evidence `results/target_speed_20261022/REVIEW.html`,REPORT.md: all6trials, actual green photos, original
  camera photos, retimed H264 videos, OCR/code crops/tokens, CMD/state/render timing, power and input/runtime/model
  hashes, source snapshots and media checks. Owner PENDING, master inventory hash unchanged; no commits/uploads.
  Updated `docs/SINGLE_BOX_CAMERAS.md` CMD/restart guidance. OPEN: owner's live timing/outline validation,
  sharp same-carton RS code view; no universal<=2s, complete-detail or physical retrieval certification.

## 2026-10-22 - Codex: tighter visible carton boundaries (completed display fix; identity/timing limits retained)

* Owner requests no top/bottom outline overhang. `src/box_target_view.py` now refines and stabilizes selected
  blue CHECKING targets as well as verified ones. Previously D455 pending targets kept the raw detector polygon.
  Source geometry/OCR/identity rules unchanged; pending-target integration test checks continued reading,
  unchanged raw pixels/polygon/history and no identity upgrade while the display corrects.
* `src/box_outline.py`: allow supported inward end correction with bounded70%area floor; former95%floor rejected
  valid shortening. Three rotated regressions fail before/pass after. Side/end continuity,neighbour,frame and
  maximum-expansion guards retained. New fixed-pixel inset keeps the2pxAAstroke3sourcepixels inside estimated
  boundaries,also used by calibration display; raw calibration metrics preserved. Known-boundary180angle probe:
  3pxinset zero leaking angles;2.5px leaked at49. Camera outline references remain unreviewed,not ground truth.
* Same-frame before/after24samples per webcam/RSsource: target found24/24each. Proof full-target crops and3x
  top/bottom patches in `results/outline_edges_20261022/REVIEW.html`. D455before uses previous raw pending
  geometry; webcambefore previous helper. No camera-code training/model bytes changed.
* Four fresh recorded replays plus three follow-ups retain all outcomes. Initial marginal-code webcam updated
  run NOT VERIFIED; separate repeat green2.460241s,barcode+text. Clear-code positive green2.525996s,barcode;
  clear wrong-date2029-03-11zero green. Both RSupdated/wrong-date runs remain unverified because independent
  code absent. Webcamwrong-datealsozero green. Do not hide initial failure or claim consistent/<=2sacquisition.
* Initial100framegeometry runs: webcam97held frames,RS88held frames after acquisition; held means display
  position only. Motion/occlusion/loss/neighbour safety tests still pass. Focused58pass; full **533passed in33.82s**.
  AC sampled connected. All input/runtime/model/catalogue hashes checked; inventories copied,master unchanged.
* Evidence root includes same-frame comparisons,top/bottom zooms,actual-output videos/photos/reader crops,CMD,
  regression and full-suite logs,all replay outcomes,source snapshots/hashes. Guide updated. Owner validation
  PENDING; no retrieval authorization,source/date/confidence relaxation,commits or uploads. OPEN: owner checks
  actual carton boundaries; no perfect/universal boundary claim for unclear/occluded/clipped ends; same-box
  readable independent codes and <=2sidentity remain separate unfinished requirements. Restart running apps.

## 2026-10-22 - Codex: webcam/RealSense single-carton audit and workflow fixes (completed; RealSense code readability open)

* Owner sessions webcam233250 and RealSense234524 end normally. Webcam first displayed green2.477984s,
  barcode+text; RealSense no green. Actual recorded delivery20.059/29.964FPS. Do not mistake camera FPS for
  detector/display speed or green latency. Webcam calibration photo is a different dated carton from its test
  recording; observations remain tied to each original input, never transferred between them.
* Calibration incorrectly required3boxes during a single-carton test. `src/box_calib.py` and
  `scripts/kalibrasi_kamera.py` now accept explicit `--min-boxes 1`, default3preserved, all other image checks
  unchanged. Shared validated `--roi x1,y1,x2,y2` in calibration and `scripts/run_target.py`, saved in provenance
  and forwarded to verified child. Region selects original-coordinate detection centres; no pixel enlargement or
  identity observations. Tests preserve raw pixels/default checks, validate regions and check command forwarding.
* Same-pixel detector diagnostic,16sampled D455 frames: v4target-region hits12/16,30outside-region detections;
  existingY26candidate16/16,zerooutside. Webcam v4outside33vsY26zero. Regions are assistant proposals,not owner
  accepted boundary labels or unseen-scene ground truth. Separate profile
  `config/box_pipeline_config_realsense_single.yaml` uses existingY26weights; generalRSprofile retained.
  Offline original-photo screening with single count: webcamPASS,RSWARN(scenechange5.6%). No new calibration
  capture or full-outline precision approval claimed. No training/class/model-byte changes.
* Four fresh-state AC replays (copied inventories,input/model/catalogue/runtime hashes checked): webcam
  green2.484198s,barcode+text; RSv4 and RSY26no green; wrong-dateRSY26control2029-03-11zero green.
  Native-code crop probe288RSframes produces0validGS1. Connected SDK read-only probe:D455,USB3.2,max
  advertisedcolour1280x800. Agent sandbox SDK context failed Windows known-folder lookup; authorized outer-context
  read-only probe/live test worked. This is not an error in the owner's original normal-exit CMD.
* New bounded liveRSY26test60processed frames,236actual frames/7.847s,29.947FPS,normalexit,one selected track;
  product and consistent printed date2029-03-24read,REF/GTIN/barcode absent,correctlyCHECKING. No retrieval authorized.
  Full suite **525passed in34.15s**,focused29pass. Power at reportAC1/battery88%,not a continuous power trace.
* Evidence `results/dual_camera_fix_20261021/REVIEW.html` and `REPORT.md`: comparison images,5retimedH264videos,
  170decodedcasePNGs,original-camera photos,actual reader crops,CMD,state/timing/SDK/source checks,snapshots/hashes.
  Updated CMD guide `docs/SINGLE_BOX_CAMERAS.md`,linked from camera/calibration docs. Owner decisionsPENDING,
  masterinventoryunchanged,no source/date/confidence relaxation,commits or uploads. OPEN: larger/sharper same-box
  D455name/date/code view,owner boundary validation,and new powered two-source <=2s test. Webcam also still>2s.

## 2026-10-21 - Codex: new Sled detail v1 and decoded-date scheduling fix (verified replay; source/timing acceptance open)

* New owner calibration22:52:46 `sled_detail_20261021_v1`: WARN,1920x1080,median box width169px,sharpness791,
  still0.74px,roll0.1deg. Original hash matches card. Four distinct native DataMatrix codes now decode GTIN08714729896845:
  left1 expiry2029-03-12/LOT39546541; left2 2029-03-10/39525412; left3 2029-03-24/39500350; left4 2029-04-26/39718940.
  Separate printed-date OCR agrees4/4,confidence>0.9999,unchanged0.90threshold. Production-warp crops also decode4/4.
  Ends and upper product names are clipped: not a full-box outline acceptance photo. Old reference data never injected.
* New actual close-view camera recording:201frames/10.035s,MJPG1920x1080,19.930FPS,no recorder drops. Battery72to71%,
  AC=0. Baseline four fresh-process replays: K01/K05 machine green2.457/2.290s; K03/K17 not acquired. Source snapshots
  retained under `results/sled_detail_verify_20261021_v1/batch`.
* Narrow fix `src/box_visual_read.py`: focused date scheduling used only OCR date history, so a supported wrong-date
  barcode sibling could monopolize reads. Include actual decoded barcode expiry in each recent reading's scheduling
  dates; still require two observations before deprioritizing. No tracker observation injection or verifier/date/source
  threshold changes. Semantic regression fails before,passes after;16focused tests pass. Full suite **521passed in32.41s**.
* Same recording after fix, fresh state per target: all4 machine green,zero later losses. First rendered green:
  K01 2.248171s,K03 2.877435s,K05 2.152932s,K17 2.926072s. All on battery/all>2s; no1-2s or powered live claim.
  Wrong absent date2029-03-11 after fix zero green over74processed frames. Machine source is **barcode only** under
  pre-existing GS1 score2 policy; separate date OCR diagnostic is not injected. Strict two independent product-source
  and physical retrieval acceptance remains incomplete because product-name text is clipped. No PICK authorization.
* Evidence report `results/sled_detail_verify_20261021_v1/REPORT.md`; current owner review
  `batch_scheduler_fixed/REVIEW.html`,code/date crops,actual reader inputs,original calibration and new video,CMD,
  retimed H264 videos,state/timing checks,source snapshots/hashes. Owner choices PENDING,inventory unchanged,no commits/uploads.
  OPEN: same-carton complete product name plus independent code (one box horizontally can use wider frame),full ends
  for outline validation,and new powered live latency. No model training or catalogue changes needed for this fix.

## 2026-10-21 - Codex: Sled Bag continuation (diagnostic complete; closer label acquisition required)

* Fresh bounded original-camera sample `results/sled_barcode_ac_20261021/continue_222246`: 61 actual frames over
  about 3 s, MJPG1920x1080, measured20.060FPS, zero recorder drops. Capture and four replay power samples AC=1,
  battery100%; later final evidence check AC=0/battery99%. No continuous power trace. Owner confirms their preview
  also shows eleven boxes: the source views agree, but this is still not a closer label acquisition.
* Four fresh-process dated Sled Bag replays K01/K03/K05/K17 completed, zero VERIFIED (17/21/22/21 processed frames).
  Short-take final exports contain product text, expiry unknown, no REF/GTIN. Do not transfer date observations from
  earlier longer recordings to this take. Bounded original-pixel DataMatrix diagnostic: 63 images (61 video frames
  plus first/last lossless photos),1512 decoder calls,zero valid GS1 observations. No catalogue/reference borrowing.
* Preserved four browser H264 videos retimed to processing timestamps,40 decoded case PNGs,actual OCR/code crops,
  CMD logs,unique owner-review batch id,source snapshots/hashes,original view and 4x diagnostic display. All owner
  decisions PENDING. Review `continue_222246/batch/REVIEW.html`; report `continue_222246/REPORT.md`.
* Full suite **518 passed in33.57s** with fresh workspace --basetemp; one cache-write warning. Initial default-temp
  run had71 fixture setup errors due to Windows access denied outside workspace, retained separately; no production
  code change to bypass tests. Inventory SHA256 unchanged. No model/gate/source/date changes,commits or uploads.
  OPEN: physically bring one same-carton label closer (start2029-03-24); name/date/DataMatrix sharp together,
  whole carton length not required. Then new two-source/date validation and fresh 1-2s latency. No Sled timing pass.

## 2026-10-21 - Codex: ready close-up attempt and DirectShow codec correction (completed fix; close-up still unresolved)

* Owner confirms close-up ready. New bounded capture `results/sled_barcode_ac_20261021/closeup_210417` saved
  227 frames, no decoded codes, AC=1/98%; viewed last original still shows eleven-carton shelf. Investigate actual
  video device identity before asking for another adjustment. DirectShow identifies the external device as Logitech
  Webcam C930e; named-device capture still shows eleven boxes. Runtime camera reports YUY2 despite requested MJPG,
  measured5.007FPS. Reapplying FOURCC after size/FPS in an isolated probe produces MJPG at measured20.131FPS.
  Narrow fix: reapply requested FOURCC after size/FPS for DirectShow in `src/box_camera.py`. Regression fails before
  and passes after. Camera tests14passed; full suite **518 passed in40.46s**. RealSense/MSMF path unchanged.
* Normal `open_camera` verification after the fix: MJPG, 1920x1080, **19.897 measured FPS**, 100 frames read/written,
  zero queue drops, AC=1/battery100%. New short recording still shows all11 cartons. New-format positive replay
  Angiolite4x19 first green **2.369s** (source2.081), text+REF, no later loss over32frames; absent4x24 zero VERIFIED
  over34frames. Do not claim the camera-FPS improvement itself achieves 1–2s VERIFIED or transfer the prior1.706s.
* Owner confirms "Ya, Logitech C930e ini". A 120s interactive preview with new codec received no Enter, so zero
  recording frames; this is explicitly not a close-up acceptance test. Avoid further blind repeated captures.
* Evidence `results/sled_barcode_ac_20261021/camera_audit/REPORT.md`: original/named/latest source photos, new
  camera-throughput recording, timestamp logs, before/after codec properties, driver modes, positive/negative CMD,
  state checks, source snapshots and hashes. `docs/TARGET_CAMERAS.md` updated. No source/date/confidence/inventory
  or model changes, commits or uploads. OPEN: a genuinely closer original-camera label view, then four-date
  two-source validation and new live latency. A selected outline/panel crop alone does not enlarge camera detail.

## 2026-10-21 - Codex: Sled Bag second-source recovery and AC recheck (diagnostics complete; actual close-up pending)

* Owner requests improving four dated Sled Bags and checking charging. Windows initially AC=1, charging flag 8,
  54%; later sampled 81% on AC. Six fresh-process AC replays of the same calibrated recording: four Sled Bags
  K01/K03/K05/K17 still NOT VERIFIED; K10 Angiolite4x19 **1.706 s** (both clock bounds <=2), K15 2.5x29 **3.437 s**.
  Both positives text+REF, no later losses. Battery baseline was 4.410/13.728 s. These are recorded rendered-frame
  times excluding startup, not live monitor/universal timing. Charger does not supply a missing barcode source.
* Original-pixel DataMatrix diagnosis bypasses perspective resampling: three lossless photos + all127 video frames,
  **130 images, 34,980 bounded decode calls**, local candidate crops and non-generative scaling/binarization:
  **zero valid GS1 observations**. No catalogue-derived values. No extra retries enabled in production without
  demonstrated recovery. Older clearer reference image decodes GTIN08714729896845 / expiry2029-03-24 / LOT39500350;
  this is a separate decoder control and never supplies evidence for a current webcam box or another date.
* Owner replies "Bisa, saya atur sekarang" to closer-view request. A subsequent bounded interactive camera capture
  saved **227 raw frames / ~45s, 5.014FPS, no drops, zero code observations, AC=1 throughout samples**. Viewed last
  original photo still shows all11 boxes at shelf distance: not a valid close-up acceptance test. Readiness question
  pending; do not treat the offer to adjust as proof that adjustment is complete or repeatedly capture blindly.
* Evidence `results/sled_barcode_ac_20261021/REPORT.md`, `ac_batch/REVIEW.html` (six AC cases with CMD, original/
  retimed H264 videos, photos and actual OCR crops), probe/native crops, reference decoder control, power records,
  attempted-close-up raw video/timing/first+last photos, checks and manifest. Six videos /352 case PNGs decode;
  static review links and pinned input hashes checked. Every owner decision PENDING. Full suite **517 passed in45.57s**.
* No production/model/threshold/inventory/catalogue changes, commits or uploads. Master inventory unchanged at
  SHA256 `01253e7d08a3f967a4dbd4465e31070ca6ac7fc1cf125f2de586a36f94edc3a3`.
  OPEN: actual closer same-box name/date/DataMatrix view and two-source verification, then each of four dates.

## 2026-10-21 - Codex: eleven-box detection after calibration v2 (completed; owner/live AC acceptance pending)

* Owner requests testing the eleven boxes now in calibration. New live webcam recording with the unchanged
  all_targets config: **127 actual frames / ~25 s, 1920x1080, measured 5.016 FPS** (20 requested), on battery.
  Initial/final detected shelf position differs from v2 by **0.2 px**, median widths 66.8/67.0 px. Raw frame/timing
  logs preserved. Device identity is still not proved by index; no synthetic/repeated stills used as live evidence.
* Fresh process/tracker per inventory request, 11/11 finished, **7 VERIFIED at end, zero later losses**: K04, K06,
  K07, K10, K15, K16, K21, all text+REF. **4 not VERIFIED**: K01/K03/K05/K17, each Sled Bag's correct printed date
  read, but no independent REF/barcode. No field/inventory borrowing or confidence/source/date relaxation.
  All trials on AC=0: recorded first greens 3.060–13.728 s are diagnostic, cannot certify the 1–2 s goal. Full
  field completion remains false. Three 100-frame absent-size/wrong-date controls: **zero false VERIFIED**.
* Readability bottleneck beyond battery: K15's initial actual `2.5` token confidence **0.83975 < 0.90**; REF and
  length readable, diameter text rejected until a later observation. Do not imply charger alone solves the delay.
  Sampled hard GS1 read of 33 actual spatial crops (first/middle/last frames): zero codes, diagnostic only.
* Evidence: `results/calibrated_eleven_20261021/REPORT.md`, `batch/REVIEW.html` (11 owner cards/export), new raw
  recording/retimed H.264, per-case CMD/photos/actual OCR crops, three negative videos, framing measurements,
  nullable fields, source snapshots/hashes, `checks.json`, `LATENCY_DIAGNOSIS.md`, source provenance and manifest.
  The raw frames are new, not trained on; physical scene/products previously seen, not an unseen-scene accuracy test.
  Existing runner has a generic overlap note in inputs.json; SOURCE_PROVENANCE.json records the actual situation.
* All 11 videos and **669 case images** decode; retimed capture/controls decode; static review links pass. Spatial
  comparisons show no proposal warning, but the assistant map is unreviewed and never enters inference. All owner
  decisions PENDING. **517 passed in 33.03 s**. Inventory SHA256 unchanged
  `01253e7d08a3f967a4dbd4465e31070ca6ac7fc1cf125f2de586a36f94edc3a3`. No production code changes, commits or uploads.
* OPEN: owner validates full edges/identities; four dated Sled boxes need closer readable independent codes; improve
  small-label focus/lighting, particularly Angiolite 2.5, and repeat live timing on AC. No universal/all-field 2 s claim.

## 2026-10-21 - Codex: transparent live calibration panel (completed)

* Owner requests transparency for the live calibration status panel. `scripts/kalibrasi_kamera.py` now overlays
  the live panel with 18% dark background / 82% source image, with outlined text. Live viewport retains the camera
  dimensions. Saved calibration photos retain their separate header; raw pixels/measurement/identity gates unchanged.
* Offline preview and pixel checks: `results/calibration_panel_20261021/transparent_preview.png`,
  `preview_checks.json`. Source frame unchanged, full-frame dimensions retained, background blending checked.
  Full suite **517 passed in 42.00 s** (`pytest_cmd.txt`). No new live camera session, commits or uploads.
  An already-running calibration process must be closed and restarted to load the updated drawing code.

## 2026-10-21 - Codex: calibration framing and lower-outline audit (completed; fresh live calibration pending)

* Owner reports uneven lower outlines and a large apparent framing difference. Reproduced a display-only bug:
  padded-ROI false end outside the image rejected the whole polygon, cancelling a valid opposite-end correction.
  `src/box_outline.py` now validates each proposed end independently. Two orientation regressions fail before and
  pass after the fix. Existing neighbour, occlusion, blank/texture, motion and clipping guards remain.
* Calibration previously used a full-resolution unbounded preview, overlaid its header on carton tops, and drew raw
  polygons. `src/box_preview.py` now shares complete-image fitting with focused view (`box_target_view.py` fit call
  only); calibration puts its header above the image and draws blue supported-edge display polygons. Raw detector
  geometry still supplies measurements/OCR. Different detail layouts can still make on-screen object size differ.
* `src/box_calib.py` warns on top/bottom margins <3% and withholds scale when populations or frame sizes differ.
  `scripts/kalibrasi_kamera.py` saves raw PNG/hash, detector/display polygons, config/model settings and configured
  camera plus actual resolution. Same-position checks warn when prior camera provenance is missing or settings
  differ; index is explicitly not physical-device identity. Driver resolution mismatches warn; existing evidence
  cannot be overwritten. Central-image sharpness is explicitly not per-label detail/readability certification.
* Existing 17:36 calibration: 11 cartons, median width 46.6 px; later original 17:45 camera photo: 11, 57.3 px
  at the same 1920x1080. Raw framing therefore changed as well (~23% pixel width), not just display size. The old
  card omitted provenance/raw pixels: no definitive camera-switch, physical-distance or driver-FOV explanation.
  Former PASS lacked explicit vertical-margin checks. New offline screening warns on the live photo's upper edge.
* `results/calibration_outline_20261021/REVIEW.html`: original/before/after images, 4x bottom pixel comparison,
  preserved source hashes, CMD, photo/crop and retimed H.264 replay videos. Three sequential 56-frame replays:
  Angiolite 2.5x29 first rendered green **1.821 s**, 4x19 **1.403 s**, both text+REF, no later losses; absent 4x24
  zero VERIFIED. Startup excluded, rendered-frame/replay timing, not new live or all-box certification. Full field
  completion false. AC=1 sampled during the first replay; no continuous power trace. Originals/media/links checked.
* Final suite **517 passed in 30.97 s**. Inventory SHA256 unchanged
  `01253e7d08a3f967a4dbd4465e31070ca6ac7fc1cf125f2de586a36f94edc3a3`. No live camera opened, commits or uploads.
  Guide: `docs/CALIBRATION_FRAMING.md`. OPEN: owner boundary approval (no new ground truth), new live calibration
  with full upper/lower visibility, physical-device/lens/robot calibration, and broader live/held-out box verification.

## 2026-10-21 - Codex: unresolved-box continuation, stable Essential Pro and separate photo evidence

* Owner requests continuing the unverified/later-lost cases under the original requirements. Narrow numeric lexer
  fix in `src/box_verifier.py`: invalid LOT/date slash fragments `20/0L92` and `20/0192` no longer supply a length.
  Literal slash sizes (`3/24`, `3/200`), genuine competing lengths and original raw OCR text are preserved. No
  confidence/source/date threshold reduction. `tests/test_size_code_fragments.py` covers no invented second source.
  Preserved Claude's Terumo REF correction. No inventory/catalogue verification changes, commits or uploads.
* A first partial follow-up found the numeric spelling and stopped via the source-hash guard after K09 when the
  correction was applied. Retained `results/all_boxes_followup_20261021` as explicitly incomplete diagnostic evidence.
  Separate corrected Essential Pro 160-frame replay: first green **1.296 s**, **zero losses**, AC=1.
* Final corrected batch `results/all_boxes_corrected_20261021`: **20/20 on AC=1**, fresh process/tracker, 80 frames
  each. **7/20 VERIFIED at end, 0 losses; 4 first-green frames <=2 s in both wall and capture bounds; 3 over 2 s;
  13 never VERIFIED.** K04 **1.249 s**, K06 **0.987 s**, K10 **1.929 s**, K15 **0.968 s**. Accuforce K07/K16/K21
  **3.180/3.546/3.509 s**. K04's later loss is fixed; K21 recovers with the Terumo rule. Do not claim all-box <=2 s.
* Nine absent-size/wrong-date video controls: **0 false VERIFIED**; two original Angiolite regression clips also
  match. All main green events have actual text + REF; no retrieval commands. Runtime/config/model/media and now
  GTIN catalogue hashes checked before/after trials; frozen source snapshots retained. Source-at-green and latest
  observed sources are explicitly separate in the review/action CSV.
* Further diagnosis: 20 evenly spaced frames across the full recording, 20 actual detector crops each: **400 crops,
  0 decoded GS1, 0 direct raw barcode decodes**. This is sampled, not exhaustive. Existing reference-photo decoder:
  **5/13** expected codes decoded. Contrast variants did not recover the Nagomi name from the sampled saved inputs;
  the existing fast region layout did not fit the 3 Accuforce crops and was not enabled for them.
* Tested every available relevant reference photo separately with the real detector/reader/verifier, isolated outputs
  and inventory copies: **7/10 positive photo requests matched** (K05, K08, K09, K12, K14, K17, K20); K11/K13/K18 did
  not match. Three photo negatives (Sled Bag same GTIN but wrong dates; Agent wrong size): **0 MATCH**. The legacy
  photo tool's PICK label is an offline photo decision, not a robot command. Never merge these photo observations
  into the current webcam fields/status/timing; main 13 unverified cases remain unverified there.
* Evidence: `REVIEW.html`, `PHOTO_REVIEW.html`, `REPORT.md`, `NEEDS_CLOSER_VIEW.md`, `validation_actions.csv`, raw CMD,
  original/retimed/browser videos, photos/crops, nullable fields, gate checks and `MANIFEST.json` in the corrected
  batch folder. Every owner decision stays PENDING. All review links/snapshots checked; media decoded; DOM review/
  export simulation passed. Full suite **507 passed in 30.60 s**. Guide: `docs/ALL_BOXES_REVIEW.md`.
* Master inventory SHA256 unchanged: `01253e7d08a3f967a4dbd4465e31070ca6ac7fc1cf125f2de586a36f94edc3a3`.
* OPEN: owner reviews physical targets and full outlines, including the unreviewed K12 slot warning. Thirteen webcam
  cases need sharper co-located identity/code evidence; the two other Sled dates need their actual dated boxes, not
  the existing March-24/March-12 photo. Scoreflex has no matching verified GTIN reference and needs the owner's
  physical-code check before catalogue acceptance. Three Accuforce sizes still need validated faster acquisition;
  forcing the existing iVascular band layout did not fit. Complete all-field reading, fresh live/held-out tests and
  RealSense remain open. Existing recordings/reference data overlap training; no universal accuracy/latency claim.

## 2026-10-21 - Claude Code: validation of Codex's final all-boxes batch, Terumo REF fix (completed)

* Owner asked to check Codex's latest results. Report: `results/codex_validation_20261021/README.md`.
* `results/all_boxes_final_20261020` matches its logs (6 / 20 VERIFIED at least once, every green text + REF, inventory
  unchanged). All verified boxes ran on battery: readings to first MATCH equal the charger rerun, so the slower seconds
  are power, not the all_targets profile. CSV column `sources` = last observed, not sources at green (K06 "REF").
* K21 Accuforce 3.5 x 15 never verified: OCR read the REF "26 RM3515HHW" without "DC-". Fixed in `src/box_ref.py`:
  a Terumo code without its prefix counts only with the full suffix (HHW / ASM). In 5,100 recorded readings every such
  read is the real box size; re-parse of Codex's final K21 texts gains REF on 4 readings, no wrong-size REF anywhere.
* K04 lost green at 20 s because one reading took length 20 from "20/0L92" (date / LOT fragment): correct release, kept.
* Rerun K07 / K16 / K21 verified with 0 losses; 9 negative replays (mixed Accuforce sizes, absent Ryurei, Codex's
  4 controls): 0 VERIFIED. Tests: 490 passed.

## 2026-10-20 - Codex: 20 shelf-box evidence packages ready for owner validation

* Tested all 20 inventory requests separately from the existing full-shelf recording, fresh process/tracker per box,
  80 processed frames per trial. Each dated Sled Bag stays separate and must supply its actual requested printed date.
  Two-source acquisition/date gates, inventory and detector classes are preserved; no retrieval command or commits.
* New `scripts/validate_box_targets.py` sequential runner, runtime snapshots and input/model/source hash checks before
  and after each trial. Shared-tree drift stops with a written reason; completed and partial evidence are retained.
  New `scripts/box_target_review.py` and `scripts/inspect_box_target_evidence.py` provide a local 20-card owner review,
  full retimed H.264 videos, original videos/timing traces, CMD, final/first-green photos, actual OCR crops, nullable
  observed fields, and separate CSV/JSON owner decisions. Every decision remains PENDING; no inventory writes.
* Candidate `config/box_pipeline_config_all_targets.yaml` / `FOCUS_ALL_BOXES.bat`: printed-region reads for Angiolite,
  Essential Pro and Xperience Pro; actual high-confidence printed mm/unit geometry excludes unpaired date digits;
  two consistent other dates can lower a sibling's read priority without changing matching. Raw and selected tokens
  are saved. Other families retain the full reader, and the precise B544 detector is unchanged.
* Preserved the baseline `results/all_boxes_review_20261020` and intermediate `results/all_boxes_validation_20261020`.
  Claude's later hold correction arrived after all intermediate trials/controls; its original source was reconstructed
  to the exact initially pinned SHA and preserved, with a version note. Concurrent validation workloads also affected
  intermediate timing. Do not merge batches or select only the fastest case to claim an all-box deadline.
* Final current-version batch includes Claude's split-decimal and hold fixes, with runtime snapshots unchanged:
  `results/all_boxes_final_20261020/REVIEW.html`, `REPORT.md`, `validation_summary.json`, `validation_actions.csv` and
  `MANIFEST.json`. **20/20 completed; 6 VERIFIED at least once; 5 still VERIFIED at end; K04 subsequently CHECKING;
  14 never VERIFIED.** Windows switched to battery during K03; the owner explicitly requested continuing detection
  proof. Only K01 was AC-eligible at both ends and it never verified. Main-batch times cannot certify the 1-2 s goal.
* Four absent-size/wrong-date controls: **0 false VERIFIED**. After AC returned, two original Angiolite regression
  clips reached first green in **1.512 / 1.305 s**, actual capture timestamps **1.186 / 1.209 s**, zero later losses.
  These are separate 40-frame regression takes, not the full-shelf 20-case result or unseen/live acceptance.
* Validation: **478 passed in 30.50 s**; 20 original/retimed/browser videos and all photos/crop PNGs decoded; all review
  links and runtime snapshots checked. Owner review/export passed a DOM simulation (not actual owner approval).
  Launcher prepare-only/ASCII/CRLF smoke passed. Master inventory SHA256 remains
  `01253e7d08a3f967a4dbd4465e31070ca6ac7fc1cf125f2de586a36f94edc3a3`.
* OPEN: owner reviews every physical target/full boundary and sources, exports Accept/Reject/Need closer image.
  K04 has contradictory OCR length; K13/K21 remain intermittent in this pose; missing independent sources need
  readable REF/barcode/size views. The assistant-proposed map/spines (especially the K12 candidate warning) remain
  unreviewed references and never enter inference or inventory. All supplied recordings overlap training; all-field
  completion, held-out/live timing for every family and RealSense testing remain open. `docs/ALL_BOXES_REVIEW.md`.

## 2026-10-20 (evening) - Claude Code: validation of Codex's latency / all-boxes results, two fixes (completed)

* Owner asked to check Codex's results and fix what is needed. Report: `results/codex_validation_20261020/README.md`.
* Codex's delivery valid in structure: suite passed, inventory unchanged, REF alone cannot turn green. The 11 boxes not
  verified match the second-source audit (physical limit). Three anomalies investigated:
  * K10 Angiolite 4 x 19 (7.9 s): `_merge_split_decimals` turned "4" "19" into diameter 4.19. Fixed in
    `src/box_verifier.py` (merge only .00 / .25 / .50 / .75). First MATCH read #20 -> #3; with both fixes 1.79 s.
  * K21 Accuforce 3.5 x 15 green -> checking 3x: the hold in `src/box_target_view.py` (Codex's file) was released when
    detail reads pushed REF out of the combined window. New `keeps_verified`: released only by evidence against it,
    a stale / lost / new-epoch track or a required date no longer held. Rerun: 0 losses on all 8 verifiable boxes.
  * K13 / K18 Nagomi: name and REF read only now and then from this pose - OCR limit, not fixed.
* Negative replays (absent 4 x 24, Accuforce 3 x 15, Angiolite 4 x 29): 0 VERIFIED. Tests: 476 passed.
* For Codex: `src/box_target_view.py` changed in two places (`keeps_verified` + the release condition); the
  all-boxes batch can be rerun as before (`scripts/validate_box_targets.py`).

## 2026-10-20 - Codex: VERIFIED latency continuation (recorded checks complete; fresh live test pending)

* Owner requests 1-2 s with the stable outline. Preserved model B at **544 x 960**, original webcam/RealSense profiles,
  Claude's parallel-read implementation, two-source/date gates and inventory. No training, staging or commits.
* Stable profile enables a joint OCR pass over adjacent actual name/REF and size regions. A real partial REF may
  return early during acquisition; it supplies only that source, never request-derived product/size values. Repeated
  REF alone cannot turn green. After initial verification, detail reads disable that partial exit to maintain full
  fresh observations in the unchanged rolling evidence gate. An intermediate batch's three later CHECKING transitions
  were found and corrected; the full longer final batch has zero later losses, with genuine conflicts still gated.
* Edge refinement is lazy during acquisition/reacquisition; the supported hold follows actual pixel motion. Pending
  blue outlines use current detections. Bounded background workers freeze crop-export snapshots and encode video;
  request generations prevent old exports appearing after a repeated request/O reset. Video's real clock continues
  across request-timer resets. Annotated video is 1280 px wide; original photos/raw camera and reader crops retain size.
* Final **17 sequential AC=1 replays**, fresh observations per take, 40 processed frames or clip EOF: all **16 positives
  verified with text + REF and first green frames ready within 2 s**. 4x19 green **1.456-1.839 s** (8/8); 2.5x29
  **1.185-1.884 s** (8/8). Absent 4x24: zero false VERIFIED. No later verification loss in the positive takes.
  Startup excluded; no-window rendered-frame timing is distinct from monitor presentation. First export notification
  is **1.442-2.241 s** overall: green does not imply all files or all eight observed fields complete within 2 s.
* 902-frame cached-detection display regression preserves both targets, median approximate-reference IoU 0.930/0.917,
  p95 step 0 px, 99.45% stationary transitions, only 3 refinement callbacks per target. This isolates display changes;
  neighbours/motion/loss have unit coverage. References are assistant proposals pending owner review and recordings
  overlap training. A 448x800 export was diagnostic only; it is not selected or enabled.
* Evidence: `results/latency_stable_20261020/REPORT.md`, `LATENCY_EVIDENCE_20261020.mp4`, `final544.json`, per-take
  CMD/videos/photos/reader crops/nullable fields, `frame_latency.jsonl`, annotated video timing CSVs, source hashes
  and manifests. Retimed montage uses actual frame-ready times, not the MP4 header FPS.
* Full suite **460 passed in 29.26 s**; launcher prepare-only/ASCII/CRLF/profile smoke passed. All 17 videos and photos/
  crop PNGs decoded; retimed montage 120 frames. Master inventory SHA256 unchanged:
  `01253e7d08a3f967a4dbd4465e31070ca6ac7fc1cf125f2de586a36f94edc3a3`. Claude's verifier/pipeline files unchanged.
* OPEN: owner live test using `.\FOCUS_WEBCAM_STABLE.bat` with charger, fresh camera placement/unseen data, complete
  per-field barcode/GTIN/LOT/expiry/brand verification, other products and RealSense live timing. These recording
  checks establish the measured batch result, not a universal <=2 s deadline. Guide: `docs/STABLE_WEBCAM.md`.

## 2026-10-20 - Claude Code: continues Codex's outline-stability / recording-training topic (completed; owner live test pending)

* Owner (20-10): the 1-2 s target is met for some boxes only (4 x 19 still 2.4-2.8 s); box outlines still wobble a
  little; wants them perfectly still and precise, trained on as many recordings as possible, any waiting time accepted,
  then the owner tests other boxes. Owner explicitly approves server use (own folder `exp_outline_max_20261020`).
* Claim: training data from ALL webcam / D455 recordings (static-scene temporal-consensus labels, verified per box),
  long server training, held-out evaluation (jitter + tightness on every box), and the <= 2 s latency of 4 x 19.
  Codex's `src/box_outline_track.py` (display hold) and `scripts/eval_outline_stability.py` stay Codex's; I only read them.
  Codex's 16-frame training package (`results/outline_stability_20261020/training_bundle`) is superseded, not deleted.
* Progress (20-10 afternoon):
  * Recordings -> labels: `scripts/static_segments.py` (static-pose segments), `scripts/auto_label_static.py` (7-model
    consensus per pose + top / bottom ends read by Claude on zoomed views; the other poses by feature-homography
    transfer), `scripts/build_static_dataset.py` (2,242 frames, 148,527 labels from 9 poses of 17 / 18 / 20 Oct + the
    D455 clip; a frame is kept only if the shelf is unchanged). All labels are model / Claude proposals, owner check pending.
  * Server `~/Downloads/exp_outline_max_20261020`: queue `A` (no 18-Oct recordings = honest test on the unseen day) then
    `B` (everything), YOLO26s-OBB final as base, 100 epochs, imgsz 1280, +196 reviewed photos x2. About 3.3 h each.
    `scripts/eval_static_models.py` measures IoU / end error / wobble (temporal std of the corners) on held-out poses;
    baseline before training (18 Oct): wobble median 1.0-1.8 px (p90 3-6 px), IoU >= 0.90 for only 15-64% of boxes.
  * Latency: the locked target may now have one OCR read per worker in flight (`visual.parallel_target_reads`, default
    on; `TrackState.inflight`, `tests/test_target_parallel_reads.py`). Same rules and readings, run side by side.
    Replay of the 16 raw takes of 20-10 (`results/outline_max_20261020/parallel_reads/`): 4 x 19 median 2.19 -> 1.31 s,
    within 2 s 3 of 8 -> 6 of 8; 2.5 x 29 median 1.59 -> 0.72 s, 5 -> 6 of 8. The two 4 x 19 takes that stay > 2 s are
    limited by OCR quality (diameter / REF not read in the raw 5 fps frames). A shelf position map made the first lock
    correct but did not change the time in the one pair tested.
  * Training done (A 14:21, B 17:51). Held-out day (model A, 18 Oct): IoU >= 0.90 32-57% -> 60-79%, end error 10-15 -> 4-6 px,
    raw wobble 0.9-1.7 -> 0.4-1.0 px. Model B (all recordings): IoU >= 0.90 on 95-100%, 0 false boxes on every pose incl. D455.
    Replay of the 16 raw takes: B verifies 16 / 16 with text + REF, 4 x 19 all within 2 s (max 1.79 s), 2.5 x 29 all <= 0.81 s
    (those takes were part of the training frames: not an independent timing test). Webcam profile now uses
    `models/box_y26_maxB_544x960_openvino_model` (go back: commented line in config/box_pipeline_config_focus.yaml).
    Evidence: `results/outline_max_20261020/README.md`. Tests: 444 passed.

## 2026-10-20 - Codex: stable target outline and recorded-model test package (completed local work)

* Owner requests a precise stationary target outline, extended recording/training and evidence before testing other
  products. Claim display tracking, motion-aware outline stabilization, recorded A/B evaluation and an isolated
  training candidate. Preserve original model/data and mixed edits; no commits. Do not train and evaluate on adjacent
  frames from the same clip as if they were independent. Automatic labels remain explicitly unreviewed.
* Keep detector/OCR geometry and two-source/date verification independent from display stabilization. A static
  outline must release on real motion, target loss, occlusion, request/track/epoch changes. Training is evaluated
  against the current model before activation; no claim of perfect unseen performance.
* Continuation check: Claude's newer 2,242-frame / 100-epoch A/B experiment supersedes the 16-frame/60-epoch package;
  do not start an additional GPU training job. A is complete and B is still running (~91/100 at inspection).
  Continue Codex-owned tracking validation, persistent candidate/evidence packaging and independent replay checks
  under `results/outline_continuation_20261020/`; retain Claude's source files and experiment inputs.
* Broader A/B training both finished 100 epochs, exit 0. Preserved both completed checkpoints and A exports locally;
  exported B to fixed OpenVINO [1,3,544,960]. B checkpoint SHA256 matches server:
  `4e3fe5bd3872c1b4567f008330a4f31e75c4cc4eab0eb2777b82bdfa68fe5c4a`. Split audit independently confirms A excludes
  all 544 18-Oct recording frames; B includes them. 2,242 recording frames / 148,527 proposal labels, 196 base photos
  repeated twice. Keep owner-review-pending provenance; B's recording fit is not held-out generalisation.
* `OutlineTracker` holds an initial consensus using direct anchor-to-current optical flow, follows real movement,
  and resets on loss/uncertainty/request/track/epoch changes. Current detector observations guard association when
  an optional display-edge refinement jumps to print. Current raw detector crops still feed OCR; display holding
  changes no identity support, threshold, date policy or physical selection state. Per-frame raw/display polygons
  and mode saved in `outline_frames.jsonl`. Added movement, occlusion, sibling-jump and refinement-jump tests.
* Same 902-frame recording, both targets found in all frames: B stable median reference IoU **0.930 / 0.917** for
  4x19 / 2.5x29; previous model's per-frame refined median **0.848 / 0.846**. Median and p95 inter-frame displacement
  **0 px** for both, **99.45%** exactly stationary transitions. Reference is Codex visual estimate +/-3 px, owner
  review pending; recording contributes to training. Initial acquisition / measured motion still allow updates.
* Sequential A/B functional replay: **14 runs**, each model verifies all 6 positive takes using text + REF; absent
  4x24 control remains unverified for both. All runs AC=0: battery wall/source times are logged, excluded from the
  1-2 s acceptance comparison. One B take reads printed 2.5 as 5 and correctly waits for consistency; no thresholds
  reduced to recover speed. Owner explicitly says continue bounding-box checks first. No new live-camera claim.
* Selected B as a separate user-test candidate for its improved outline fit and valid identity controls. New
  `FOCUS_WEBCAM_STABLE.bat`, `config/box_pipeline_config_focus_stable.yaml`, `docs/STABLE_WEBCAM.md`; runner now
  supports `--config` and rejects cross-camera profiles before creating a session. Original profiles unchanged;
  RealSense still uses its existing profile. Training on all brands does not certify other brands' verification speed.
* Evidence: `results/outline_continuation_20261020/REPORT.md`, full-length `BOX_STABILITY_20261020.mp4`, comparison
  photos, per-take videos/CMD/photos/crops, training recipes/results, split/export/selection/artifact/implementation
  manifests. Full suite **444 passed in 41.68 s**, `pytest_full.log`. Stable launcher prepare/menu-exit smoke both
  passed, ASCII/CRLF. Master inventory SHA256 unchanged. No staging/commits; mixed Claude work preserved.
* OPEN: owner live test and review of proposal outlines, AC-powered timing of the new B model, complete per-field
  verification, fresh camera poses/other product requests, and RealSense live validation. No universal perfection
  or <=2 s guarantee from the recording checks.

## 2026-10-20 - Codex: common webcam/RealSense CMD workflow (local work completed; deadline still open)

* Owner requests one comparable CMD workflow for ordinary webcam and RealSense, same recorded/crop outputs, and
  1-2 s VERIFIED acquisition. Claim launcher/profile integration and focused evidence/scheduling improvements,
  isolated replay/live validation and docs. Preserve mixed changes; no commits or inventory updates.
* Keep independent-source/date gates unchanged. Measure actual VERIFIED latency; do not promise an unsupported
  universal deadline. Webcam-specific model must not silently replace the RealSense model.
* Added `scripts/run_target.py`, `TEST_TARGET.bat`, and camera-specific FOCUS wrappers. One menu/direct CLI provides
  preflight, live/replay, actual CMD transcripts, raw/annotated recordings, paired photos, numeric/label crops and
  nullable observed fields. Each take has its own session config and inventory copy. Webcam uses explicit #1 DSHOW
  1920x1080 and its existing YOLO26 fine-tune; RealSense retains SDK capture and v4 OpenVINO. No stress menu in this flow.
* Focused Angiolite reading tries narrow actual product/REF and size bands, then detected-label geometry and the
  existing full reader. Geometry groups actual numerals/units and excludes tiny date fragments; values/confidence
  are never filled from the request. Prioritize a supported track; recognition warmup is discarded before timing.
  Pending crop export waits for verification/the deadline. UI/JSON explicitly report within/over target and cached
  subsequent requests. Independent-source/date gates, full reader fallback and physical retrieval policy unchanged.
* Final stable-code sequential live webcam batch: 4x19 **2.496 / 2.759 s**, 2.5x29 **1.408 / 1.726 s**, text + REF.
  Corresponding first crop exports: 2.597 / 2.854 / 1.490 / 1.815 s. Absent 4x24 remained unverified in its 40-frame
  control. AC=1 before/after all runs. Startup excluded; fresh observations, not cached-track success. This small
  batch does not establish a universal <=2 s deadline; 4x19 still misses it. Earlier diagnostic geometry-first
  trials ranged 1.877-6.516 s; final code uses narrow bands first. Mixed-code/concurrent adapt diagnostics excluded.
* Owner explicitly deferred RealSense live testing. Existing D455 recording replay with its own profile: VERIFIED
  **4.708 wall seconds**, not a live-camera result. Do not claim either camera universally meets 1-2 s.
* Evidence/report: `results/speed_dual_20261020/REPORT.md`, `metrics.csv`, `accepted_results.json`, `manifest.json`,
  `artifact_checks.json`, per-take videos/CMD/photos/crops. All 11 resulting MP4 files decode, 48 photos readable;
  inventory SHA256 unchanged (`01253e7d08a3f967a4dbd4465e31070ca6ac7fc1cf125f2de586a36f94edc3a3`).
  Launcher menu/exit smoke passed for all 3 BATs (ASCII/CRLF). Full suite **433 passed in 32.44 s**, `pytest_full.log`.
  No staging/commits; mixed Claude files and original evidence preserved. Guide: `docs/TARGET_CAMERAS.md`.
* OPEN: consistent <=2 s for 4x19 and other layouts; RealSense live validation; complete independent per-field
  barcode/GTIN/LOT/expiry/brand verification. Both ends must be in frame for full boundary measurement; some current
  Angiolite tops remain clipped. Green means verified identity, not completion of all eight fields or pick authority.

## 2026-10-20 - Codex: model test recording and daily report (completed)

* Owner authorizes model testing and recording for today's report. Claim isolated session/preflight, live capture
  if the webcam shows the intended shelf, and a reviewable local report/evidence package. Preserve mixed work,
  inventory and historical evidence; no commits. Distinguish fresh live capture from replay and still-photo tests.
* Initial Windows power check AC=1. Keep two-source/date requirements; report missing fields and unmet latency.
* Owner confirmed the external webcam was connected and facing the shelf. Initial sandbox preflight could not read
  it; camera #0 outside sandbox was dark 640 x 480. After connection, a bounded explicit check found camera #1 /
  DSHOW at 1920 x 1080 showing the real shelf. An owned stalled automatic 4K-enumeration probe was stopped; no other
  application processes were touched. Fresh live captures used explicit camera #1 to avoid the stall.
* Two new 100-processed-frame live takes, raw/video/CMD/photos/crops: 4 x 19 identity VERIFIED **2.406 s**; 2.5 x 29
  **3.114 s**, text + REF. AC=1 at both start/end of both takes. One OCR worker due to limited RAM (initial free
  ~0.3-0.4 GB; live starts ~655/1221 MB). No universal 1-2 s claim. 2.5 x 29 returned to CHECKING at 16.663 s after
  single-source readings, then VERIFIED at 17.515 s. Full clips retain that transition; no stale event was logged.
  Some target ends were initially outside frame; do not describe these as pixel-perfect full box outlines.
* Fresh raw captures: 103 frames / 20.3529 real seconds, 107 / 21.134 s (actual timing CSV). Nominal MP4 FPS=20,
  annotated FPS=6; playback duration is not capture latency. Green is identity verification; GTIN/barcode, LOT,
  expiry, brand and some direct size-text fields remain incomplete. No physical retrieval/inventory mutation.
* Deliverables in `results/daily_report_20261020_083114/`: `report_package/REPORT_20261020.html`, three-page PDF,
  Markdown, 47.33-second H.264 `MODEL_TEST_REPORT_20261020.mp4`, metrics CSV/JSON, CMD logs, paired photos, raw/annotated
  live videos, actual crop inputs, validation and SHA-256 manifest. ZIP `MODEL_TEST_REPORT_20261020.zip` is 35.5 MB,
  62 files, CRC clean. Separate Drive still-photo results are clearly labelled; earlier new replay 4 x 19 5.81 s
  is diagnostic and is not used as a live timing result. Original source/model/inventory files not bundled.
* Validation: **415 passed in 33.22 s**, `pytest_full.log`. All 284 report-video frames, 103/107 raw frames and
  both 100-frame annotated clips decode; all live photo pairs readable; PDF header/3 pages/EOF checked. Master
  inventory hash unchanged (`01253e7d08a3f967a4dbd4465e31070ca6ac7fc1cf125f2de586a36f94edc3a3`). No staging/commits.
* OPEN: consistent 1-2 s acquisition, complete per-field verification, better framing/low-confidence numeric OCR,
  and robust automatic camera selection (global auto/MSMF profile deliberately not overwritten by today's index).

## 2026-10-20 - Codex: existing-media tests and public Drive contents (completed)

* Owner asks to test existing recordings/photos and check the two shared Drive folders and their contents.
* Claim read-only Drive listing/media audit and isolated replay/photo evidence/report. Preserve mixed work and
  inventory; no commits. No new live-camera or universal latency claims. Two-source/date gates unchanged.
* Both supplied folders re-accessed over read-only HTTPS (browser fetch failed, direct public download succeeded).
  Root product folder: 39 JPEGs (38 named-product photos + demo samples), Item List, Duplicates with 28 JPEGs.
  Weekly root: five team folders, Navigate_to_room.mp4 and a document. Listed immediate children of all five team
  folders; deeper AMR/LLM folders and Drive videos not processed. Root lists and Angiolite JPEG unchanged from 18 Oct.
* Read Item List and Weekly document, Verification HTML + 11-slide 30-Sep deck, 3-slide grasping deck. Two Vision PDFs
  downloaded/catalogued, not full-text extracted. Important distinction: Verification specifies per-field states,
  barcode/HRI cross-checks and confirmed dimensions/GTIN for a complete record. Current identity green/all_fields_read
  does not implement that entire contract. Item List explicitly allows mixed-product photos; filename is not box truth.
* New isolated 60-frame replays of owner's latest 18-Oct clips: Angiolite 2.5 x 29 VERIFIED 5.539 s; 4 x 19 3.286 s,
  both text + REF; absent 4 x 24 never VERIFIED. AC status 0 before/after every run: battery diagnostics only, not
  evidence for 1-2 s live acceptance. All produced playable annotated videos, CMD transcripts and paired photos.
* Newly downloaded actual Angiolite photo: 22 detected boxes, each read once with original full evidence policy;
  4 x 19 one two-source MATCH, 4 x 24 one MATCH + three candidates, 2.5 x 29 no MATCH. Single-photo identity
  decisions, not repeating a still to simulate temporal stability or live latency. Training overlap unknown.
* Evidence/report: `results/media_review_20261020/REPORT.md`, `DRIVE_CONTENTS.md`, JSON inventories/summaries,
  per-run CMD logs/videos/photos, per-box photo readings/crops/panels, `validation.json`. Core pipeline unchanged.
  **415 passed in 49.02 s** (`pytest_full.log`). Master inventory hash unchanged:
  `01253e7d08a3f967a4dbd4465e31070ca6ac7fc1cf125f2de586a36f94edc3a3`. No staging/commits.
* OPEN: complete per-field verification, readable barcode/HRI/LOT/expiry/brand, controlled charged live timing;
  still no general 1-2 s claim. Other Drive photos/videos were inventoried, not all tested or read.

## 2026-10-19 - Codex: recording, CMD transcript and session photos (completed)

* Owner requires video, written CMD evidence and photographs for each detection session.
* Claim focused launcher logging, paired automatic/manual photographs, tests and instructions. Keep existing raw/view
  videos, two-source verification and inventory isolation. Preserve mixed work; no commits.
* FOCUS_WEBCAM now runs the camera through `scripts/run_logged.py`: displayed stdout/stderr are also written to
  `cmd_<take>.txt` with command, UTC start/end and exit code. Child inherits stdin so in-terminal requests remain
  possible; launch/child failures propagate and the launcher points to the transcript rather than saying saved.
* `--evidence-photos` in focused modes enables original + annotated PNGs on state/evidence changes, manual S and
  final frame. `src/box_photo_evidence.py` freezes input frames, writes on one background thread with a bounded
  backlog, waits at close and surfaces write errors. Per-photo JSON/index records status, source and timing.
  Existing raw/video recordings and OCR crops retained; annotated writer open failure is now explicit.
* Validation: **415 passed in 29.03 s** (`results/session_evidence_20261019/pytest_full.log`). Includes stdout/stderr,
  inherited input, nonzero exit, queued photo pixel preservation, write failures and state/loss/photo integration.
  Real-model replay through transcript wrapper: exit 0, 45-frame playable annotated video, CHECKING/PENDING/
  VERIFIED/FINAL photo pairs all readable, complete CMD footer. Raw VideoRecorder copy test saved 20 frames and
  20 timing rows. `async_replay/validation.json` and artifacts in the same scratch folder. Not a new live-camera
  test or latency benchmark (the replay's VERIFIED event was 2.24 s; no new 1-2 s claim).
* Launcher startup/exit and ASCII/CRLF passed (`smoke/checks.json`). Inventory SHA-256 unchanged at
  `01253e7d08a3f967a4dbd4465e31070ca6ac7fc1cf125f2de586a36f94edc3a3`. No staging/commits. Guide updated.
* OPEN: live terminal/window S/Q smoke and full-field/consistent-latency work from the entry below. Automatic
  transcripts apply to new runs; missing older terminal output cannot be reconstructed.

## 2026-10-19 - Codex: focused VERIFIED view and observed crop evidence (local implementation completed)

* Owner requests actual two-source verification, green target outline and number/barcode/detail crops, with measured
  time to VERIFIED. Latest live visual-only locks were 2.142 s / 2.360 s, not verified results.
* Claim: focused view/reader, isolated evidence export, launcher, tests and documentation. Preserve mixed work,
  inventory and historical recordings; no commits. Keep two-source and requested printed-date gates unchanged.
* Evaluate latest recorded sessions; report unmet timing/detail requirements explicitly. No stress workflow.
* Added `--verify-target`, selected by FOCUS_WEBCAM / DEMO F. Green requires existing stable MATCH plus independent
  combined evidence and the requested date. Blue remains pending. Missing details continue reading after green;
  stale/lost/conflicting tracks lose verification. No SelectionSession, grasp, inventory writes or retrieval command.
* Angiolite tries the readable label orientation first and quick barcode decode; every fourth read uses the normal
  orientation/full-spine/hard-barcode fallback. Exact size still comes from observations, not appearance or inventory.
  Normal physical reader unchanged. OCR tokens now optionally preserve polygons for actual numeral crops.
* `src/box_target_evidence.py` exports same-track/epoch reader inputs, enlarged number crops, full size labels/spine,
  raw evidence, field provenance and JSON. REF-derived dimensions explicitly marked. Unknown fields stay null.
  Barcode-like letters are only candidate artifacts, never displayed as decoded barcode proof. Panel is below image.
* Latest owner session `focus_20261018_192726_400`: previous live locks were blue only. New AC-powered replays of those
  recordings: 2.5 x 29 4.837 / 1.722 s; 4 x 19 1.749 / 1.714 s; absent 4 x 24 never VERIFIED in 60 frames.
  Final numeral-panel checks: 2.5 x 29 1.698 s (crops ready 1.747 s), 4 x 19 1.730 s (crops ready 1.804 s).
  All positives use text + REF; all charged runs logged AC=1 at start/end, sequential processes. Timers exclude
  model startup, use first-frame wall time, not video timestamps. Initial battery diagnostics ~9.7-10 s are not a
  controlled comparison. Small replay sample, not live guarantees. Numeric enlargement did not cure weak 2.5 OCR;
  the 0.90 confidence gate was retained and unsuccessful enlargement is not shipped.
* Validation: **409 passed in 29.08 s**, `results/verified_20261019/pytest_final.log`. Same folder holds final_results,
  panel_final_results, summary JSON, videos, previews, crops and diagnostic scripts. Launcher cmd startup/exit and
  ASCII/CRLF passed (`smoke/checks.json`). Master inventory SHA-256 unchanged:
  `01253e7d08a3f967a4dbd4465e31070ca6ac7fc1cf125f2de586a36f94edc3a3`. No staging or commits.
* OPEN: consistent 1-2 s requirement (one AC run 4.837 s), live retest, complete field verification. Barcode/GTIN,
  LOT, brand and expiry are not all read; `all_fields_read` remains false. Some 4 mm text is only a supporting
  diameter hint under existing rules, so the structured dimension remains explicitly REF-derived. Need clearer
  close-up / visible barcode face. Do not equate green identity verification with completing all requested details.
  Instructions and full timing caveats: `docs/FOCUSED_WEBCAM.md`. Preserve mixed Claude/Codex ownership.

## 2026-10-18 - Codex: focused sibling-box speed and precision (local work completed)

* Owner requests review of latest webcam results and two Drive folders, no stress test in the current workflow,
  precise target outlines and 1-2 s target acquisition among same-product boxes with different details.
* Scope: visual-only target reader/scheduling and launcher, isolated replay diagnostics/tests and guidance. Review
  Claude's webcam fine-tune and current launcher edits first; preserve mixed changes, inventory and historical evidence.
  No commits. Physical PICK and printed-expiry rules remain unchanged. Timings must distinguish startup, first request,
  subsequent requests, and replay versus live. Claude-made reference/training labels remain pending owner review.
* Owner chose Angiolite and confirmed charger connected (Windows AC status also observed as 1). The supplied command
  was all-box detection only: it omitted --request / --lock-only. New `FOCUS_WEBCAM.bat` and
  `config/box_pipeline_config_focus.yaml` use the owner's existing webcam YOLO26 544 x 960 candidate, two single-thread
  OCR workers, and isolated session outputs/inventory copy. DEMO F opens it. Stress options removed from DEMO and
  TEST_WEBCAM menus; Claude's no-stress session S change retained. Historical scripts/results remain intact.
* New `src/box_visual_read.py`: tolerant colour-band reading priority, a two-sibling shortlist, exact evidence checks,
  early exit after a complete printed source, no barcode/extra-source-only work for visual mode, and re-evaluation of
  cached evidence on continuous tracks when requests change. Lost/stale tracks cannot reuse identity. Normal verified
  gather/read policy unchanged. Visual scheduling consumes completed OCR after detection to avoid an extra frame delay.
  Requested expiry ranks the correct dated sibling ahead of an otherwise equal candidate.
* Outline refinement now allows a supported outer side with the opposite detector side fixed, and an ends-only retry
  when neighbours block lateral correction. Work is deferred until target identity is established. Known 18-Oct
  Angiolite examples now reach the visible bottom beyond the colour seam; uncertain top flaps can still fall back.
  Display geometry is not used as OCR/identity/grasp evidence.
* Final 60-frame replays, one process at a time: 18-Oct Angiolite 2.5 x 29 **1.557/1.507 s**, 4 x 19 **1.593/1.841 s**;
  17-Oct 2.5 x 29 **1.947 s**, 4 x 19 **2.115 s**. Elapsed wall time starts at first frame after model initialization;
  these are not video timestamps or live-camera guarantees. Earlier same-model 18-Oct baseline: 28.27 s for 2.5 x 29.
  Absent 4 x 24 did not lock. All positive locks have repeated requested REF in the read logs, no PICK/grasp outputs.
* Accessed both owner-supplied public Drive folders read-only. Product folder's Angiolite photo: 22 detected cartons
  with REF sizes read; 4 x 19 supported one, 4 x 24 four, 2.5 x 29 none. Single-image support only; training overlap
  unknown, not held-out accuracy or PICK. Weekly/Vision contains reports; interactive HTML recaps June-August, not
  a new detection recording. Source copies/listings and annotated audit stay in ignored scratch.
* Validation: **397 passed in 29.35 s**, `results/codex_speed_20261018/pytest_full.log`; cmd.exe startup/exit smoke and
  ASCII/CRLF checks passed. `final_results.json`, `validation.json`, per-run videos/reading logs, `drive_audit/`, and
  `smoke/` in that folder contain evidence. Master inventory SHA-256 remains
  `01253e7d08a3f967a4dbd4465e31070ca6ac7fc1cf125f2de586a36f94edc3a3`. No staging/commits.
* Guide: `docs/FOCUSED_WEBCAM.md`. OPEN: live webcam pose/timing and loss/reacquisition test, owner inspection of edge
  quality, unreadable/occluded details and larger sets of identical designs. Latest-session trials met 1-2 s; older
  view has a 2.115 s case. No universal 2 s claim. 17-Oct frames contributed to Claude's fine-tune, so that clip is not
  held out. Global webcam/RealSense defaults and physical verifier remain unchanged; only the explicit focus profile
  selects this webcam-specific model.

## 2026-10-18 - Claude Code: tight box outlines on the webcam; stress test out of session S (completed)

* Owner: drop the stress test from the session, chase speed and accurate, tight outlines. `DEMO.bat` session S is now
  5 steps without the stress test (menu 6 / TEST_WEBCAM 7, 9 still run it; code untouched). CRLF/ASCII kept.
* New `scripts/bbox_precision_eval.py`: reference outlines (consensus + ends set by eye) and per-setting tightness
  (mean IoU, IoU >= 0.85 share, end error px). Owner's 18-10 webcam session: every model cut box ends (v4 live 45% tight,
  ends 13.7 px off); training labels are whole-spine, the webcam view / scale is the cause.
* Webcam fine-tune of YOLO26 final (server `~/Downloads/exp_webcam_ft_20261018`, 10 frames of the 17-10 session,
  Claude-made labels): on the unseen 18-10 session 85% tight at 736 x 1280, 0 false boxes and 66% tight at 544 x 960
  (~33% faster than v4 1280). D455 regression: 0.88 false boxes per frame at 544 x 960. Not deployed; details and
  models in `results/bbox_precision_20261018/README.md`. Next: more webcam poses, owner-checked labels.
* Tests: 383 passed. Laptop on battery: no speed claims.

## 2026-10-18 - Codex: webcam target-lock view and full-spine outline (local work completed)

* Owner requested reducing PICK-related delay, focusing on one locked target, inspecting terminal/session results and
  outlining the complete visible box. Review latest saved session first. Root owns webcam runner/launcher integration;
  independent geometry work is limited to a new outline module/tests; session audit is read-only.
* Keep visual lock distinct from verified PICK; preserve two-source and printed-expiry requirements for physical
  selection. Never infer hidden box boundaries or use display-only outline refinement as identity evidence.
* Preserve mixed uncommitted changes, inventory master and historical evidence. No commits.
* Added `--lock-only` via `src/box_target_view.py`: one display outline, CHECKING before repeated complete identity
  support, visual LOCKED afterwards with OCR paused. Detection/tracking continue; loss, stale tracking, conflicting
  pending readings or a new request resume reading. Dated requests need the matching date before pausing. This mode
  never instantiates SelectionSession, writes grasp commands, or changes inventory status.
* New display-only `src/box_outline.py` follows supported carton sides/ends beyond printed colour seams. An unresolved
  end retains detector corners; uncertain sides, image clipping and increased neighbour overlap fall back. Tracking,
  OCR and grasp retain detector geometry. Header sits above the image rather than covering box ends.
* DEMO menus 3/4/5, TEST_WEBCAM 4/8 and TASK_DEMO live use visual mode; DEMO V retains verified PICK, S/6 remain
  verification tests. Removed the unmeasured 1-2 s promise from TASK_DEMO. Guide: `docs/TARGET_VIEW.md`.
* Read-only audit of saved session `demo_20261017_193144_697/sesi_20261017_193153_220`: Conqueror text matched but no
  second source; first dated Sled Bag stress request timed out (not a full 20-box evaluation); inventory had no `pos`
  map. Saved terminal showed Ctrl+C during preflight followed by false READY. DEMO now rejects all nonzero exit codes,
  including the negative Windows interruption code. Original session/evidence untouched.
* Validation run 17 Oct, final review 18 Oct: **383 passed in 152.34 s**, log
  `results/codex_target_view_20261017/pytest_full.log`. This includes 29 new visual/outline tests. Real cmd.exe startup,
  exit-code branching (0/1/-1073741510), ASCII/CRLF checks passed in `smoke/checks.json` under the same scratch folder.
* Real-model Conqueror replay: 120 frames, 10 OCR reads, visual target #15 held, process exit 0. Evidence:
  `replay_checked/checks.json`, `target_view.jsonl`, `run.log`, `preview.jpg`, `target_vis.mp4`. No PICK/grasp outputs;
  inventory copy and master still SHA-256 `01253e7d08a3f967a4dbd4465e31070ca6ac7fc1cf125f2de586a36f94edc3a3`.
  Replay logs distinguish video time from elapsed wall time; model startup is excluded. This was a functional replay,
  with other validation work running, not an AC-powered live latency benchmark.
* OPEN: live webcam test on charger, current-pose map review, lock loss/reacquisition at the physical shelf, and edge
  quality across multiple boxes. No guarantee of 1-2 s or exact full outlines for ambiguous/occluded boxes. The active
  v4 model is unchanged; owner label review, YOLO26 deployment decision and physical grasp work remain separate.

## 2026-10-17 - Codex: external webcam as current demo camera (completed)

* Owner requested using a normal external USB webcam for now. Scope: DEMO.bat / TASK_DEMO.bat, camera-aware demo preflight and its
  tests, session/follow-up guidance. Use the existing webcam profile (v4 OpenVINO 736 x 1280, requested 1920 x 1080);
  preserve two-source PICK, inventory copies, isolated outputs and all pre-existing uncommitted edits.
* No commits. RealSense support stays available separately; current webcam grasp output is pixels only.
* DEMO defaults to the webcam profile; TASK_DEMO defaults to webcam even if D455 is connected. Preflight briefly
  checks the configured webcam frame/resolution and releases it; it does not import the RealSense SDK for webcam
  backends. Missing webcam is FAIL, lower-than-requested resolution is WARN. Explicit RealSense profile still works.
* DEMO menu 5 asks for a webcam recording instead of defaulting to historical D455 evidence. Session pytest uses a
  fresh temporary directory inside the session. Updated TEST_SESSION_GUIDE, REMAINING_WORK and CODEX_FOLLOWUP docs.
* Validation: **354 passed in 33.94 s**; log results/codex_followup_20261017/pytest_webcam_final.log. Real cmd.exe
  startup/exit and missing-video checks passed, confirming msmf / auto / 1920 x 1080, model input 736 x 1280 and isolated
  outputs; master inventory hash unchanged. Both changed launchers are ASCII/CRLF. Smoke logs: webcam_smoke/ under
  the same scratch folder. No physical webcam test or FPS measurement; owner still needs charger and shelf setup.

## 2026-10-17 - Codex: remaining-work fixes and YOLO26 candidate preparation (completed local work)

* Owner requested continuing unfinished work and fixing remaining issues. Scope: safe demo/session outputs and current
  config, review of target-lock / PICK / grasp integration, and preparation of the YOLO26 laptop candidate without
  switching the live model. File scopes are separated between workers; this entry and final validation are coordinated
  by Codex. Existing Claude and Codex edits remain uncommitted; no staging, commits or ownership reassignment.
* Preserve two-source PICK, required printed expiry, verified inventory state and historical evidence. Human label
  review, physical camera/arm validation and the scanner approach decision remain with the owner/team.
* Fixed DEMO.bat: current RealSense config copied into a fresh output folder per launch, isolated logs/screens/grasp,
  preflight uses that same config/inventory, inventory-derived request plan, missing replay/config/copy error handling.
  Changes also in scripts/sesi_config.py and scripts/demo_preflight.py; live v4 unchanged.
* Fixed grasp frame pairing: src/box_camera.py never substitutes another frame's depth; src/box_stream.py pins the
  delivered colour/depth/intrinsics through slow detection/OCR and reconnects. scripts/test_webcam_box.py consumes
  that pair and includes located/decoded barcode polygons in no-grip regions (location is not identity evidence).
* Fixed src/box_grasp.py metric dimensions/gaps to use both focal lengths; src/box_crops.py rectification at 45 degrees;
  src/box_lock.py falls back after partial-only readings and no longer calls expiry-pending two-source evidence "1 source".
  Tests added/extended; docs/GRASP_OUTPUT.md and docs/TARGET_LOCK.md updated. PICK/expiry verifier rules unchanged.
* New scripts/prepare_yolo_candidate.py: downloaded final YOLO26 weights match server SHA-256; exported fixed
  OpenVINO [1,3,544,960] and checked two task photos in isolated scratch. .pt photo profile: Braun PICK (text + REF),
  CHROMA candidate; fixed OpenVINO: neither PICK. Candidate not activated. Details: docs/CODEX_FOLLOWUP_20261017.md.
* Validation: full suite **350 passed** in 38.31 s with fresh workspace --basetemp and -p no:cacheprovider. Initial
  default-temp run had 301 passes / 28 fixture PermissionErrors from an inaccessible existing Windows temp folder.
  Full log: results/codex_followup_20261017/pytest_full_final.log. No physical camera/arm test or timing benchmark;
  laptop AC status was 0 and owner confirmed charger disconnected. No commits or staging.
  Launcher startup/exit and missing-replay guard also passed real cmd.exe smoke checks (logs in
  results/codex_followup_20261017/demo_smoke/); no camera was opened.
* OPEN: charger-powered laptop FPS/shelf A/B, owner review of Claude labels/outlines/audit places, original/closer CHROMA
  photo, new-brand full training after review, physical expiry/depth tests with the team, written scanner approach decision.

## 2026-10-17 - Claude Code: open items of 16-10 (second sources, size tables, fair YOLO comparison, shelf A/B) (completed)

* `src/box_ref.py` `VALID_SIZES`: Terumo Accuforce / Ryurei / Ultimaster Nagomi ordering tables (terumo-europe.com,
  read 17-10); a REF decoding to a size Terumo does not make counts as "not read" (like `_x100`). Shelf clip audit:
  wrong REF 7 -> 0, right REF unchanged; Accuforce 2.75 x 20 (K16) now PICKs (2 of 3 server replays; text + REF
  DC-RM2720 in the verify log). Tests in test_box_ref.py.
* `scripts/second_source_audit.py` -> `results/meeting_next/second_source_audit/` (README): the 11 boxes still without
  PICK carry only a DataMatrix / 1D barcode / tiny REF; forced whole-spine OCR and the barcode digits cut out and enlarged
  gave 0 REF and 0 GTIN. 9 of them are in the verified GTIN catalog: a scanner or closer view gives the 2nd source.
  The audit uses an inventory COPY with 9 box places set by Claude by eye (marked in the note column).
* YOLO comparison (`results/meeting_next/yolo_compare/`, server `exp_yolo_compare_20261016`): fair-start control
  YOLO11 from yaml + COCO weights, batch 4 = 0.679 mAP50-95 (YOLO11 DOTA 0.708, YOLO12 0.645, YOLO26 0.746). Far
  detection now matched per box (`far_eval.py`, 20 reference spines = consensus of the 12 fold models, checked by
  Claude, owner check pending: `gt_ref_preview.jpg`) at the laptop input 544 x 960: perfect frames v8 64%, v11 6%,
  v12 46%, v26 100%; the box COUNT hid a missed + false box (YOLO11 960).
* YOLO26s-OBB final trained like the 14-Oct YOLO11 final (server `exp_yolo26_final_20261017`). Shelf A/B on the server
  (`results/meeting_next/shelf_ab_20261017/`, three runs at the same time, server heavily loaded so times are 3-5x the
  laptop): v4 8 of 19, YOLO11 final 9/20, YOLO26 final 8/20 PICK, 0 wrong box in all; YOLO26 counted all 20 boxes.
  Not switched: needs an OpenVINO laptop run on the charger + the two new-brand photos, owner's decision.
* `src/box_detector.py` `fixed_input` (config `model.fixed_input`, default off): a .pt model keeps [h, w] with the full
  letterbox, like the fixed OpenVINO export (v4 on a test frame: 20 boxes OpenVINO, 20 fixed .pt, 17 square-960 .pt).
  Tests in test_box_detector_input.py.
* Task 2 (CHROMA): the WhatsApp photo is 1108 x 1477; no barcode on it decodes, the REF is ~6 px. Stays candidate;
  needs the original / a closer photo, and CHROMA GTINs are not in the verified catalog.
* Tests: 329 passed. Laptop was on battery today: no laptop timings were taken.
* OPEN: owner check of the Claude-made labels / reference outlines / audit places; scanner decision; YOLO26 laptop run.
* Short version for Codex (incl. the uncommitted mixed working tree on `codex/cvat-expiry-review-20261014`):
  `docs/CODEX_BRIEF_20261017.md`.

## 2026-10-16 - Claude Code: meeting actions (target lock, FPS, crops, confusion matrix, YOLO comparison) (completed)

* Target lock `src/box_lock.py` + `scripts/test_webcam_box.py` (default on with an inventory, `--no-lock` / config
  `selection.lock_target`): only the locked box is read, locked by its place in the shelf map (nearest detected box
  to the mapped spine centre after the median camera shift; works with missed detections), else slot, else look;
  PICK rule unchanged. "Not the target" needs 2 consistent readings; a box whose size text matches stays locked when
  only a REF / GTIN disagrees ("code doubt"). `docs/TARGET_LOCK.md`. Session starts searching at once
  (`SelectionSession.start_searching`, count in background).
* Shelf map `scripts/rak_inventaris.py peta` (menu P in TEST_REALSENSE / TEST_WEBCAM, asked in TASK_DEMO task 3):
  updates only slot + place (`pos`, new inventory field) + look of verified rows (identity / check status untouched,
  backup written). Found on the way: the old `order` column was the track creation order, not the slot.
* Lock A/B on the laptop (charger, 15-Oct D455 stress recording, same plan): lock 8/20 PICK, median 4.2 s, slowest
  10.0 s; no lock 8/20 (same boxes), median 9.5 s, slowest 66 s; box locked 0.1-0.3 s; 0 wrong box.
  `results/lock_test_20261016/README.md`. Limit left: small REF print at shelf distance (REF scale table in the doc).
* FPS badge top right; PICK crops panel (spine / barcode located or decoded / size label + structured output)
  `src/box_crops.py`, saved per PICK, barcode position into grasp.jsonl.
* `scripts/confusion_identity.py` -> results/meeting_next/confusion (out-of-fold 14 Oct: 0 wrong PICK of 1,276;
  live 15 Oct: 8/20 PICK, all 8 on the asked box, checked on the verify log).
* Server: `~/Downloads/exp_yolo_compare_20261016` (YOLOv8/11/12/26-s OBB, same 3 folds and recipe as 14 Oct; YOLO12 has
  no published OBB weights -> yaml + COCO weights) and `~/Downloads/exp_lock_20261016` (lock vs no-lock replay of the
  15-Oct stress recording, .pt on CUDA). Videos of results/ moved to `~/Downloads/video_archive_laptop_20261016`
  (SHA-256 checked, MANIFEST.sha256; `arsip_video_20261016/LANJUTKAN_ARSIP_VIDEO.bat` resumes).
* YOLO comparison done (`scripts/yolo_compare_table.py --dir results/meeting_next/yolo_compare`, server
  `results.json` + `far_detection.json`): mAP50-95 3-fold v8 0.700, v11 0.708, v12 0.645 (no OBB weights, batch 4),
  v26 **0.746**; laptop FPS 15.8 / 16.9 / 11.2 / 16.1; D455 shelf (20 boxes, imgsz 960) frames with exactly 20:
  2% (v8 adds duplicates) / 66% / 33% / **96%**. Far detection is reported as exact / fewer / more, never a mean
  count (extra boxes would hide missed ones). Not switched yet: YOLO26 first needs training on all data + shelf A/B.
* Tests: 314 passed.
* Fixed 16-10: the slot rule needed all 20 boxes detected (the server .pt model saw 17-18) and the look cannot tell
  siblings apart (3 Accuforce 0.97, 2 Angiolite 0.89-0.90) -> place rule above. The server lock A/B was stopped for this
  (not representative); the laptop A/B replaces it.
* OPEN: owner / mentor decision whether the arm may approach a box locked with 1 source, with the barcode scanner as the
  2nd source before the basket (rule unchanged until then). Product size tables from the makers' catalogues so
  impossible REF sizes count as "not read". Full 5-fold training with the two new brands.

## 2026-10-15 (afternoon) - Claude Code: mentor tasks 1-3, two new brands, safer REF reading (completed)

* Mentor's request: text in, one bounding box out (Task 1 "Braun, DCB, 2.75mm, 30mm", Task 2 "Chroma, 2.75 , 33" on the
  13-Oct WhatsApp photos; Task 3 = live shelf, our assumption). Final: Task 1 **PICK** (text + REF 5023243), Task 2
  candidate on the right box (REF ~6 px, unreadable), Task 3 PICK (webcam 6.8 s, D455 recording 11 s).
* `src/box_verifier.py` + the four configs' `catalog:`: products `chroma` (biosensors), `sequent please neo` (b braun),
  aliases incl. "braun dcb" (bare "dcb" deliberately not: Agent is a DCB too). Before this the OCR did not know CHROMA.
* `src/box_ref.py`: B. Braun REF table (50232 + length code + diameter code, official brochure 6050333 p.7) and CHROMA
  `BCR-<dd><ll>`, both in the new `CODED` list, matched only as a whole code (no digit glued before / after): real
  misreads "NEFE50232243" / "NE502324246" gave plausible wrong sizes without it. Tests in test_box_ref.py.
* `src/box_evidence.py`: narrow spines enlarged 1.5x before the whole-spine OCR (`FULL_BOX_UPSCALE`), factor chosen on
  the 32 non-target boxes of the two photos (REF 9/32 -> 13/32, 0 wrong). Applies live too: stress replay below.
* `scripts/test_webcam_box.py`: target-only view (`--view`, key v; photos redraw on v); blue candidate only when its
  evidence covers the requested size (ranked by combined_score). `scripts/demo_foto_target.py`: no in-process patching
  any more, `--show`, extra args passed through. New `TASK_DEMO.bat` (tasks 1-3, camera choice) and `TEST_WEBCAM.bat`;
  TEST_REALSENSE.bat / TASK_DEMO.bat now take `config\box_pipeline_config_realsense.yaml` (the evidence copy stays
  untouched; it lacked the new catalogue).
* Labels: 8 + 27 boxes traced on image edges (labeling by Claude; owner checks the previews in
  `results/task_20261015/training_data/preview`). Target boxes greyed out of the training images.
* Quick fine-tune on the server: `~/Downloads/exp_newbrands_20261015` (own folder; master labels, exports and runs only
  read): v4 + 196 photos (2 classes) + the 2 new photos x6, 40 epochs. Model `models/box_v4s_nb1015*` used for the task
  photos only; the live demo stays on v4 until the full 5-fold training + identity safety evaluation.
* Tests: 303 passed. Stress A/B on the 02:21 recording (only the OCR scale differs): 1.0x 7/20, 1.5x 8/20 PICK, 0 wrong. Open: full training, more Braun / CHROMA photos, expiry
  labels for the two photos (not drawn yet), DEMO.bat still uses the evidence config.

## 2026-10-15 - Codex: student explanation and Gemini context (completed)

* Reviewed architecture, active configuration, key source paths, training split and the 14 October evaluation report. Created Indonesian undergraduate teaching material, a self-contained Gemini prompt and comparison rubric, and review notes in workspace `project_brief_20261015/`, outside both repositories.
* Clarified weighted evidence versus statistical independence, current two-class demo versus the three-class candidate, burst splitting versus whole-session separation, and working-tree depth/grasp code versus physical validation. No pipeline, model, label or inventory changes; preserved other assistants' changes. No external messages or commits.
* Validation: current local full suite, 282 passed in 30.67s. Training/evaluation metrics in the material remain the documented 14 October snapshot; no new camera/robot experiment or server audit was run. Open: live date tests, fair model comparison, deployment-domain data and physical grasp validation.

## 2026-10-14 - Codex: CVAT audit, remaining expiry review and three-class training

* Checked the supplied 33-photo CVAT XML ZIP with rotated OBB geometry. The server already had this batch imported/converted; preserved its newer specs and added three missing visible product/logo panels.
* Reviewed the remaining 97 expiry photos individually; server/laptop snapshot now has 196/196 completed specs and 1,489 expiry labels. Corrected ten spine regions in five shelf photos and date ownership at Agent/Sled Bag boundaries. Backups retained. No inventory state changed.
* Readiness passed: 196 seeds, folds 66/65/65, zero new geometry problems/crashes (49 existing accepted border warnings). Synced 789 master files with SHA-256 checks; verified all 1,176 training-label copies against the frozen snapshot.
* Training completed: three 120-epoch folds plus final 200 epochs, batch 8, 70 minutes. Accuracy gate against v6 FAIL: box mAP50-95 0.706 -> 0.666; size_label 0.719 -> 0.697. Configurations also differ (v6: five 200-epoch folds). Do not replace the demo model.
* Out-of-fold identity safety evaluation PASS: 116/123 detected, zero wrong MATCH over 1,276 wrong-identity requests (87 same-product/different-size). Correct requests: 13 MATCH, 42 candidate, 61 confirmation. Fixed evaluator's three-class crash by ignoring expiry output in identity evaluation; regression test and full suite: 271 passed.
* Owner confirmed concurrent server label review by Claude late in this run. Stopped further master-label edits; training/evaluation inputs are frozen and checked. Final master hashes remain unchanged from our snapshot. No commits made.
* Evidence: docs/CVAT_EXPIRY_REVIEW_20261014.md; labeling/audit_local/cvat_20261014/ (relative to workspace); server audit/codex_20261014/. Candidate weights and final 33-photo label ZIP are available locally. Guide/checklists updated; launcher completion text corrected locally.
* Open: coordinate any further label revisions with Claude. Physical D455 shelf tests, date-specific Sled Bag requests 2/16/0, and new rack recordings still require the owner at the shelf. The identity evaluation does not validate date-specific live PICK.

## 2026-10-14 (night) - Claude Code (laptop): step-by-step guide for the remaining work

* `docs/id/PANDUAN_SISA_PEKERJAAN.md` (owner's working copy) and `docs/REMAINING_WORK.md`: task 0 install tools, task 1 CVAT
  correction of the 33 photos, task 2 date review + 3-class training, task 3 shelf tests (incl. Sled Bag date test 2 / 16 / 0).
* Found while writing it: every date tool numbers the boxes from `label_specs/`, so boxes corrected in CVAT
  (`labels_reviewed_obb/`) were invisible to them and `mulai_training_3kelas.sh` would stop ("cek kelas 2 manual"). New
  `labeling/tools/reviewed_to_spec.py` converts them back into the spec and re-proposes the dates; `review_tanggal.py` now
  skips box drafts, prints the sheet path (VS Code Remote-SSH) and counts all photos instead of a fixed 163. Tested on a copy of
  the project (CVAT-style correction -> import -> convert: geometry 0.0 px off, queue correct). CVAT runs on server port 8080.
* Open: owner runs `RUN_ON_SERVER.bat langkah9_tools_review` (package already uploaded; remote writes are the owner's).

## 2026-10-14 (evening) - Claude Code (laptop): audit, server comparison run, commit

* Folder move is DONE (the entry below is outdated on this point): `mackay_vision\code` and `\labeling` are real folders, the old
  paths are junctions. Both repos `git fsck` clean, 268 tests pass. Parent `README.md` updated.
* Server comparison run on the real manifest (`labeling\server_results\langkah7_manifest.txt`, 14 Oct 04:00) with
  `tools/compare_server.py`; report in `labeling\server_results\COMPARE_SERVER_LAPTOP.md`. Server 51 done / 112 auto, laptop
  76 / 87; 104 expiry specs differ in content (77 are `auto` on both sides, so the server changed them, probably
  `rapatkan_label_tanggal.py`); 26 are `auto` on the server but `selesai` on the laptop (the 11 Oct batch rule); 1 the other way.
  Date tools exist only on the server (`review_tanggal.py`, `rapatkan_label_tanggal.py`, `cek_tanggal_terlewat.py`,
  `bandingkan_model.py`, ...); `labels_final_obb/IMG_20260923_132329_751.txt` differs.
* SSH key installed by the owner (`ssh trse-server` works without a password). Server copy pulled read-only to
  `labeling\server_copy_20261014\`. Finding: the server is newer (12 Oct 02:31-02:56) and built on the laptop's 76: it fixed the
  hourglass-as-"8" bug, tightened the iVascular date labels and deliberately reopened 26 photos with missed dates. So the laptop
  took the server's expiry specs, `labels_final_obb` and date tools (laptop copy kept in
  `labeling\archive\laptop_before_server_merge_20261014\`); labels are now identical. Exception: `labeling_progress.csv` is newer
  on the LAPTOP (7 photos re-checked 8 Oct = `selesai`; the server still has `perlu dicek manual`, likely why v6 used 156 photos).
* 33 `New image` photos (28 Sep 16h) put in the label queue: copied to `raw/`, v4 drafts (`box_v4s.pt`, 1280, cpu) via
  `pred_to_spec.py` (note SEDANG DIKERJAKAN, out of training), expiry proposals (19 dates), batch
  `labeling\batches\baru_20260928_16h` with `REVIEW.md`. They are one-box close-ups (~40 boxes, not ~400); v4 drafts include the
  drink bottle and monitor as boxes. Seed set unchanged (51 photos), the 33 are pool "sedang dikerjakan".
* Fixes: `labeling/tools/auto_expiry.py` crashed on a photo with 0 boxes and on very thin tiles; `export_datasets.seed_state`
  let tool-reviewed labels into training without the expiry review (now same rule as specs; verified on simulated input).
  `code/src/box_expiry.py` got the server's hourglass fix (`82029-03-24` was dropped, so a dated Sled Bag request could not PICK);
  2 new tests, 270 pass. The live pipeline is not re-tested with the camera.
* Server sync DONE: the owner ran `labeling\RUN_ON_SERVER.bat langkah8_sync_laptop` (14 Oct 06:26): 4 server files backed up to
  `~/Downloads/cadangan_sebelum_sync_laptop_20261014.tgz`, 33 photos into `raw/` (201 files), 278 files extracted, batch rebuilt.
  Checked with a fresh read-only manifest: 640 of 644 files identical, expiry status 51 / 145 on both sides, the 33 photos
  sha1-identical. Remaining differences are docs only (server keeps 3 Indonesian guides at the root; the laptop root
  `PANDUAN_TRAINING_3KELAS.md` has a newer "Status 11-10" section). From now on edit labels on the server only.
* Push: branch `claude/sync-20261014` is on GitHub in both repos (PRs not opened). The old labelling remote
  `Saca2346/mackay-med-robot-vision-` never existed; the owner created `Saca2346/mackay-med-robot-vision-labeling` (private),
  now `origin` of `labeling\`, with `main` (first commit) and the branch pushed. History checked: code and docs only (72 files).
* Missing locally, not needed by the box pipeline: `data/test_assets/synthetic_shelf.mp4` (`scripts/generate_test_assets.py`),
  `data/book_db.sqlite`. The 3-class model and v5/v6 are only on the server.
* Commit: all work since 8 Oct committed on branch `claude/sync-20261014` in both repos (not pushed). The labelling repo got a
  whitelist `.gitignore` first (it had none; `git add -A` would have added the photos).

## 2026-10-14 (later) - Claude Code (laptop): folders, English, server comparison

* Owner decisions: one parent folder `Downloads\mackay_vision\` with `code\` + `labeling\`; English names with the old names kept as
  one-line wrappers; compare server and laptop label data before choosing a master.
* Folder move NOT done yet: the first attempt with PowerShell Move-Item half-moved the repo (62 files) because a program held `.git`;
  everything was moved back with robocopy and verified (31,967 files, `git fsck` clean, 268 tests pass). The move is now a script the
  owner runs with VS Code / Claude closed: `Downloads\MOVE_TO_MACKAY_VISION.bat` (cmd `move` = rename only, junctions keep old paths).
* English entry points: `TEST_REALSENSE.bat`, `TEST_LIVE_LOCAL.bat`, `TEST_ROBOT_LAPTOP.bat`, `TEST_CAMERA_TO_SERVER.bat`; labelling
  repo `RUN_ON_SERVER.bat` (outputs to `server_results\`). Old `UJI_*.bat` / `SERVER_JALANKAN.bat` forward to them (tested).
* Docs: English `README.md`, `docs/BOX_PIPELINE.md`, `CHECKLIST_DEMO.md`, `LIVE_TEST_LOCAL.md`, `LIVE_TEST_CHECKLIST.md`,
  `CAMERA_TEST_PRODUCT_LIST.md`; Indonesian originals in `docs/id/`, superseded docs in `docs/archive_id/` (references updated).
  Labelling repo: `docs/LABELING_TOOLS_GUIDE.md`, `docs/TRAINING_3_CLASSES.md`, new `README.md`; root clutter moved to `archive\`
  (list in `archive\MOVED_FILES_20261014.csv`) and `server_results\`.
* Server comparison: `server_scripts/langkah7_manifest.sh` (read-only) + `tools/compare_server.py`; tested on a simulated list
  (found exactly the 2 injected differences). Not run on the server yet.

## 2026-10-14 - Claude Code (laptop)

* Sled Bag by expiry: a request may contain a date (`permanent sled bag 2029-03-24`); PICK only after that printed date is read.
  Files: `src/box_verifier.py`, `src/box_inventory.py`, `src/box_demo.py`, `scripts/test_webcam_box.py`, `scripts/demo_request.py`.
  Tested: `tests/test_expiry_request.py` (8 tests) + full suite 268 passed. Not tested live with the camera yet.
* Box list in the cmd: one number per box (Sled Bags by date), 0 = any Sled Bag (`scripts/demo_daftar_kotak.py`).
* Recorded test session: `DEMO.bat` menu S (hardware check, tests, calibration, hardware test 1, 20-box stress test, random
  requests, report). New: `scripts/rekam_layar.py`, `scripts/sesi_config.py`, `scripts/ringkas_sesi.py`. Not run live yet.
* Open: server and laptop label data differ (server: expiry status 51 done / 112 auto; laptop: 76 done). Decide which is master
  before the next training (see `docs/REMOTE_SERVER_CLAUDE_CODEX.md`).
* Open: 13 rows + 4 pictures from Claude's replays (13 Oct 16:17-16:32) are in `results/bukti_laporan_20261008`; cleaned copy in
  `results/demo_20261012/replay_menu5_analisis/`, owner decides whether to apply it.

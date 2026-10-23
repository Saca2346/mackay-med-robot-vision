# Brief for Codex: Claude Code's work of 15-17 Oct 2026

Read with `AGENTS.md` (safety rules, unchanged) and `docs/HANDOFF.md` (entries 15-10 afternoon, 16-10, 17-10 have the
details). This page is the short version: what is finished, where it is, what is open, and what to watch out for.

Team jobs (meetings of 15 and 17 Oct; this repository is the vision part):

| Who | Sub-system | Touches this repo through |
| --- | --- | --- |
| Saca Setya Wardana (owner of this repo) | Vision, OCR & barcode | everything here; assigns and reviews the assistants' work |
| Leon | Grasp point & Isaac Sim (geometric grasp from the box + RGB-D depth instead of GraspNet) | reads `grasp.jsonl` (`docs/GRASP_OUTPUT.md`) |
| Farhan & Makata | Dual arm, end-effector CAD, MoveIt | the "move closer" step and a scanner mount on the arm (open) |
| Leo | Voice, LLM & database | requests in text form (`parse_request`) |
| Jonathan | AMR, sensors & integration | not directly |
| Pak Satriyo | Mentor | gave tasks 1-3 and the meeting actions |

A name "Joey" in older meeting transcripts is a transcription error: there is no Joey on the team.

## 1. Working tree: read this before any commit

* The checkout is on **`codex/cvat-expiry-review-20261014`** and holds **uncommitted changes of two assistants**:
  * files last changed on 14-10 (Codex's CVAT / expiry review and earlier work): `scripts/eval_verify_detector.py`,
    `tests/test_eval_verify_detector.py`, `docs/CVAT_EXPIRY_REVIEW_20261014.md`, `docs/REMAINING_WORK.md`,
    `docs/id/PANDUAN_SISA_PEKERJAAN.md`, `docs/id/PROTOKOL_DATA_DUA_RAK.md`, `docs/DATA_RAK_PROTOCOL.md`,
    `scripts/kalibrasi_kamera.py`, `src/box_calib.py`, `tests/test_box_calib.py`, `scripts/ambil_frame_data.py`
    (the 14-10 calibration / data-protocol changes are not logged in HANDOFF: check their diff and ask the owner
    before committing them);
  * files changed 15-17 Oct by Claude Code (this brief): everything else in `git status` (list in section 6). The
    15-10 night files (`src/box_stress.py`, `scripts/ringkas_sesi.py`, `tests/test_box_stress.py`,
    `tests/test_box_verify_multi.py`: stress-test fixes such as the Terumo "8" REF hint) are not logged in HANDOFF
    either; they belong to Claude's stress-test work of that night.
* Nothing of 15-17 Oct is committed. AGENTS.md says one branch per assistant: **do not commit Claude's files into the
  codex branch**; the owner decides how to split (e.g. `claude/meeting-actions-20261017`). Commit only when the owner asks.
* Full suite on 17-10: **329 passed** (`venv\Scripts\python.exe -m pytest tests -q`).

## 2. What is finished

| Area | What | Where |
| --- | --- | --- |
| Mentor tasks 1-3 (15-10) | Text in -> one box out. Task 1 B. Braun 2.75 x 30 photo: PICK (text + REF 5023243). Task 2 CHROMA 2.75 x 33 photo: candidate (REF ~6 px). Task 3 live shelf: PICK | `results/task_20261015/README.md`, `TASK_DEMO.bat` |
| New brands | Products `chroma` (Biosensors), `sequent please neo` (B. Braun) in `src/box_verifier.py` + the configs' `catalog:`; REF rules B. Braun (official table) and CHROMA `BCR-ddll`, matched only as whole codes (`CODED`) | `src/box_ref.py` |
| Wider spine OCR | Narrow spines enlarged 1.5x before whole-spine OCR (`FULL_BOX_UPSCALE`) | `src/box_evidence.py` |
| Task-photo model | v4 fine-tuned with the 2 new photos (targets greyed out) | `models/box_v4s_nb1015*` (task photos only) |
| Grasp output for Leon | `grasp.jsonl` per PICK: box corners, centre, gaps, no-grip zones, depth (D455) | `src/box_grasp.py`, `src/box_camera.py`, `docs/GRASP_OUTPUT.md` |
| Webcam version | `TEST_WEBCAM.bat` next to `TEST_REALSENSE.bat` | root |
| Target lock (meeting action 1) | Only the requested box is read. Locked by its place in the shelf map (`pos`), else slot, else look. "Not target" needs 2 consistent readings; size text matching + only a code disagreeing keeps the lock. FPS badge on screen. `--no-lock` = old behaviour | `src/box_lock.py`, `docs/TARGET_LOCK.md` |
| Shelf map | `rak_inventaris.py peta` (menu P): writes only order / `pos` / look of verified rows, backup first | `scripts/rak_inventaris.py`, `src/box_inventory.py` (`pos` field) |
| PICK crops (action 4) | Spine, barcode (decoded or located), size label, structured output panel; `barcode_px` / `barcode_xyz_m` in grasp.jsonl | `src/box_crops.py` |
| Confusion matrix (action 3) | Out-of-fold photos: 0 wrong PICK of 1,276; replays: 0 wrong box | `scripts/confusion_identity.py`, `results/meeting_next/confusion/` |
| YOLO comparison (action 2) | v8 / 11 / 12 / 26 s-OBB, same 3 folds and recipe; fair-start control for YOLO12; laptop FPS; far detection matched per box | `scripts/yolo_compare_table.py`, `scripts/yolo_laptop_fps.py`, `results/meeting_next/yolo_compare/` |
| Terumo size tables (17-10) | `VALID_SIZES` from Terumo's ordering tables (Accuforce, Ryurei, Ultimaster Nagomi): an impossible size = "not read" | `src/box_ref.py`, tests in `tests/test_box_ref.py` |
| Second-source audit (17-10) | Why boxes get no PICK from the shelf, per source | `scripts/second_source_audit.py`, `results/meeting_next/second_source_audit/README.md` |
| `.pt` with the laptop input (17-10) | `model.fixed_input` (default off): a .pt keeps [h, w] with the full letterbox, like the OpenVINO export | `src/box_detector.py`, `tests/test_box_detector_input.py` |
| YOLO26 final + shelf A/B (17-10) | YOLO26s-OBB trained like the 14-10 YOLO11 final; 3-model replay on the server | `results/meeting_next/shelf_ab_20261017/README.md` |
| Storage | 59 project videos moved to the server (SHA-256 checked) and deleted on the laptop | server `~/Downloads/video_archive_laptop_20261016/` |

## 3. Key numbers (all with evidence in the folders above)

* Lock A/B, laptop on the charger, 15-Oct D455 recording: PICK 8/20 both; median 9.5 -> **4.2 s**; slowest 66 -> 10 s;
  box locked 0.1-0.3 s; 0 wrong box.
* YOLO mAP50-95 (3-fold): v8 0.700, v11 0.708, v12 0.645, **v26 0.746**; control v11 with v12's start (COCO, batch 4)
  0.679. Laptop FPS (OpenVINO 544 x 960): 15.8 / 16.9 / 11.2 / 16.1. Shelf frames perfect at the laptop input
  (all 20 found, no false box): 64% / 6% / 46% / **100%**.
* Size tables: wrong REF readings 7 -> 0 on the shelf clip, right ones unchanged; Accuforce 2.75 x 20 now PICKs (2 of 3
  server replays, REF DC-RM2720 in the log).
* Server shelf A/B (loaded server, times 3-5x the laptop): v4 8 of 19, YOLO11 final 9/20, YOLO26 final 8/20 PICK; 0 wrong.
* 11 of 20 shelf boxes still get no PICK: their only second source is a DataMatrix / 1D barcode / tiny REF; forced
  whole-spine OCR and the barcode digits cut out and enlarged gave 0 REF and 0 GTIN. Physical limit of the D455 there.

## 4. Open (not done)

* **Owner checks** of Claude-made data: task-photo labels (`results/task_20261015/training_data/preview`), the 20
  far-detection reference outlines (`results/meeting_next/yolo_compare/gt_ref_preview.jpg`), 9 box places in the audit
  inventory copy (`second_source_audit/inventaris_audit.csv`). Never describe these as human-reviewed.
* **Team decision**: may the arm approach a box locked with 1 source, with the barcode scanner as the 2nd source? Rule
  unchanged until decided in writing.
* YOLO26 not switched: needs an OpenVINO export, a laptop run **on the charger**, and the two new-brand photos; owner decides.
* Task 2: needs the original / a closer photo (the WhatsApp copy is 1108 x 1477; no barcode decodes); CHROMA GTINs are not
  in the verified catalog (registering them needs the physical box and the owner).
* Full 5-fold training with the new brands; laptop timing of the lock with the size tables (charger); grasp test with
  Leon (depth); `DEMO.bat` still uses the evidence config. Leo's Drive folder "藥材" was not accessible.

## 5. Pitfalls found (save yourself the time)

* `pkill -f` / `pgrep -f` over ssh match the ssh command itself and kill the session: use `pgrep -f "[p]attern"` or PIDs.
* `scp host:dir/{a,b}` must be unquoted; the server sometimes closes back-to-back scp connections: retry per file.
* `BoxDetector` turns a [h, w] imgsz into a square for .pt models (needed for portrait photos). v4 on a shelf frame: 20 boxes
  OpenVINO 544 x 960, 17 boxes .pt 960 square. For laptop-equivalent .pt runs set `model.fixed_input: true`.
  OpenVINO is not installed on the server; ONNX on its CPU took ~1 s per frame under load.
* Laptop `Win32_Battery.BatteryStatus` 1 = on battery: timings 3-8x slower, do not report them.
* The lab GPU (RTX A5000) is shared: YOLO12 at batch 8, imgsz 1280 ran out of memory.
* Replays run on video time: 20 requests x 90 s exceed the 21-min recording, so a slow run may not reach request 20.
* A box COUNT per frame hides a missed box plus a false box; far detection is matched per box now (`far_eval.py`).

## 6. Claude's files of 15-17 Oct

New: `src/box_lock.py`, `src/box_crops.py`, `src/box_grasp.py`, `scripts/confusion_identity.py`,
`scripts/yolo_compare_table.py`, `scripts/yolo_laptop_fps.py`, `scripts/second_source_audit.py`, `tests/test_box_lock.py`,
`tests/test_box_grasp.py`, `tests/test_box_detector_input.py`, `docs/TARGET_LOCK.md`, `docs/GRASP_OUTPUT.md`,
`TASK_DEMO.bat`, `TEST_WEBCAM.bat`, this file.
Changed: `src/box_ref.py`, `src/box_verifier.py`, `src/box_evidence.py`, `src/box_detector.py`, `src/box_session.py`,
`src/box_inventory.py`, `src/box_camera.py`, `src/box_stress.py`, `scripts/test_webcam_box.py`, `scripts/rak_inventaris.py`,
`scripts/demo_foto_target.py`, `scripts/ringkas_sesi.py`, the four `config/box_pipeline_config*.yaml`, `TEST_REALSENSE.bat`,
`tests/test_box_ref.py`, `tests/test_box_verifier.py`, `tests/test_box_verify_multi.py`, `tests/test_box_stress.py`,
`tests/test_box_camera.py`, `docs/HANDOFF.md`.

Server folders (own experiment folders; the master `~/Downloads/prelabel_size_label_full` was only read):
`exp_newbrands_20261015`, `exp_lock_20261016` (superseded), `exp_yolo_compare_20261016` (backups of the 4-model results
and script inside), `exp_yolo26_final_20261017`, `exp_shelf_ab_20261017`, `video_archive_laptop_20261016`.
Results, photos, videos, models and inventory files stay off GitHub.

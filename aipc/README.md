# AI PC photo pipeline (Tasks 1-3 on photos, 4 Nov 2026)

The production program runs on the laptop (model B, YOLO26s-OBB 544x960 on the Intel iGPU through OpenVINO, RapidOCR on
two CPU threads). These scripts run the mentor's requests on **photos** on the lab AI PC (RTX A5000 24 GB, 32 threads),
with the laptop's decision code (`src/`) unchanged. Nothing here changes the laptop's production configuration.

Server folder: `~/Downloads/claude_offline_20261101/aipc_20261104/` (the scripts expect `code/` next to `tasks/`, a copy
of this repository's `src/`, `scripts/`, `config/`, `data/gtin_catalog.csv` and the task photos; `pylib/` holds
onnxruntime-gpu 1.23.2 with its CUDA 12 libraries, installed with `pip install --target pylib`, so the shared venv is
untouched). Photos, recordings, models and inventory files stay on the lab server and the laptop, never in git.

## Box detector (`det/`)

| Script | What it does |
|---|---|
| `train_det.py MODEL IMGSZ BATCH [EPOCHS] [split] [PHOTO_MULT] [FRAME_STEP]` | Trains a candidate. `split hold` leaves out the 28-Sep photo session (the test); `hold2` leaves out 23 Sep; `all` = final. At most 60% of the shared GPU's memory; a file `PAUSE` holds the next job back. |
| `queue.sh`, `queue2.sh`, `queue3.sh`, `queue4.sh`, `queue5.sh` | The candidate queues that were run (round 1: s/m at 1280; round 2: photo-weighted 1600 px, YOLO11m; round 3: YOLO26s at 1600 px, whose queue3 slot was stopped in epoch 1; round 4, 6-7 Nov: YOLOv8s/m and YOLO12s/m at 1280, YOLO12 built from its yaml with the detect weights). |
| `eval_det.py WEIGHTS IMGSZ NAME` | Held-out outline test on the 28-Sep photos: recall, precision, IoU, end / side errors ("past" or "short" of the box), share of tight boxes. |
| `tightlib.py` | Shared truth loading, polygon IoU, matching, per-edge errors, the edge correction. |
| `snap_eval.py WEIGHTS IMGSZ NAME` | Edge correction: settings chosen on the dev photos (23 Sep + 1 Oct) only, then applied once to the test. |
| `viz_boxes.py` | Deck figure: owner outlines, detector outlines, corrected outlines, zooms. |
| `refine_lib.py` | Learned edge refiner (6 Nov): a small CNN on full-resolution strips across each end of the detected rectangle says where the box ends; sides by a constant. |
| `train_refiner.py PRED_NAME [EPOCHS] [truth\|pred] [TAG] [--all]` | Trains it on the dev photos only (owner outlines; inputs = detector outlines and jittered copies). |
| `cv_refine.py PRED_NAME [EPOCHS]` | Chooses the set-up by 5-fold cross-validation on all 109 dev photos, then trains the final net (`weights/refiner.pt`). `select_refine.py` = the earlier choice on 16 validation photos (noise; kept for the record). |
| `eval_refine.py PRED_NAME ...`, `viz_refine.py` | The 28-Sep test, once: as detected / old snap / refiner; deck zooms. `run_refine.sh` = the v1 / v2 runs. |
| `det_probe.py heldout CONFIG --tilt 0,20,35` / `count CONFIG OUT PHOTO...` | 8 Nov, tilted and missed boxes: input sizes, turned views, ensembles and ordered merges on the 28-Sep photos also turned 20 / 35 degrees (truth turned with them), and box counts with pictures on Leo's Drive photos (`probe/`). Model B 544x960 94.7 / 93.9 / 92.3% recall, 1280 98.2 / 98.8 / 98.5% (seen: upper bound); held-out YOLO26s / YOLO12s lose 2.8 / 3.6 points at 35 degrees at 544x960 and none at 1280. `probe_run.sh`, `probe2.sh`, `misses.py`. |

## Tasks (`tasks/`)

| Script | What it does |
|---|---|
| `aipc_tasks.py score SET ...` | Every box x every printed size of the photo (confusion matrix), truth `gt_<SET>.csv` (printed size read by Claude). |
| `aipc_tasks.py time SET ...` | The mentor's request: photo from disk to PICK, repeated runs. |
| `aipc_tasks.py task3` | The 41 Task 3 requests of 29 Oct, each on its own shelf still. |
| `aipc_tasks.py demo SET / task3:Kxx` | Deck picture with the one green box. |
| `aipc_main.py <same arguments>` | Launcher to use with `--procs N` (the spawned OCR processes import this tiny file, not `aipc_tasks.py`). |
| `ocr_worker.py` | One OCR process: its own RapidOCR engine (same models as the laptop), optionally the direction classifier on the CPU. |
| `task3_positions.py` | Box positions of the Task 3 shelf from the outlines the program itself verified (live 29 Oct + replays). |
| `profile_*.py`, `cprof_ocr.py`, `ort_*.py` | The profiling that found where the OCR time went. |
| `task_eval.sh` | Each detector candidate on the task photos (accuracy only, OCR on CPU processes). |
| `equiv.sh` | Proof that `--lazy 1` gives the same cell as `--lazy 0` for every task-photo pair and Task 3 request. |
| `final.sh` | The final timing run (four outlines, lazy, edge-corrected outline), Task 3 both ways, one-outline comparison. |
| `final_all.sh`, `quiet_ab.sh`, `quiet_ab2.sh` | Timing runs gated on an idle GPU (wait until utilisation < 10% for 30 s), GPU load logged. |
| `ab_speed.sh`, `prio_ab.sh`, `t3_ab.sh` | Interleaved speed A/B of the fixes (arena, lazy crops, ordered queue, priority queue, eager outlines). |
| `close_runs.sh` | WhatsApp timing of the final reader and the equivalence of the scheduling fixes on all 814 pairs. |
| `video_runs.sh`, `evidence_video.py` | Logged runs and, on the laptop, the evidence videos replayed from their logs (nothing decided in the video). |
| `stop_mine.sh` | Stops this work's runners and OCR processes on the server (not the training queue, not other users). |
| `stages.sh` | Stage times of the final reader (photo read, detection, crops, skim, each box verified, PICK, all read). |
| `refine_runs.sh`, `refine_ab.sh` | The refiner in the reader (`--refine 1`): timing, Task 3, deck zooms; A/B of its cost against the old snap. |
| `aipc_tasks.py live SET_DIR ...` | The laptop's real-time recordings (20 / 12 / 10 boxes) replayed from the first camera still of each take (`livesets/<set>/set.json`; ids by left-to-right order, read by Claude). `live_runs.sh`. |
| `aipc_tasks.py drive` / `demo_drive ABS_JSON` | Leo's Drive photos: every photo read fully (untimed), then for a catalogue photo one size seen on it is asked and timed; `drive.json`, pictures with the green box only. `drive_runs.sh`. |
| `ocr_compare.sh`, `ocr_gpu.sh` | OCR engines compared with `OCR_VARIANT` (ocr_worker.py): v6s (laptop), v5det, v5srv, v5en (PP-OCRv5, ONNX from huggingface monkt/paddleocr-onnx), easy (EasyOCR 1.7), tess (Tesseract 5); accuracy and time. |
| `det_tasks.sh` | The YOLOv8 / YOLO12 candidates on the task photos, like `task_eval.sh`. |
| `final_quiet.sh`, `chain_1107.sh` | Final run of 7 Nov on a quiet GPU (gate: utilisation < 30% for 1 min, load logged): Tasks 1-3, recordings, Drive, demo pictures. |

Final reader (5 Nov; refiner 6 Nov; 1280 drawing 7 Nov): `aipc_main.py time|task3 ... --imgsz 544x960 --workers 27 --procs 24
--cls-cpu 1 --proc-threads 1 --order skim --snap 1 --refine 1 --refine-device cuda --draw-imgsz 1280 --variants 1 --lazy 1
--ordered-prefetch 1 --priority 1` (Task 3 with `--procs 20`: 24 processes ran out of the shared card's memory). Quiet GPU,
7 Nov (`final_quiet.sh`): Task 1 1.18 s, Task 2 1.55 s, WhatsApp copy no box (all read 1.67 s), Task 3 median 1.96 s with 9
of 20 right boxes green; recordings 20 / 12 / close: 12 / 3 / 0 green (laptop live 12 / 4 / 3); Drive: 6 green photos.
Same cells as without the scheduling options on all 814 task-photo pairs and 41 Task 3 requests (`equiv.sh`,
`close_runs.sh`). The refiner needs `det/weights/refiner.pt` (server only, not in git).

Main options: `--det W --imgsz N` detector; `--procs N --proc-device cuda|cpu --proc-threads T --cls-cpu 1` OCR in N
processes; `--order skim` likely box first (order only); `--variants 1` each box read from four outlines (as detected,
sides +-6%, ends +2%), green only if one reaches two sources and none reads another identity; `--claude-slots 1` Task 3
slot truth for the 6 slots no run verified, read by Claude from the shelf photo (not owner review); `--snap 1` the
PICKed box's drawn outline is edge-corrected (setting chosen on the dev photos by `snap_eval.py`), inside the PICK time;
`--refine 1` that outline is placed by the learned refiner instead (`--refine-device cuda|cpu`); `--draw-imgsz 1280` the
PICKed box is detected again at 1280 px first (the 1280 outline with IoU >= 0.3 to the PICKed one, else the PICKed one;
25.8% tight on the 28-Sep test against 19.4% from 544x960);
`--lazy 1` the other outlines of a box are skipped when its first outline already reads another identity (that box can
never turn green under the variant rule), and their crops are cut only when read; `--ordered-prefetch 1` every box's OCR
calls queued from one thread in skim order (27 racing threads lost the order); `--priority 1` a priority queue in front of
the processes (calls a read waits for, then a live box's other outlines, then the rest); `--barcode 1` experiment only
(2 wrong Sled Bag greens: a GTIN is not compared with the requested date, see docs/HANDOFF.md).

Decisions cannot be changed by the speed-ups: the OCR cache is keyed by the exact image bytes, the skim only orders the
reads, and every box still gets the laptop's full read and the laptop's two-source rule. The edge correction only
changes the drawn outline, never the crops that are read.

### 8 Nov: fail-case fixes (AI PC only; the laptop's src/ is unchanged)

| Option | What it does | Effect (asked box green, T1 · T2 · WA · T3, of 7 · 26 · 7 · 20; wrong green 0 in every run) |
|---|---|---|
| `--split-sizes 1` (`size_split.py`) | A joined size word ("3.533") is split when exactly one split is a valid size (d.0/d.5 + 6-99, or d.00/.25/.50/.75 + 6-200). | 6 · 17 · 2 · 9 -> 6 · 19 · 2 · 9 |
| `--spine-size 1` (`size_split.py`) | No size label found: the big printed numbers of the spine, by the laptop's `size_field_tokens` rule (numerals >= 22% of the strip height). | -> 6 · 23 · 7 · 11 |
| `--views "B:1280:0+B:544x960:0"` (`multiview.py`) | Detector views merged in order: the first view's boxes kept, a later view adds only boxes overlapping none (IoU >= 0.3 or centre inside). Model B was trained at 1280 and ran at 544x960; 1280 holds on tilted boxes (`det/det_probe.py`). | with its outlines only: 7 · 15 · 7 · 11 |
| `--alt-views 1` | Each box also read from the other view's outline of the same box (IoU >= 0.5), in its 4 variants; the variant rule over all 8. | **7 · 24 · 7 · 11 = 49 / 60** (`rd_altb`) |
| `--alt-lazy 1` | Speed: the other view's outlines only when the box's own 4 give no MATCH and no conflict (a MATCH there is decided by the 4-outline rule of 7 Nov). | same cells (`rd_altz`) |
| `--skim-name 1` | Speed: read order by the requested product name in the skim, then the numbers; the 4 outlines of the few boxes with the name are queued at once. | same cells (`rd_altz`) |
| `--refine-webcam 0` | Webcam frames (Task 3, recordings) get the 1280 outline without the refiner (it ran into the shelf bar on K04 / K10; rule set after seeing them). | drawing only |
| `--drive-outlines 1` | `drive`: the discovery reads every outline the reader reads (before: the first outline only). | Drive two-source boxes 38 -> 100 (544) -> 138 (final) |

`--alt-views` was designed after seeing which Task 1 / Task 2 boxes each outline read: chosen on the test photos
(disclosed in the deck). The view order (1280 first) was chosen on the held-out 28-Sep session. Leo's Drive photos and
the recordings are the check on other photos. Runs: `acc_chain.sh`, `alt_chain.sh`, then `queue_runner.sh` (job files
in `queue/`, one GPU run at a time, each stopped after its time limit; a Drive run left waiting 18 h on 7 Nov after an
OCR worker could not get GPU memory while the detector probe ran). `ocr_worker.py` logs an engine error with its message
and retries the image once at 0.75 scale (the errors are GPU memory, `BFCArena`, worse with more detector views).
Diagnosis: `dbg_alt.py`, `dbg_reads.py`, `drive_cmp.py`.

## Product model (`look/`)

`apply_owner_review.py` applies the owner's review of 4 Nov (`review_labels.csv`: 1,583 crops) to the product-look
labels; `run_look.sh` retrains EfficientNet-B0 (3 seeds, both splits). Scored on the laptop with
`results/id_model_20261102/score_owner.py` (the shelf truth comes from the inventory, which stays on the laptop).

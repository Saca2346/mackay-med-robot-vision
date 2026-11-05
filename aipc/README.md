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
| `queue.sh`, `queue2.sh`, `queue3.sh`, `queue4.sh` | The candidate queues that were run (round 1: s/m at 1280; round 2: photo-weighted 1600 px, YOLO11m; round 3: YOLO26s at 1600 px, whose queue3 slot was stopped in epoch 1). |
| `eval_det.py WEIGHTS IMGSZ NAME` | Held-out outline test on the 28-Sep photos: recall, precision, IoU, end / side errors ("past" or "short" of the box), share of tight boxes. |
| `tightlib.py` | Shared truth loading, polygon IoU, matching, per-edge errors, the edge correction. |
| `snap_eval.py WEIGHTS IMGSZ NAME` | Edge correction: settings chosen on the dev photos (23 Sep + 1 Oct) only, then applied once to the test. |
| `viz_boxes.py` | Deck figure: owner outlines, detector outlines, corrected outlines, zooms. |

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

Final reader (5 Nov): `aipc_main.py time|task3 ... --imgsz 544x960 --workers 27 --procs 24 --cls-cpu 1 --proc-threads 1
--order skim --snap 1 --variants 1 --lazy 1 --ordered-prefetch 1 --priority 1` (Task 3 with `--procs 20`: 24 processes
ran out of the shared card's memory). Task 1 1.07 s, Task 2 1.25 s, Task 3 median 1.6 s; same cells as without the
scheduling options on all 814 task-photo pairs and 41 Task 3 requests (`equiv.sh`, `close_runs.sh`).

Main options: `--det W --imgsz N` detector; `--procs N --proc-device cuda|cpu --proc-threads T --cls-cpu 1` OCR in N
processes; `--order skim` likely box first (order only); `--variants 1` each box read from four outlines (as detected,
sides +-6%, ends +2%), green only if one reaches two sources and none reads another identity; `--claude-slots 1` Task 3
slot truth for the 6 slots no run verified, read by Claude from the shelf photo (not owner review); `--snap 1` the
PICKed box's drawn outline is edge-corrected (setting chosen on the dev photos by `snap_eval.py`), inside the PICK time;
`--lazy 1` the other outlines of a box are skipped when its first outline already reads another identity (that box can
never turn green under the variant rule), and their crops are cut only when read; `--ordered-prefetch 1` every box's OCR
calls queued from one thread in skim order (27 racing threads lost the order); `--priority 1` a priority queue in front of
the processes (calls a read waits for, then a live box's other outlines, then the rest); `--barcode 1` experiment only
(2 wrong Sled Bag greens: a GTIN is not compared with the requested date, see docs/HANDOFF.md).

Decisions cannot be changed by the speed-ups: the OCR cache is keyed by the exact image bytes, the skim only orders the
reads, and every box still gets the laptop's full read and the laptop's two-source rule. The edge correction only
changes the drawn outline, never the crops that are read.

## Product model (`look/`)

`apply_owner_review.py` applies the owner's review of 4 Nov (`review_labels.csv`: 1,583 crops) to the product-look
labels; `run_look.sh` retrains EfficientNet-B0 (3 seeds, both splits). Scored on the laptop with
`results/id_model_20261102/score_owner.py` (the shelf truth comes from the inventory, which stays on the laptop).

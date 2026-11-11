# Resume for Codex: AI PC vision work, 4-9 Nov 2026 (state at hand-over)

Written by Claude Code on 9 Nov 2026 for the owner's move to Codex. Read `AGENTS.md` first (safety rules, repository
hygiene), then this file, then the newest entries of `docs/HANDOFF.md` and `aipc/README.md` (every option with its
measured effect). Everything below is AI PC only: the laptop's production code and configs in `src/` / `config/` are
unchanged by this work.

## 1. What the work is

The mentor's request on a photo: medicine text in, ONE box out (PICK), with two independent sources (`AGENTS.md` rule 1).
The model was built for the laptop; it now runs on the lab AI PC (RTX A5000 24 GB, 32 threads, shared server) on photos.
Tasks:

| Task | Request | Photos |
| --- | --- | --- |
| Task 1 | "Braun, DCB, 2.75mm, 30mm" | Leo's Drive photo (`t1_drive`) and the mentor's own photo (`t1_orig`, file `results/task_20261015/input/task1_braun.jpg`). The owner asked (9 Nov) not to name the second one "WhatsApp": it is Task 1, the mentor's photo. |
| Task 2 | "Chroma, 2.75 , 33" | Leo's Drive photo (`t2_drive`) |
| Task 3 | 41 requests of 29 Oct | one shelf still per request (20 boxes on the shelf, 21 absent products) |
| Recordings | the laptop's live tests | replayed from up to 5 stills per take (20, 12, 10-box sessions) |
| Leo's Drive | every catalogue photo | discovery + one timed request per photo |

## 2. Final state (night of 8 Nov, run `fqzf*`, deck Version 24)

Final configuration (queue job `aipc/tasks/queue_1108/102_pick_crops_final.job`):

```
OCR_ALGO=HEURISTIC aipc_main.py <mode> ... --imgsz 544x960 --workers 27 --cls-cpu 1 --proc-threads 1 --order skim
  --snap 1 --refine 1 --refine-device cuda --draw-imgsz 1280 --variants 1 --lazy 1 --ordered-prefetch 1 --priority 1
  --refine-webcam 0 --webcam-ends 1 --views "B:1280:0+B:544x960:0" --alt-views 1 --alt-lazy 1 --skim-name 1
  --split-sizes 1 --spine-size 1 --par-variants 1 --pick-crops 1 --procs 24|20 [--claude-slots 1 for task3]
```

| Result | Value |
| --- | --- |
| Asked boxes green on the task photos | 49 / 60 (35 on 7 Nov), 0 wrong green in 1,574 pairs |
| Task 1, time to PICK (crops included) | Drive 1.05 s, mentor's photo 0.87 s |
| Task 2 | 1.28 s, 24 / 26 boxes green for their own size |
| Task 3 | 11 / 20 right box green, 0 wrong, 0 / 21 absent; median 1.79 s, 8 / 11 under 2 s (Accuforce 2.17-2.24 s on a busy server; 2.00-2.05 s on the quietest) |
| Recordings | 13 / 4 / 4 green (laptop live 12 / 4 / 3), 0 wrong; median 1.61 / 2.24 / 5.22 s |
| Leo's Drive | 9 PICKs (8 catalogue photos + demo carton), 1.1-4.5 s, each checked by Claude against the printed size |
| PICK crops | brand 41, detail 44, barcode located 24 of 44 PICKs |
| Same-load A/B, no crops vs crops | T1 1.04 / 1.15 s, T1 mentor 0.91 / 0.87, T2 1.26 / 1.26, T3 median 1.69 / 1.62 |

Times are photo on disk to PICK (including the drawn outline and the crops), median of 5 runs after a warm-up, on a quiet
GPU; the server's CPU load from other users is logged and moves the times by up to ~30 % (compare rows at the same load).

## 3. Where things are

| What | Where |
| --- | --- |
| Code | this repo: `aipc/tasks/` (pipeline, tools), `aipc/det/` (detectors, outline refiner), `aipc/README.md` |
| Runs on the server | `trse-server:~/Downloads/claude_offline_20261101/aipc_20261104/` (`tasks/out/<tag>/`, `tasks/queue/`, `tasks/queue_runner.log`, `livesets/`, `det/`); python `~/trse_venv/bin/python`, `PYTHONPATH=../pylib`, entry `aipc_main.py`; work only in this folder |
| Copies on the laptop (not in git) | `results/aipc_20261104/final_1108/out/<tag>/` (fqzf, fqzf_t3, fqzf_live, fqzf_drive, fqzq*, fqzp2*, fqzp*, ...) with `crops/` |
| Result photos (Task 1-3, recordings) | laptop `results/aipc_20261104/final_1108/stills_f/` |
| Result videos | laptop `results/aipc_20261104/final_1108/videos_f/` (t1_drive, t1_orig, t2_drive, task3, live x3, drive) |
| Crop sheets | laptop `results/aipc_20261104/final_1108/deck_f/` |
| Deck (Claude Slides, 42 slides) | https://claude.ai/artifact/6fwfRNqgyqKheB1GyQE3yE (private; the owner shares it) |

Photos, videos, models, inventory and results are never committed (public repo, `.gitignore`).

## 4. The tools (all in `aipc/tasks/`)

* `aipc_tasks.py` modes `time`, `score`, `task3`, `live`, `drive`, `demo`; every option is listed with its effect in
  `aipc/README.md`. Run through `queue_runner.sh` with job files in `queue/` (one GPU run at a time; `queue/STOP` makes
  the runner exit when no job is left; it waits for an idle GPU before each job's runs).
* `pick_parts.py` (`--pick-crops 1`): brand / detail / barcode of the PICKed box, cut after the MATCH inside the PICK
  time; the barcode is located (never decoded) in its own process; `upright()` turns crops for display only.
* `recheck_upright.py` re-decides the display turn of saved crops on the laptop (used once for `fqzf*`).
* `pick_still.py`, `evidence_video.py`, `crop_sheet.py`: deck pictures, videos and crop sheets drawn on the laptop from
  the run logs (nothing decided or drawn by hand).
* `webcam_ends.py` (`--webcam-ends 1`): ends of the drawn outline on webcam stills.
* Tests: `tests/test_aipc_pick_parts.py`; the whole suite: `venv\Scripts\python.exe -m pytest tests -q` (672 passed).

## 5. Open items, in order

1. Sled Bag rule gap (owner first): for a dated request a GTIN may only agree together with the same printed date.
2. Laptop: bring the 8 Nov reading options (split sizes, spine size, 1280 + 544 views, both outlines) into the laptop
   reader - owner decision; none of it is in `src/` yet.
3. Toward 1 s: one batched OCR reader in one GPU process (also ends the OCR GPU-memory errors, `OCR_ENGINE_ERROR` in the
   run logs); read only the Accuforce REF band.
4. 98 % detection on new photos: the owner corrects the 936 pre-drawn outlines on Leo's 39 photos, then retrain YOLO12s
   and model B; the same relabel covers packed identical boxes.
5. 11 held boxes: a second frame or a closer photo. Decoding the located barcodes (and a 2-D reader for the Sled Bag
   code) only after the rule gap is closed.
6. Owner checks: 6 Task 3 slots (K08, K11, K12, K14, K19, K20) and the left-to-right ids of the recordings were read by
   Claude; the webcam end rule and the crop display turns were checked by Claude only.

## 6. Pitfalls met (save time)

* Git Bash converts arguments that start with `/` (e.g. `/_blob/<id>`) into Windows paths
  (`C:/Program Files/Git/_blob/...`); this broke the Task 1-3 pictures of deck Version 23. Pass such values inside a
  file, or set `MSYS_NO_PATHCONV=1`.
* Never print text in the same `ssh` command that streams a tar archive; use `< /dev/null` on `ssh`.
* Stop server runs by PID only (others share the server); the GPU is shared, so runs wait for an idle GPU.
* The OCR worker logs GPU-memory engine errors and retries once at 0.75 scale; the untimed Drive discovery can differ by
  a box or two between runs because of them (the timed PICKs did not).
* Display orientation of digit crops: OCR confidence, string closeness, classifier on/off and the direction classifier
  each failed somewhere; `pick_parts.upright` uses the tokens read in the run first.
* Commit and push only when the owner asks; one branch per assistant (`codex/<topic>` for Codex).

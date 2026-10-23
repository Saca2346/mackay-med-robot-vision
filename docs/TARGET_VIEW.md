# External webcam: one visual target

**19 October:** FOCUS_WEBCAM / DEMO **F** now uses `--verify-target`, with green
two-source identity verification and saved number/label/barcode evidence. See
[FOCUSED_WEBCAM.md](FOCUSED_WEBCAM.md) for measured verification latency and missing
details. The `--lock-only` mode described below remains explicitly visual-only.

**18 October:** for the requested YOLO26 webcam candidate and faster Angiolite
size selection, use `FOCUS_WEBCAM.bat` or DEMO **F**. See
[FOCUSED_WEBCAM.md](FOCUSED_WEBCAM.md) for measurements and limits. Stress actions
were removed from the DEMO/TEST_WEBCAM menus; historical scripts and results remain.

Use `DEMO.bat` -> **1** (preflight) -> **3** (target request). Choose a number from
the displayed list or enter, for example, `conqueror nc pro 3.5 x 15`.
The default is the external USB webcam profile at requested 1920 x 1080, with the
existing v4 OpenVINO model. Close other camera apps and unplug unused cameras.

Only one cyan outline is displayed. The status panel sits above the camera image
so it does not cover the top or bottom of the carton.

* **CHECKING**: an appearance/location hint or initial reading has selected a
  provisional box; its requested identity is still being read.
* **LOCKED - tracking; OCR paused**: at least two consecutive readings have each
  supplied complete matching identity support without contradictions. One source
  suffices for this visual mode. A dated request also needs its matching printed
  date. These repeated readings do not become two independent sources for PICK.
* The detector and tracker keep running. Losing the target, uncertain tracking,
  contradictory pending evidence, or a new request resumes reading. No outline
  is drawn for a target that is no longer detected.

Type another number/request in the terminal while the view is open; `?` lists
requests. In the image window, **o** reads the target again, **a** selects the shelf
area, **s** saves a picture, and **q** closes the view. The shelf crop is local to
this run unless an explicit `--roi` file is supplied. A map from another camera
pose may be inaccurate; appearance/location alone remains provisional.

`TEST_WEBCAM.bat` menus **4** and **8**, and the live task in `TASK_DEMO.bat`, also
use this mode. `DEMO.bat` **V** retains verified PICK; **S** retains the recorded
verification session without stress testing. PICK still requires the existing independent
sources and printed-date rules. Visual mode does not run PICK/TAKEN, save grasp
commands, or change inventory status.

## Outline refinement

`src/box_outline.py` looks for visible carton edges near the detector polygon.
It follows supported long sides past internal colour seams to supported carton
ends. One supported side can suffice when the opposite side stays at the detector
line. An uncertain end retains its original detector corners. Refinement falls
back if neither side is supported, the box is clipped, the proposed extent is too
large, or it would increase overlap with a neighbouring detection.

This is a display correction, not a newly trained detector. It cannot guarantee
exact corners for every box or recover hidden boundaries. The detector geometry
used by tracking, OCR and physical grasp remains unchanged.

## Outputs and timing

DEMO creates a new `results/demo_<timestamp>/` folder with its own config and
inventory copy. `target_view.jsonl` records changes of visual state, request,
track, display polygon, OCR pause status and `verified_pick: false`. Raw and
annotated recordings remain separate. The annotated image adds a 90-pixel header;
logged polygon coordinates refer to the original camera frame.

Events distinguish `time_basis` (`video` or `live`) and `seconds` from
`wall_seconds` since the first frame of the request. Model/camera startup is not
included. In an `--all-frames` replay, short video time is **not** live latency.
Neither detector milliseconds nor display FPS establishes time to correct target.
There is no verified 1-2 second guarantee. A charger-powered live test at the
current webcam pose is still required for latency claims.

For an isolated replay:

```bat
venv\Scripts\python.exe scripts\sesi_config.py config\box_pipeline_config_webcam.yaml results\my_target_replay
venv\Scripts\python.exe scripts\test_webcam_box.py --config results\my_target_replay\config_sesi.yaml --lock-only --source my_webcam_video.mp4 --inventaris data\rak\inventaris.csv --request "conqueror nc pro 3.5 x 15"
```

The inventory argument is read-only in visual mode. Keep the video's
`.mp4.times.csv` sidecar beside it when available. Do not direct trial outputs to
historical `results/bukti_*` evidence folders.

## Saved-session review, 17 October

Read-only terminal/video audit: `results/codex_target_view_20261017/audit/README.md`.
The original Conqueror run had repeated matching text with no readable REF/GTIN,
so it remained a candidate while visually locked. The original dated Sled Bag
stress run timed out on its first request; it was not a completed 20-box benchmark.
The inventory had appearance signatures but no `pos` map for the current pose.

The saved terminal also showed a preflight interrupted by Ctrl+C that was later
logged as READY. DEMO now treats every nonzero exit code, including Windows'
negative interruption code, as failure. The original session artifacts remain
unchanged.

# Webcam and RealSense eleven-carton target view

Run from `C:\Users\sacab\Downloads\mackay_vision\code`. Close an older camera process first and connect
the charger. The profile uses the existing YOLO26 model and D4551280x800@30FPS colour stream.
Barcode/QR decoding is disabled, as requested. Independent-source/date/confidence gates are unchanged.

## Calibration first, then detection

Webcam (external C930e on the current configured index):

```bat
venv\Scripts\python.exe scripts\run_target.py --camera webcam --config config\box_pipeline_config_webcam_eleven.yaml --request "angiolite 4 x 19" --calibrate-first
```

RealSense D455:

```bat
venv\Scripts\python.exe scripts\run_target.py --camera realsense --config config\box_pipeline_config_realsense_eleven.yaml --request "angiolite 4 x 19" --calibrate-first
```

Run one camera command at a time. Arrange the eleven cartons in the live calibration preview, then
press Enter for the hands-off countdown and measurement. Detection opens a fresh stream only when
the calibration completes with PASS or WARN. WARN remains an image-quality advisory, not identity
or speed approval. FAIL, ESC/cancel, or missing/invalid results stop detection. A replay `--source`
cannot use `--calibrate-first`, so recorded evidence never opens an unrelated live camera.

Calibration uses the exact isolated camera/model config, ROI and expected count of the detection
session. Its original photo, annotated photo and JSON go to that new session's `calibration/` folder;
`cmd_calibration.txt` and `session.json` retain the result. Names are generated automatically, so old
v1/v3 calibration evidence is preserved without manually choosing a new name.

The focused detection header now uses82%source image with18%dark background and outlined text inside
the native camera viewport, matching the transparent calibration preview. No opaque extra90pixel bar
is added. Raw camera/OCR images are unchanged; setting `visual.transparent_header:false` retains the
older outside-image header when needed. Close and restart already-running programs to load this change.

The view displays every visible detection, the selected target and observed/expected count11. It
does not cap the detector at11 or treat the count as correct identities. Background cartons use a
neutral outline and tracking number. Only the selected target is refined/stabilized and read for
identity; its blue/green status follows actual evidence. Green still requires independent sources.
Sled Bag name/date alone remains `TEXT READ`, not VERIFIED. Exact-size products with readable size
text plus independent printed REF can verify normally. Neither view authorizes physical retrieval.

A numbered CMD request menu includes the eleven reference boxes from the existing shelf session.
Type a menu number or a product/size/date while the window remains open; ? shows the list again.
Menu numbers select reference requests; on-image #numbers are live tracking IDs. They are not an
automatic identity map. The menu never supplies OCR fields, accepted labels or verification evidence.

| CMD menu | Reference box | Request |
| --- | --- | --- |
| 1 | K01 | permanent sled bag2029-04-26 |
| 2 | K03 | permanent sled bag2029-03-10 |
| 3 | K04 | essential pro3x40 |
| 4 | K05 | permanent sled bag2029-03-24 |
| 5 | K06 | xperience pro2.5x15 |
| 6 | K07 | accuforce2.5x12 |
| 7 | K10 | angiolite4x19 |
| 8 | K15 | angiolite2.5x29 |
| 9 | K16 | accuforce2.75x20 |
| 10 | K17 | permanent sled bag2029-03-12 |
| 11 | K21 | accuforce3.5x15 |

Actual requests/date strings are generated from the session's inventory copy; the table describes
the current reference scene only. A typed request changes the target using continuous-track cached
observations, explicitly labelled as warm timing. O resets observations for a fresh test. S saves
an extra original/display photo pair; Q finalizes videos. All outputs stay in a new session directory:
raw/annotated video, real timestamps, actual OCR inputs/crops, nullable fields, CMD and per-frame
`shelf_detections.jsonl`. Master inventory is never changed.

For a new full-shelf calibration choose an unused name:

```bat
venv\Scripts\python.exe scripts\kalibrasi_kamera.py --sesi realsense_eleven_20261022_v1 --config config\box_pipeline_config_realsense_eleven.yaml --min-boxes 11
```

Display all eleven cartons with their complete ends and labels. Check native source text quality;
calibration PASS does not certify OCR identity, independent sources or a two-second response. Optional
`--roi x1,y1,x2,y2` selects the same original-frame target region in calibration and detection.

## Sequential eleven-target live test

Run one command at a time from the project directory. Connect the charger, close an older
camera window, keep the same eleven cartons stationary and leave both ends visible.

RealSense:

```bat
venv\Scripts\python.exe scripts\validate_box_targets.py --live --camera realsense --config config\box_pipeline_config_realsense_eleven.yaml --calibrate-first --max-frames 60
```

Webcam:

```bat
venv\Scripts\python.exe scripts\validate_box_targets.py --live --camera webcam --config config\box_pipeline_config_webcam_eleven.yaml --calibrate-first --max-frames 60
```

Press Enter in calibration to measure. PASS/WARN starts the eleven requests sequentially;
failure/cancel stops the batch. Each request gets a separate fresh process, inventory copy,
raw/annotated video with timestamps, photos, OCR crops and CMD transcript. Each ends after
60 processed frames, including failures; there is no requirement to reach green before moving
on. Model loading repeats between requests and is excluded from the reported detection timer.
The CMD prints the current reference ID/request and a result after each trial. Do not type
another target or press O/A during a case; a changed request or reset timer invalidates that
case's deadline result. Use a new batch or --only Kxx for another fresh trial.
Q ends the current case early; Ctrl+C interrupts the batch. There is no live resume/overwrite.

The generated `results/all_boxes_review_<timestamp>/REVIEW.html` updates after every completed
case. Inspect each original/view photo, video, actual OCR inputs and independent sources; export
owner Accept/Reject/Need closer image decisions as CSV/JSON. Owner choices remain PENDING until
you decide. `results.json` retains text and green timings, count histograms, power and failures.
A blue TEXT READ result is not green verification, and identity VERIFIED is not full-field
completion. Calibration PASS checks framing/image quality, not identity or a two-second deadline.
Menu reference IDs do not establish physical box identity. Inventory is never updated.

For an existing recording, use the same script with `--source <video>` instead of `--live`
and omit `--calibrate-first`. Camera backend is inferred from the supplied profile, so an
actual D455 recording is no longer routed through a hardcoded webcam runner.

## Acquisition optimization and validation scope

For dated products without size fields and with barcode disabled, the reader now retries a small
date-shaped OCR token region before repeating whole-spine OCR. The region comes from actual token
geometry, with native pixels retained and a180degree orientation retry. Low-confidence text supplies
only a location: accepted date values still come from new OCR at the existing confidence threshold.
No request date fills a token, and multiple orientations remain one frame observation and one source.
Unusable regions retain the previous whole-spine fallback.

Evidence: `results/realsense_eleven_20261022/REVIEW.html` and `REPORT.md`. The available eleven-carton
recording originates from the webcam. It can test the new view/scheduler but cannot certify live
D455shelf timing, D455label readability or camera-equivalent accuracy. Recorded D455speed measurements
use its existing single-carton recording. The owner subsequently recorded D455eleven-carton session `target_realsense_20261022_085253_734152`; its fresh replay review is in `results/eleven_owner_20261022/batch_final/REVIEW.html`. Replays remain separate from new live acceptance.

## Powered follow-up, 22 October

The five recent profiles now reference `models/box_y26_maxB_544x960_openvino_model`.
The existing BIN bytes match the earlier results-folder export; XML graph structure matches
apart from friendly names and metadata differs only in export date. The old artifacts remain.

D455 focused sized requests refresh appearance priority every two frames during the first15processed
frames, then resume the existing ten-frame interval. `visual.early_hint_refresh:false` is the
comparison switch. The four webcam profiles explicitly keep it false. Unsized/date-only Sled Bag requests retain their previous interval because
an interleaved trial did not improve them. Appearance affects read order only, never identity.
Claude's queued-crop cancellation and observed-support priority are preserved.

The 085253D455 calibration has median width50.4px and a longest spine edge about692px in an
800px-high source. Uniformly moving closer to70px median projects that edge to962px, so full
spines would no longer fit. ROI filtering does not create native text pixels. Retain the full
ends and experiment with layout/orientation or a separate detail view rather than assuming
that a closer full-shelf view can always achieve70px. These are measured image-geometry
calculations, not intrinsic calibration or a certified camera working distance.

The owner selected existing recordings for this follow-up. Powered replays are in
`results/powered_followup_20261022/REVIEW.html`; these are separate recordings and do not
constitute a controlled simultaneous camera comparison or new live acceptance. Barcode stays
OFF; Sledname/datealone remains unverified. Disk audit lists old videos without deleting them.

For Angiolite, a located merged decimal token such as `2.529` now triggers one bounded OCR
read of each native size column. This does not split the string, derive numbers from REF or
lower confidence: the new values come from the actual left/right pixels. Original observations
and conflicts remain, both retries are part of one frame observation, and the independent
printed REF is still required for verification. Saved `size_retry_left/right` bands retain
actual inputs/tokens. `visual.size_column_retry:false` disables this fallback for comparisons.
The original scheduled date checks, including two consistent different dates before exclusion,
remain unchanged.

## 23 October continuation

For Angiolite, an uncertain size numeral with confidence between 0.75 and the existing
acceptance threshold gets one tight crop from its observed token geometry, with a small
white border. At most two numerals are retried per observation. Whole-column retries remain
for merged numerals. A broader whole-column fallback for uncertain individual numerals was
tested and retained as a historical experiment; it did not improve the webcam case.
This is a location cue, not acceptance of uncertain text. The actual retry must reach the
unchanged confidence threshold. Conflicting original readings remain visible; crops remain
a single text source, with independent REF still required. Actual retry crops appear in the
owner review page. `visual.size_token_retry:false` disables only the tight-token fallback
for comparisons; `visual.size_column_retry:false` disables both numeric fallbacks.

Automated evidence capture can use `--no-window` with the sequential runner. With
`--calibrate-first` it automatically measures framing and stops on calibration failure;
without this flag the usual interactive Enter/ESC calibration remains available.

```bat
venv\Scripts\python.exe scripts\validate_box_targets.py --live --camera realsense --config config\box_pipeline_config_realsense_eleven.yaml --calibrate-first --no-window --max-frames 60
```

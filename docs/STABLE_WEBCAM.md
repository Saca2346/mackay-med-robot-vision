# Stable target outlines: webcam candidate, 20 October 2026

From `C:\Users\sacab\Downloads\mackay_vision\code`, run:

```bat
.\FOCUS_WEBCAM_STABLE.bat
```

Choose **2** for Angiolite 2.5 x 29 or **3** for Angiolite 4 x 19. The launcher
uses the separately evaluated profile `config/box_pipeline_config_focus_stable.yaml`.
The original webcam and RealSense profiles remain available through `TEST_TARGET.bat`.

Direct command:

```bat
venv\Scripts\python.exe scripts\run_target.py --camera webcam --config config\box_pipeline_config_focus_stable.yaml --request "angiolite 4 x 19"
```

Keep both ends of the carton in frame and the camera on a firm support. The
outline settles from the first few observations. Pixel tracking then holds it
while the carton remains still and follows measured motion. Losing the target,
an uncertain image, a different track/request, or pressing **O** starts fresh
outline acquisition. Printed internal edges cannot replace a pixel-supported
target solely because the display refinement changes between frames.

Green still requires independent identity evidence. Outline holding affects the
display only; OCR reads current detector crops and the original confidence/date
requirements. Complete barcode/GTIN, LOT, expiry and brand fields remain separate
from identity verification. The view does not emit a retrieval command.

**S** saves another original/annotated photo pair. **Q** closes the recording.
Typing another request in the terminal keeps the models loaded; cached-track
request times are labelled in the log. **O** requests fresh observations.

Each take saves raw and annotated video, actual capture timestamps, CMD output,
photos, OCR inputs, numeric crops and structured observations. `outline_frames.jsonl`
adds the current detector polygon, displayed polygon and tracking mode for every
processed frame. See [TARGET_CAMERAS.md](TARGET_CAMERAS.md) for the common layout.

## Training and evidence

The broader recording dataset contains 2,242 frames and 148,527 generated labels,
with the original 196 reviewed photos retained in training. Model A excludes the
18 October recordings and provides the comparison on a separate recording day.
Model B uses all supplied recording days. Both complete 100 epochs; the last
checkpoint is kept. Final training-set validation numbers are not held-out accuracy.

Generated labels and approximate outline references are assistant/model proposals;
owner review remains pending. B's measurements on its training recordings describe
fit and display stability. New live camera placements require direct testing.

The local evidence, model checksums, training recipes, comparison video, photographs
and replay transcripts are in `results/outline_continuation_20261020/REPORT.md`.
Recorded replays and fresh live-camera timing are labelled separately. Battery
timings are excluded from the charger-powered 1-2 second comparison.

For a different product, select menu **4** and enter the product plus exact size.
The focused fast OCR path currently targets Angiolite; other products use the
existing general reader and need their own accuracy/timing evaluation. RealSense
live validation remains a separate next step.

## Latency continuation (20 October)

The stable profile keeps the full 544 x 960 model B detector. Adjacent name/REF and
size regions can be read in one OCR invocation (`joint_spine_read`). An observed
partial REF can return promptly (`partial_spine_read`) rather than repeating the
full reader immediately. Repeated REF alone remains one source and cannot turn
the target green. Missing product/size fields are never filled from a request.
Periodic full-spine/barcode retries and the original confidence/date gates remain.
After initial verification, detail reads disable the partial early exit so current
full evidence can continue to support the rolling identity history.

Outline edge searches run during acquisition and reacquisition; a pixel-supported
hold follows measured motion. Provisional blue outlines use the current detector.
Crop/panel export and annotated-video encoding run in bounded background workers.
Old-request export results cannot populate the current target panel, including
repeated requests or **O** resets. Video timestamps keep advancing across those
request-timer resets. Closing the
take flushes pending files and reports background failures.

Annotated video is 1280 pixels wide in this profile to reduce encoder load; original
camera recordings, paired photos and reader crops retain their original resolution.
The annotated video has its own `view.mp4.times.csv` with measured frame times.
`frame_latency.jsonl` separates processing, outline, export submission, rendering
and recording costs. `DISPLAY_READY` records when the first green frame is ready
(or submitted to the window); it is distinct from the verification decision and
from `EVIDENCE_SAVED`. It does not measure the monitor compositor's presentation.

See `results/latency_stable_20261020/REPORT.md` for charged, sequential replay
measurements, all individual CMD/video/photo/crop evidence, and their limitations.
These recordings overlap training and do not replace a fresh live acceptance test.

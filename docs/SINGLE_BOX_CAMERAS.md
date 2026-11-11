# Single-carton webcam and RealSense testing

Updated 22 October 2026. Run from `C:\Users\sacab\Downloads\mackay_vision\code`.
Close the previous camera window before starting another capture. Keep the charger connected.

For the eleven-carton RealSense view, numbered target menu and new small date-region retry, see
[REALSENSE_ELEVEN.md](REALSENSE_ELEVEN.md). It shows all detections while reading the selected target.
Both camera commands there now support `--calibrate-first`, automatic isolated calibration names,
and the transparent focused header. Existing calibration references are never overwritten.

## Optional OCR/REF acquisition without barcode or QR decoding

Add `--no-barcode` to `scripts/run_target.py` for either camera:

```bat
venv\Scripts\python.exe scripts\run_target.py --camera realsense --config config\box_pipeline_config_realsense_single.yaml --request "permanent sled bag 2029-03-24" --no-barcode
```

This isolated session disables barcode/QR decoding on fast and full fallback readers, and skips
barcode-region location/crops in evidence exports. Name/date/size OCR, printed REF and GTIN text
remain active. Original source profiles are unchanged. Video, paired photos, actual OCR inputs and
CMD logs still save automatically; S saves an extra pair and Q finalizes recording.

When consistent target text and its required date are read without a second identity source, the
header says `TEXT READ` and CMD records `text frame ready in ...s; identity still unverified`.
Its blue outline is still refined/stabilized. Text readiness is separate from VERIFIED latency.
Sled Bag name + date alone cannot turn green under the existing independent-source rule. Products
with readable printed size text and an independent REF can still verify normally. No PICK/TAKEN,
inventory update or source/date/confidence relaxation is introduced. O starts a fresh reading test.

## Calibration for one carton

Use `--min-boxes 1` for one carton. The default remains three for shelf screening.
Sharpness, movement and frame-margin checks are unchanged. PASS does not certify a decoded
barcode, identity, lens intrinsics or a two-second deadline. Original calibration files are preserved;
choose an unused session name on later runs.

Webcam:

```bat
venv\Scripts\python.exe scripts\kalibrasi_kamera.py --sesi webcam_single_20261022_v1 --config config\box_pipeline_config_all_targets.yaml --min-boxes 1
```

RealSense D455:

```bat
venv\Scripts\python.exe scripts\kalibrasi_kamera.py --sesi realsense_single_20261022_v1 --config config\box_pipeline_config_realsense_single.yaml --min-boxes 1
```

Click the preview, press Enter, and leave the camera untouched during the countdown and measurement.
Show the full product name, requested printed date and code on the same carton. A horizontal carton
can use more of the frame width; leave room around both ends when assessing the complete outline.
Use separate references for shelf and detail views, and for each camera.

## Detection with video, photos and CMD evidence

```bat
venv\Scripts\python.exe scripts\run_target.py --camera webcam --config config\box_pipeline_config_all_targets.yaml --request "permanent sled bag 2029-03-24"
```

```bat
venv\Scripts\python.exe scripts\run_target.py --camera realsense --config config\box_pipeline_config_realsense_single.yaml --request "permanent sled bag 2029-03-24"
```

S saves an extra paired photo; Q finishes the recordings. Each session copies inventory and routes
raw/annotated video, timestamps, reader crops, photos and CMD logs to its own folder under `results`.
No physical retrieval or master-inventory update is authorized.

An optional `--roi x1,y1,x2,y2` selects detections by centre in the original frame, with coordinates
between zero and one. Both calibration and `run_target.py` accept the same option. It does not crop
or enlarge the camera pixels and cannot supply identity evidence. Use the same region only while
the physical camera and target positions remain the same. A selects an area interactively in the
target window if background detections need to be excluded.

## Evaluated limits

The single-carton D455 profile is a separate candidate using the existing YOLO26 model, evaluated
on the owner's 21 October recording. The existing general RealSense profile remains available.
On sixteen sampled D455 frames, the original v4 detector found the assistant-proposed target region
in twelve frames and produced thirty detections outside it; YOLO26 found it in sixteen with zero
outside detections. These regions are diagnostic proposals, not owner-accepted outline labels.
This is not a general shelf accuracy benchmark or a camera-only comparison.

Connected D455 inspection reports USB 3.2 and a maximum advertised colour frame size of 1280x800;
the owner's recording delivered about 30 FPS. The remaining verification problem is small code
detail: a bounded native-pixel probe decoded no GS1 code in its 288 recorded frames. The new live
single-carton test read product name and printed date but remained CHECKING without a second source.
Do not promote that candidate to green by changing thresholds or copying catalogue/reference fields.

The owner's webcam run reached green with barcode + text at 2.478 seconds. A powered replay reached
2.484 seconds. Neither meets the two-second target. Preserve capture-to-display and replay timing
labels; exclude model startup only where explicitly stated.

Evidence: `results/dual_camera_fix_20261021/REVIEW.html` and `REPORT.md`.

## Visible-outline update

The 22 October outline update refines and stabilizes a selected blue CHECKING target as well as a
verified target. Pixel-supported end corrections can shorten an overlong detector polygon. The
two-pixel antialiased stroke is inset three source pixels to keep its ink inside the estimated edges.
Calibration uses the same inward display convention. Original detector geometry still supplies
tracking, OCR and verification, and blue never becomes green solely because its outline is stable.

Uncertain, occluded or frame-clipped ends are not reconstructed. Restart an already-running camera
program to load the changes. Same-frame top/bottom comparisons and replay proof are under
`results/outline_edges_20261022/REVIEW.html`. These are visual diagnostics, not independently labelled
full-outline ground truth or proof of a two-second identity deadline.

## Sled Bag acquisition speed update

The later 22 October update reads the complete spine once for products without size fields (currently
Permanent Sled Bag), instead of several label orientations followed by tiled spine OCR. It keeps actual
name/date tokens and decodes the original spine code. Existing confidence, catalogue, two-observation,
source and expiry gates are unchanged. Every fourth attempt and post-verification detail reads retain
the full reader; an unreadable product name also falls back. Exact-size products use their existing reader.

Three fresh-process powered webcam replays reached the first rendered green frame in 1.306422,
1.058592 and 1.135472 seconds, all with barcode + text and no subsequent verification loss during the
45 processed frames. The wrong-date control produced zero green. The baseline acquisition trial on
the same recording did not verify within its 45 processed frames; previous successful runs were over
2 seconds. All outcomes remain in `results/target_speed_20261022/REVIEW.html`.

These are recorded-input measurements after model startup, not a universal live deadline or a
complete-detail certificate. Latest RealSense replay selected the target at 0.758 seconds but stayed
CHECKING because its code was unreadable. A larger, sharper same-carton code view is still needed.
Restart the application, use the detection commands above, keep the charger connected and camera
still, then check the CMD line `green frame ready in ...s` and its `within/over 2s` result. S/Q retain
paired photos and recordings. Do not interpret a blue CHECKING time as successful verification.

## D455 detail-focus finding

The horizontal `realsense_detail_20261022_v3` original photo improved the apparent label size, but its
DataMatrix is still blurred and no valid code decoded in ten bounded original/label/code/spine probes.
Its global sharpness755PASS is not a per-code readability check. Read-only inspection of the connected
D455 RGB sensor reports no focus control, consistent with the manufacturer's fixed-focus RGB sensor:
[D400 datasheet, section 3.6.3/table 3-19](https://realsenseai.com/wp-content/uploads/2024/10/Intel-RealSense-D400-Series-Datasheet-October-2024.pdf).
Depth minimum distance must not be presented as a measured RGB focus limit.

Keep the carton horizontal, remove bright background distractions, provide diffuse front/side light
and compare native code photos while moving away from the current close position in approximately
5cm steps. Record actual lens-to-label distances. This is a proposed distance sweep, not a certified
distance range. Choose clarity and actual decode rather than apparent zoom or global sharpness alone.
Then run a fresh O/reset timing trial from a still position. No autofocus setting, sharpening filter
or calibration PASS can replace an independently decoded code. Diagnostic crops, original input hash,
native OCR results and supported RGB controls are in `results/realsense_focus_20261022/REPORT.md`.

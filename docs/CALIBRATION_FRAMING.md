# Camera framing and visible carton ends

Single-carton update, 22 October: pass `--min-boxes 1` to avoid the shelf check's default minimum
of three. Optional `--roi x1,y1,x2,y2` selects the same original-frame area in calibration and target
testing. See [SINGLE_BOX_CAMERAS.md](SINGLE_BOX_CAMERAS.md). All image quality and identity gates remain separate.

Updated 21 October 2026. Applies to the focused webcam view and the camera-position checker.

`kalibrasi_kamera.py` screens image quality and position. It does not solve lens intrinsics/distortion, identify a
camera by its Windows index, certify barcode readability, or establish a physical camera-to-box distance.
Sharpness is measured in the central half of the image and does not certify every label's focus.

## Findings from the owner's 21 October session

The 17:36 calibration card reports 11 cartons at 1920 x 1080 with median short-side width 46.6 px. A later original
camera photo in `target_webcam_20261021_174101_331983/take_174507_787687` contains 11 cartons at the same image
resolution, with median detected width 57.3 px using the same selected model. The pixel size differs by about 23%.
The earlier 20 October recording contains 20 cartons and is a different scene. Neither comparison establishes
physical distance or a camera switch. The old calibration card did not save camera settings or an unannotated frame.

Two software presentation differences also existed: calibration displayed the full-resolution picture directly,
whereas focused view fitted the picture plus detail panel to a bounded window; calibration drew raw polygons while
the focused VERIFIED view refined visible edges. No digital zoom or image crop was found in either input path.
The detail panel still takes display space, so apparent size in different layouts is not a reliable distance gauge.

The former PASS checked left/right margins and allowed a tolerance when counting clipped boxes, but did not
explicitly check top/bottom margins. It also placed the title/checks over the image, hiding some carton tops. That
PASS was image screening, not proof that every full carton boundary or small barcode could be read.

## Corrections

* Both previews use the same fitter, preserving all pixels' field of view and aspect ratio within 1280 x 720.
  Saved calibration photos put their header outside the image. At the owner's request, live calibration overlays
  a transparent status panel (18% dark background / 82% image) with outlined text, preserving the full viewport.
  Calibration uses blue detected outlines, not green VERIFIED styling.
* Calibration uses the same supported-edge display helper as focused target view. Metrics/OCR retain the original
  detector geometry. No global baseline is imposed on cartons of different heights or positions.
* Each proposed end is checked against the actual image independently. A false end in padded pixels outside the
  image no longer cancels a supported correction at the opposite visible end. Neighbour/overlap guards remain.
* Calibration warns when top/bottom margins are under 3%, when the driver returns a different requested resolution,
  or when a same-position reference lacks matching camera settings. Different frame sizes/carton counts no longer
  produce a pixel-scale comparison that could be mistaken for a distance change.
* New cards save original PNG pixels and their hash, raw/refined polygons, config hash, model settings, configured
  backend/index, and actual frame size. Windows index is explicitly not certified physical-device identity.
  Existing calibration evidence is preserved: choose a new session name instead of overwriting it.

## Repeatable capture

Keep the camera on a fixed support and leave space around all four ends of the desired cartons. The later live
photo has targets near the upper edge; aim slightly higher to bring their tops fully into the frame. Check the
visible blue edges against the physical cartons. Do not add digital zoom to imitate a previous screenshot.

Run from the `code` folder:

```bat
venv\Scripts\python.exe scripts\kalibrasi_kamera.py --sesi webcam_detail_20261021_v2 --config config\box_pipeline_config_all_targets.yaml
FOCUS_ALL_BOXES.bat
```

For a subsequent session with the same physical camera, resolution, cartons and position:

```bat
venv\Scripts\python.exe scripts\kalibrasi_kamera.py --sesi webcam_detail_20261021_v3 --config config\box_pipeline_config_all_targets.yaml --ref results\data_rak\webcam_detail_20261021_v2_kalibrasi.json --same-position
```

Choose another unused name on later runs. Keep JSON, annotated JPG, original PNG and the recording together.
Exit code 0 includes WARN; read every check. A fresh green identity still requires two independent sources and
any requested printed date. Whole-field completion, physical retrieval and timing are separate checks.

Evidence and before/after pixels: `results/calibration_outline_20261021/REVIEW.html`. Its preview of the saved live
photo is explicitly offline; it is not a replacement for a new live calibration.

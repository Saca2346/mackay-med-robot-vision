# Shelf data protocol (RealSense D455)

Goal: training and test data from the camera that the robot really uses. Until now the box model was trained only
on 163 phone photos (4080 × 3072); it has never seen a D455 frame.

## 1. Minimum conditions per frame

Every number is a measurement from our own recordings (evidence: `results/bukti_arsitektur_20261009/`). Where a
value is a guess, it says so.

| Condition | Value | Basis |
| --- | --- | --- |
| USB | USB 3.x | on USB 2.1 the D455 sends no colour frames at all |
| Box width in the image (short side of the box, px) | **< 28 fails; 28–40 works for iVascular; ≥ 40 turns the whole-spine read on** | 08-10 11:18: **24 px** (19 boxes in view) → size text read in **7 %** of readings, no PICK. 08-10 18:28: **36 px** (20 boxes still in view) → **34 %** and PICK for iVascular. 09-10: 33.6 px → PICK in 17 s. 40 px is only the config value `full_box_min_width_px`; **no live test reached it yet** |
| Boxes in the frame | the whole shelf was in view in every run so far | what changed between the 7 % and the 34 % runs was the camera distance, not the number of boxes. A "10–12 boxes" framing (≈ 50–60 px by geometry) is a **guess, not measured** — session `s2_mid` is the first test of it |
| Sharpness (HUD "sharpness") | ≥ 300; good runs were 700–850 | below 100 the program reports a blurry image |
| Camera | still, on a tripod | frames with camera motion get no OCR at all |
| Target position | away from the left/right image edge | a box put back at the left edge (x ≈ 25 % of the width) took ~200 s to read; mid-frame: 17–19 s |
| Next to the shelf | shelf post and monitor outside the picture, or a shelf area with one box width of margin | both were counted as boxes (22 instead of 20) |
| Light | no glare on the labels | glare erases the size text |

Second source (REF / barcode): only iVascular prints the REF inside the size label, so only iVascular reaches PICK at
shelf distance. For Terumo, Boston Scientific, APT and OrbusNeich the REF / barcode was **never** read (80 whole-spine
reads at 24 px forced, 0 REF). The close-up distance that makes them readable is still to be measured.

## 2. Camera calibration at the start of EVERY session

Before recording, `UJI_REALSENSE.bat` → **D** runs `scripts/kalibrasi_kamera.py` (live preview while you adjust):

| Check | PASS | Advice when not |
| --- | --- | --- |
| Boxes found | ≥ 3 | aim at the shelf, check the light |
| Box width | ≥ 40 px (WARN 28–40, FAIL < 28) | move closer |
| Sharpness | ≥ 300 | refocus, change the distance, check glare / lens |
| Camera still (3 s) | frame change < 3 | tripod; something moved in the picture |
| Camera roll | boxes lean < 3° (WARN < 8°) | level the camera |
| Shelf inside the frame | ≥ 3 % free left and right, no box cut | centre the shelf |
| Same position as the reference | shift ≤ 25 px, scale ± 8 %, roll ± 2° | put the camera back (only with `--ref … --same-position`) |

It saves `results/data_rak/<session>_kalibrasi.json` (every number + the verdict) and `…_kalibrasi.jpg` (the frame with
the boxes it measured). **Keep both next to the recording**: they document under which camera conditions the data was
taken, and they let a later model result be traced back to the camera position. A FAIL does not stop you (the
menu asks), but a recording after a FAIL is marked in the card and should not be used for the test set.

To repeat a camera position between two sessions (for a fair comparison), calibrate the second one against the first:
`venv\Scripts\python.exe scripts\kalibrasi_kamera.py --sesi s3_close --ref results\data_rak\s2_mid_kalibrasi.json --same-position`
The shift is measured on the outer edges of the shelf; if the two views hold different boxes the program says
"different part of the shelf" instead of inventing a number.

## 3. What to record (sessions)

One named session each (1–2 minutes, camera still for a few seconds at each position, moved slowly in between):

| Session | Camera | Shelf |
| --- | --- | --- |
| `s1_far` | whole shelf (20 boxes) | normal order |
| `s2_mid` | closer: aim at box width ≥ 40 px, left half then right half | normal order |
| `s3_close` | 3–5 boxes, slide along the shelf | normal order |
| `s4_order` | as `s2_mid`, calibrated with `--ref s2_mid --same-position` | order changed (swap 5–6 boxes) |
| `s5_hard` | as `s2_mid` | boxes leaning, one lying, one upside down (180°), one slot empty, a hand reaching in |
| `s6_angle` | as `s2_mid`, camera 15–20° from the left and from the right (roll stays level) | normal order |
| `s7_test` | as `s2_mid` | a new order, **kept out of training** (honest test set) |

Videos go to `results/data_rak/<session>.mp4` (raw, for training) and `<session>_vis.mp4` (annotated, for reports).
Never delete a session that went badly: it is still evidence.

## 4. From video to labelled photos

1. Extract frames — laptop, cmd, folder `medicine_shelf_vision`:
   `venv\Scripts\python.exe scripts\ambil_frame_data.py results\data_rak\s1_far.mp4 results\data_rak\s2_mid.mp4`
   (one frame every 3 s, blurry and unchanged frames dropped; output `prelabel_size_label_full/raw_d455/`).
   A still shelf gives very few frames — that is intended.
2. Pre-label with the current model (`tools/predict_pool.py` → `tools/pred_to_spec.py`), review each photo
   (box, size_label, expiry_label), then train on the server.
3. One recording = one burst (frame names carry the real time), so a session is never split between train and
   validation.

Target: about 150–300 kept frames from s1–s6; s7 stays a separate test set.

## 5. "Accurate, precise and fast" — acceptance targets (proposal)

| Measure | Target | Today |
| --- | --- | --- |
| Boxes found on held-out D455 frames (s7) | ≥ 98 % | not measured yet (phone photos, out-of-fold: v5 96.7 %, v6 95.1 %) |
| Wrong MATCH | **0** (hard rule) | 0 in every evaluation so far |
| Detection time on the laptop (960 model) | ≤ 110 ms / frame | 52–110 ms |
| Request → PICK, iVascular | median ≤ 30 s | 7–55 s |
| Request → PICK, other brands | decided after the close-up test | no PICK at shelf distance |

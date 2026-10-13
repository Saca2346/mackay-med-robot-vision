# BOX + SIZE_LABEL + OCR pipeline - how to use

This addition does not change the old files (the book & medicine detector / verifier / test_webcam versions are still there).
All new files start with `box_`. (Indonesian original: `docs/id/BOX_PIPELINE.md`.)

| File | Purpose | Runs on |
|---|---|---|
| `config/box_pipeline_config.yaml` | every tunable number: model, thresholds, OCR, product catalogue | - |
| `data/box_classes.txt` | class order: 0 = box, 1 = size_label | - |
| `src/box_geometry.py` | polygon IoU, rotated crop made straight, label file reading | - |
| `src/box_detector.py` | runs the model (.onnx / .pt), pairs each size_label with its parent box | - |
| `src/box_ocr.py` | reads the text of a size_label crop (RapidOCR, Tesseract as fallback) | - |
| `src/box_verifier.py` | text -> product / diameter / length -> MATCH / IGNORED / CONFIRM | - |
| `scripts/check_box_labels.py` | finds loose labels (boxes overlapping each other) | server / laptop |
| `scripts/export_box_onnx.py` | best.pt -> models/box_obb.onnx + CPU speed test | server |
| `scripts/test_webcam_box.py` | live webcam / photo test: stage 1 (detection) and stage 2 (--request) | laptop |
| `scripts/eval_box_manual.py` | TP / FP / FN table + precision / recall on test photos | laptop / server |
| `tests/test_box_*.py` | unit tests (geometry, parsing, decisions, mentor cases A/B/C) | - |

## 0. Install (once, on the laptop and on the server)

```bash
cd medicine_shelf_vision
source venv/bin/activate            # the existing venv
pip install -r requirements_box.txt
python -m pytest tests -q           # all must pass (268 on 14-10)
```

## 1. Server - check labels, retrain, export

```bash
# 1a. find photos with loose labels (straight boxes on tilted packages)
python scripts/check_box_labels.py --labels <labels folder> --images <images folder>
#     -> results/label_check/label_check.csv + pictures of the photos that need redrawing

# 1b. after fixing the labels: retrain (use the same data.yaml as the bootstrap run;
#     see the "data:" line in runs/bootstrap_v.../args.yaml)
yolo obb train data=<data.yaml> model=yolo26n-obb.pt imgsz=640 epochs=150 batch=16 device=0 project=runs name=obb_v2

# 1c. export to ONNX + CPU speed test
python scripts/export_box_onnx.py --weights runs/obb_v2/weights/best.pt --image <one test photo>
```

Note: the existing bootstrap model can also be exported directly (step 1c only) so laptop tests can start today.

## 2. Move the model to the laptop

```bash
# from a laptop terminal (the server must run SSH: sudo systemctl enable --now ssh)
scp Zen@hucenrotia-ai:<project path on the server>/models/box_obb.onnx models/
# if the host name is not known, replace hucenrotia-ai with the server IP (check: hostname -I on the server)
```
Fallback if SSH does not work: upload `box_obb.onnx` to Google Drive from the server and download it on the laptop.

## 3. Laptop - webcam test

```bash
# Stage 1: detection only. Put 4 boxes of the same size + 7 items in front of the webcam, count x/11 and note the FPS.
python scripts/test_webcam_box.py

# Stage 2: choose the box for a request, plain sentences are fine
python scripts/test_webcam_box.py --request "pilih ivascular angiolite diameter 4 mm panjang 19"
python scripts/test_webcam_box.py --request "angiolite,2.5,29"

# from photos (no webcam), results saved to results/screens/*_result.png
python scripts/test_webcam_box.py --source <photo folder> --request "accuforce,2.75,20" --no-window
```
Keys: `q` quit, `s` save the raw + annotated frame (for slides), `l` log the decision to `results/selection_log.csv`,
`o` force OCR again, `r` the same request once more, `b` detailed scan of one box.

**Requests** are understood as plain sentences: "pilih ivascular angiolite diameter 4 mm panjang 19", "terumo accuforce
2,75 x 20", "nagomi 3/24", "angiolite 2.5 L29", "boston scientific permanent sled bag", and since 14-10 with a date for one
specific box: "permanent sled bag 2029-03-24". A brand may be written (it must match the product); numbers without a keyword =
diameter, then length. The program prints how it understood the request ("request understood as: angiolite (ivascular),
diameter 4 mm, length 19 mm"); incomplete or implausible requests are refused with an explanation. While the program runs,
a NEW request can be typed in the terminal + Enter: the whole shelf is decided again for that request.

**Only boxes ON THE SHELF can be PICK**: the box must be still relative to the other boxes (not held / pulled out) and must not
cover other boxes (not in front of the shelf). In the test video of 2026-09-28 13:06 a box that had been taken and was held in
front of the shelf was chosen again and its slot stuck to another box; now such a box is refused ("not on the shelf").
Measured on w300 recordings: with a still camera and no hands, 100 % of shelf boxes pass this rule.

**Detailed scan (key `b`)**: bring ONE box close to the camera (any side, also the back), hold still, press `b`.
All codes are read - 1D GS1-128 barcode (Terumo), DataMatrix / QR (Boston Scientific, also without brackets and the QR
"GS1 Digital Link"), plain QR (web address) - plus all text (OCR at several scales). Data card: GTIN + catalogue content
(verified / DRAFT), expiry, LOT, serial number, REF -> size, product & size from the text, and the decision against the
request. Size numbers read differently between OCR passes are marked doubtful (not used). Saved to
`results/screens/pindai/*.jpg + .json`.

**A barcode is used for MATCH only when its GTIN is in the catalogue AND verified by a person** (`data/gtin_catalog.csv`).
Terumo (small size text) and Boston Scientific (no size-coded REF) depend heavily on this:
`python scripts/enroll_gtin.py --review` to check drafts (the checker's name is asked in the terminal), `--camera auto` to
register a new box (press `e` when the barcode is visible), `--source <photo folder>` for barcode photos from a phone (sharpest),
`--belum` = list of products / sizes without a GTIN yet, `--review --semua` = also review verified entries (option `e` changes
product, size, brand and REF / package mark), `--ganti-pemeriksa "OLD NAME"` (the new name is asked). Examples copied
literally ("nama anda", "NamaAsliAnda", `<...>`) are refused.
REF in the catalogue is only information (not used for decisions). Boston Scientific: REF is filled with "MR" (the word
MONORAIL at the end of the spine, the same for every size) - the size is confirmed through the GTIN in the DataMatrix / QR +
the size text. The catalogue is changed ONLY on the laptop, then sent to the server each time it changes
(`scp data\gtin_catalog.csv Zen@<server>:~/Downloads/box_pipeline/data/`). The server zip package does NOT contain the
catalogue, so unzipping a package never brings back an old catalogue.

Box colours in stage 2:

| colour | meaning | picked? |
|---|---|---|
| thick green + `PICK` | the requested box, verified (>= 2 independent sources, 2 agreeing readings) | yes, only this box |
| thin green `MATCH` | also matches, but another matching box expires earlier (FEFO) | later |
| light blue `CANDIDATE` | only 1 matching source so far (e.g. size text only); for a dated request also: date not read yet | no, bring it closer / show the barcode |
| yellow `CONFIRM` | doubt: contradicting sources, unreadable, too far | no, the system does not guess |
| grey | certainly not the target (product OR size read differently) | no (ignored) |
| white `reading...` / `match? 1/2` / `recheck` | not enough readings yet / box number was doubted | not yet |
| purple `EXPIRED` | expiry date (barcode) has passed | never |

### Selection flow (`src/box_session.py`)

The second HUD line shows the step (green = found / done, red = not found / shelf empty, yellow = needs confirmation,
white = working):

1. **CHECKING SHELF** - point the camera at the WHOLE shelf and hold still. 0 boxes for 2 seconds -> **SHELF EMPTY** (red),
   no OCR. Boxes visible and stable for 1.5 seconds -> the number of boxes on the shelf is logged (`BOX COUNT`).
2. **SEARCHING** - every box is read. The requested box verified -> **FOUND** (green, `PICK #n`, the other 19 boxes are
   ignored). **NOT FOUND** (red) only when EVERY box visible NOW is certainly not the target AND there are at least as many
   as on the shelf, so move the camera back until the whole shelf is visible. A single doubtful / unread / unseen box ->
   **NEEDS CONFIRMATION** (yellow) with its box number; the system never guesses "not found".
3. **PICK -> being taken -> TAKEN** - the slot of the PICK box is locked to its place ON THE SHELF (follows camera motion and
   the other boxes, not the box being pulled out). A box moving out of its slot = "being taken". Once out, the system makes sure
   the slot is empty for 2 seconds (slot fully visible, camera still, no box in the slot, the slot image differs from that box and
   is still, so a passing hand or a detector missing one frame does not count). Then that box number is dropped.
   While FOUND, the PICK box is re-read first; partial / empty readings (hand covering, blur) do not cancel the lock, one
   contradiction -> yellow (DO NOT take) and read again, two contradictions -> PICK CANCELLED.
   **RECHECK** (yellow, "PICK? rechecking"): the PICK box number is doubted (a hand reaches / covers the box, the box is bumped),
   another still box with a different number sits in its slot (the number changed while covered), or the look of the box in the
   slot changed. The lock is KEPT but the robot must not pick; only NEW readings count: a reading that names exactly the requested
   product + size (full text, REF, barcode or GTIN) with no contradiction -> **PICK CONFIRMED** (green again). "angiolite" alone
   is not enough; the sibling box (angiolite 4x19) contradicts through its size -> CANCELLED. Not confirmed within 10 seconds while
   the box is in its slot (`relock_s`) -> PICK CANCELLED. A box pulled out during RECHECK is followed until TAKEN (before: CANCELLED
   exactly when the nurse reached for the box, and TAKEN was never logged). A neighbour box that tilts / shifts into the gap of the
   taken box (read twice as not the target, PICK box no longer visible, and there was movement: the PICK box was pulled or the
   neighbour shifted >= 0.5 x width) -> TAKEN; without movement -> normal PICK CANCELLED.
   Final check since 10-10, "moved or taken?": TAKEN only if the camera is still, the target is read nowhere and the slot no longer
   looks like the picked box; otherwise cancel and search again.
   Box numbers (tracker): a box whose centre stays in place (sideways shift <= 0.3 x width, along <= 0.15 x length, not ambiguous)
   keeps its old number even when its detected angle / length changes. A spine tilted only 6-8 degrees already gives IoU < 0.5, and
   earlier its readings were dropped every time that happened (tripod test 30-09: PICK 39.5 s). This rule is used only while the
   camera is calm: measured motion < 0.1 x box width for 5 frames in a row, and pairs that certainly did not move together. The
   first version (motion < 0.5 x width) moved 3 box numbers to the neighbouring box in a replay with a hand-held camera. While the
   camera moves, the old IoU rule applies.
   **Upside-down box (180 degrees):** the RapidOCR direction classifier flips long lines (REF, date) but not short numbers, so
   "2,5" and "29" on an upside-down box were read "25 29" / "62 55". When a reading is incomplete, the size label is also read
   upside down; if that direction reads the text much more clearly, the normal-direction numbers are dropped. While searching, only
   for boxes whose product / REF matches the request or that so far only had numbers read; while scanning the shelf, all boxes.
   Tripod replay 30-09 (Angiolite 2.5x29 put back upside down): without this no PICK in 91 s, with it PICK in 10.1 s and it stayed
   locked; OCR +22 % per box. eval_verify: 0 wrong MATCH out of 1353, correct MATCH 14 -> 16 of 123.
4. **The same request again** (`r` or `--qty 2`) - ANOTHER box is searched; a slot already empty is never chosen. None left ->
   **NOT FOUND** "none left". An empty slot filled again within 5 seconds -> warning.
5. **Orders** (`src/box_order.py`, in `--request` or typed in the terminal while running): `angiolite 2.5 29` one box;
   `2 kotak permanent sled bag` (or `jumlah 2`) a quantity; `angiolite 2.5 29; accuforce 2.75 20 jumlah 2` several items
   (separators `;` `+` `dan` `serta` `lalu`); `+ essential pro 3 40` adds to the queue; `lewati` skips to the next item. A line
   without `+` REPLACES the order. The next item starts automatically after DONE, or after NOT FOUND has held for 3 s; NEEDS
   CONFIRMATION waits for `lewati`. When the request changes, the shelf readings are NOT dropped: each box's evidence is decided
   again for the new request (`TrackState.redecide`), and OCR readings finishing after the switch are judged against the new
   request. One unclear item cancels the whole line (no item silently disappears).
6. **Shelf area** (tripod camera): key `a` in the video window -> drag a box around the shelf -> Enter. Saved to
   `results/area_rak.json` and reused automatically for the live camera (not for old recordings); `--roi none` turns it off,
   `--roi x1,y1,x2,y2` (fractions 0-1) sets it directly. Detections outside the area are ignored: in the 30-09 test a paper box
   and a GPU box next to the shelf counted as boxes (24 instead of 20); with the shelf area the same recording gave 19.
   Camera moved -> choose the area again. A warning "SHELF AREA CUTS n BOX(ES)" appears when the area cuts through a box.
7. **Round iVascular diameters** ("4" then "19", the unit mm on another line): `Identity.diameter_hint`, may only AGREE with the
   requested diameter, never contradict it (a "3,5" read as "3" does not make a 3.5 box "not the target"). Text alone stays
   1 source. eval_verify: 0 wrong MATCH out of 1353.
8. **Filtered misreadings** (replay 30-09): numbers stuck to letters are part of a code, not a size (REF "SCCDSR1415..." read as
   "5CCD5A1415..." next to "mm" once became diameter 5 and cancelled the PICK); "4mm", "19rnm", "2.5x29" are still read. A REF with
   a diameter that is not a multiple of 0.25 mm ("259" = 2.59) is dropped. Misread lengths in the REF (about 3 % of readings, e.g.
   4x19 read as 4x10) still count as a contradiction: safe, only slower. `--debug-readings` logs the raw OCR text (column `teks`:
   label || REF || spine) to find the origin of the next misreading.

Same brand and name, different size (e.g. Angiolite 2.5x29 vs 2.5x24 vs 2.25x29) never MATCH: product AND diameter AND length
must be exactly equal from 2 independent sources (size text + REF code, or a verified barcode); one source with another size =
grey / yellow. Evaluation on 123 boxes: 92 "same product, different size" requests, 0 wrong MATCH.

Is the number of boxes on the shelf known from the inventory? Use `--expected 20`: if the detector sees only 18, the system does
not conclude NOT FOUND. Every event (BOX COUNT, PICK, TAKEN, NOT FOUND, ...) is logged to `results/verify_log.csv` (or the session
folder, see `docs/TEST_SESSION_GUIDE.md`) and its picture is saved to `screens/peristiwa/` (audit evidence).

Laptop default: OpenVINO model on the Intel iGPU (`intel:gpu.0`); the current RealSense config uses `box_v4s_960_openvino_model`.
The camera is read in its own thread (only the newest frame is used), and on live sources OCR runs in a background thread, so the
window is never "Not Responding" while OCR reads. The end of every run prints a **Summary** (real FPS, detection ms, iterations
> 1 s, OCR ms per crop) for comparison.

## 3b. Run on the GPU server (viewed through Remmina) - if the laptop is choppy

(Superseded for daily work by `docs/REMOTE_SERVER_CLAUDE_CODEX.md`; kept for reference.)

The webcam stays on the laptop. The laptop only sends images (1080p MJPEG stream through an SSH tunnel, open only on
127.0.0.1); the server (RTX A5000) detects, tracks, reads OCR (3 threads) and runs the selection flow; the result is viewed on
the Remmina desktop. Port 8090 on the server is used by another service, so on the server the stream is on port **18090**. The
camera can be used by one program only (close test_webcam_box.py on the laptop).

Once (or for every new package):

```bash
# laptop, cmd (enter the server password): send the package
scp C:\Users\sacab\Downloads\box_pipeline_server.zip Zen@140.113.149.94:~/Downloads/
# server, terminal on the Remmina desktop
mkdir -p ~/Downloads/box_pipeline && cd ~/Downloads/box_pipeline && unzip -o ~/Downloads/box_pipeline_server.zip
source ~/trse_venv/bin/activate
pip install rapidocr onnxruntime zxing-cpp
python -m pytest tests -q
```

For each test (3 windows, leave them all open):

```bash
# laptop, cmd 1: send the webcam (prints the camera FPS and Mbit/s every 5 seconds)
venv\Scripts\python.exe scripts\stream_webcam.py
# laptop, cmd 2: tunnel (enter the server password)
ssh -N -R 18090:127.0.0.1:8090 Zen@140.113.149.94
# server, Remmina terminal
cd ~/Downloads/box_pipeline && source ~/trse_venv/bin/activate
python scripts/test_webcam_box.py --config config/box_pipeline_config_server.yaml --source http://127.0.0.1:18090/video.mjpg --request "angiolite,2.5,29" --expected 20 --record results/uji_server.mp4
```

- The display window on the server is scaled to 1280 wide (`output.view_width`) so RDP stays smooth; detection, OCR and evidence
  pictures keep full resolution. Remmina: quality "Good" is enough.
- Top HUD line: `shown X fps` (frames really processed), `input Y fps` (frames arriving from the laptop), `process max Z fps`
  (speed of one iteration without waiting). Low "input" = the laptop -> server network is saturated.
- Server OCR: 3 engines in parallel, each with its own share of cores (`ocr.threads` 0 = automatic). Before, each engine used all
  cores and they competed (measured 0.7-1.9 seconds per box). OCR on the GPU (optional): `pip uninstall -y onnxruntime && pip
  install onnxruntime-gpu`, then `ocr.device: "cuda"`; if the GPU cannot be used it falls back to the CPU. (On the laptop MX350 the
  CPU was faster: 79 vs 235 ms per label, see `docs/FIXES_LOG_20261008-11.md`.)
- Stream default 1080p JPEG 80, max 15 fps (about 25 Mbit/s). Slow network (cmd 1 prints FPS < 15, or the server image lags):
  `--quality 70` or `--fps 10`. Do not lower the resolution if possible: the size text needs 1080p.
- Server test without a webcam using the same recording: laptop cmd 1
  `venv\Scripts\python.exe scripts\stream_webcam.py --video results\uji_tahap2.mp4 --video-fps 10.7 --start 220`
  (new recordings from `--record` have a `.times.csv`, so `--video-fps` is not needed).
- Tested on the laptop with a w300 recording through the http stream + 3 OCR threads: box count, PICK on the right box, `--record`
  on the receiving side complete with the time of every frame.

## 3c. External webcam (HP w300, 1920x1080)

Config `camera.index: "auto"` picks the camera with the largest resolution (w300 = 1920x1080 30 fps via MSMF + MJPG; the laptop
camera is only 640x480), so `--camera` is not needed. Force it: `--camera 0` (w300) / `--camera 1`.
Measured on the laptop (i3-1115G4, iGPU), real 1080p frames: model v2m 960 = 7.1 real FPS, 1280 = 3.8 FPS.
OCR and barcode always use crops from the full 1080p frame (not from the small detector image).
The w300 has fixed focus: the HUD shows `sharpness N` (centre of the frame). Move the box distance until the number is highest
(below ~100 = blurred, the size text will not be read). `stream_webcam.py` also uses the w300 at 1080p (max 15 fps, about
35 Mbit/s; slow network: `--width 1280 --height 720`).

### Stage 2 with a hand-held camera

- Box numbers (`#12`) follow the camera motion (optical flow, `src/box_motion.py`), so OCR readings already collected for that box
  are not lost when the camera moves. When in doubt (one detection covering two boxes, motion not measurable), the box gets a new
  number: readings are lost, but they NEVER move to the neighbouring box.
- Frames taken while the camera moves (> `ocr.max_motion_px` px per 1/30 second, measured with real time) are not OCR'd; the HUD
  says `CAMERA MOVING - hold still`. The final summary prints the percentage of moving frames. Unmeasurable motion (heavy blur):
  boxes in that frame are not matched to old numbers; more than `ocr.max_lost_frames` frames -> all numbers are forgotten and the
  boxes are read again (safe, no mix-up).
- Boxes that leave the screen are remembered (their position is shifted along), so the shelf can be checked part by part up close.
  Back on screen: matched only if it fits and is not ambiguous; a box that matched before must be read again (`recheck`) before it
  may PICK.
- Doubted box numbers: on w300 recordings a box number sometimes moved to the NEIGHBOURING box after the camera moved far / away.
  Each time a number-box pair is weak (IoU < 0.5, through the loose rule, or back on screen), ALL old readings of that box are dropped
  (also OCR results of old crops that finish late) -> `recheck`, 2 new readings needed. On w300 recordings 3 of 3 such moves through
  weak pairs were caught by this rule.
- A size_label is used only for the box that contains its centre (config `grouping.min_label_overlap`); labels outside a box are no
  longer paired with the nearest box, because on a shelf that is almost always the neighbour's label.
- OCR gives priority to boxes that are not yet certain and closest to the CENTRE of the screen. How to test: aim the target box at
  the centre, hold still 2-3 seconds per position (laptop OCR about 1-3 seconds per box).
- `#5 match? 1/2` = only 1 of the 2 needed readings so far (not usable yet); `...` = being read.
- `--record results/uji.mp4` saves ALL raw camera frames (no annotation) in its own thread, plus the real time of every frame in
  `results/uji.mp4.times.csv` (cameras often deliver less than 30 fps). Replay with `--source results/uji.mp4`: frames are skipped
  like a live camera using their real timing (`--all-frames` = every frame). Old recordings without `.times.csv`: give the real fps,
  e.g. `--video-fps 10.7`. These frames are also close-range training material (do not use screen recordings: the HUD is in them).
  `--record-vis` saves the ANNOTATED video (boxes, labels, status) for reports.

## 3d. Shelf inventory (preparation before surgery) - scripts/rak_inventaris.py

Operational flow: BEFORE the day of surgery the shelf is mapped and checked by a nurse; DURING surgery the requested box is already
known and only that box is confirmed again.

```bash
# (optional) an existing list of the shelf content: CSV columns produk,diameter,panjang,jumlah[,gtin,lot,kedaluwarsa]
python scripts/rak_inventaris.py impor daftar_isi_rak.csv
# 1. scan: camera STILL (tripod) in front of the shelf, wait until the boxes are green, q = save
python scripts/rak_inventaris.py pindai
# 2. a nurse checks each box against the real box (y correct / e edit / n delete); the checker's name is logged
python scripts/rak_inventaris.py cek
python scripts/rak_inventaris.py lihat
# 3. surgery: requests are searched through the inventory
python scripts/test_webcam_box.py --inventaris data/rak/inventaris.csv --request "angiolite,2.5,29"
```

- Box identity while scanning: CERTAIN only when >= 2 independent sources agree (barcode 2 / size text 1 / REF 1 / GTIN text 1) over
  >= 2 readings with no contradicting reading; text pieces from several readings may be combined. One source only = PROPOSAL (check
  carefully). Everything stays a draft until checked by a nurse; during surgery ONLY `terverifikasi` (verified) entries are used.
- During surgery: matching inventory boxes are sorted FEFO; every box on screen is compared with the spine look of that box (colour
  per section, both directions) -> similar ones are read first ("looks like K03"). Old positions are NOT used, so a moved box is
  still found. PICK still only after a new verification. TAKEN -> the entry is marked `diambil` (taken). HUD and summary show the
  seconds from the request to PICK.
- Rescan = map the whole shelf again (the old inventory is backed up). `--lanjut` = the next part of the shelf.
- The camera must be still while scanning: every time the camera position is lost all box numbers restart; duplicate records are
  merged automatically (same identity + look, one per segment), the rest are marked "maybe the same box".
- Files: `data/rak/inventaris.csv`, `data/rak/punggung/Kxx.jpg`, `data/rak/lembar_cek.jpg` (print for the nurse).

## 3e. Stress test (all boxes in turn, changed arrangements, random requests)

`scripts/stress_plan.py` builds a plan from the inventory: (1) all boxes in turn, (2) N arrangement changes, each changing K boxes
(other slot / reversed / lying / other side / tilted) and requesting them again, (3) random requests. `test_webcam_box.py --rencana`
runs it: each item gets a time limit (`--waktu-per-barang`, default 90 s); no PICK yet -> skipped and logged. Items are not taken
(a PICK held for 2 s is enough), so the inventory does not change; with `--until-taken` each box must really be taken (TAKEN). Each
item is read from scratch (times per box are comparable); `--pakai-ulang-bacaan` = like normal operation. At an arrangement pause:
change the boxes, hands out of the shelf, type `lanjut` in the terminal; `lewati` = skip the current item.
`DEMO.bat` menu S builds a 20-box plan (one request per box, Sled Bags by date) and runs it inside a recorded session.

```bat
:: laptop, cmd
venv\Scripts\python.exe scripts\stress_plan.py --formasi 5 --per-formasi 5 --acak 10 --out results\rencana_stress.txt
venv\Scripts\python.exe scripts\test_webcam_box.py --rencana results\rencana_stress.txt --waktu-per-barang 90 --expected 20 --inventaris data/rak/inventaris.csv --log-stress results\stress_01.csv --debug-readings results\bacaan_stress_01.csv --record results\uji_stress_01.mp4
```

```bash
# server, Remmina terminal (stream + tunnel from the laptop stay open, see 3b)
python scripts/stress_plan.py --formasi 5 --per-formasi 5 --acak 10 --out results/rencana_stress.txt
python scripts/test_webcam_box.py --config config/box_pipeline_config_server.yaml --source http://127.0.0.1:18090/video.mjpg --rencana results/rencana_stress.txt --waktu-per-barang 90 --expected 20 --inventaris data/rak/inventaris.csv --log-stress results/stress_server_01.csv --debug-readings results/bacaan_stress_server_01.csv --record results/uji_stress_server_01.mp4
```

Result per item (`--log-stress`): result (PICK / TAKEN / TIMEOUT / NOT FOUND / SKIPPED), seconds, box number, evidence
(text / REF / barcode), text read, how many RECHECK / CANCELLED, number of boxes on the shelf, number of readings and ms per
reading. A summary per part is printed at the end. Send the result CSV + readings file + recording for analysis.

## 3f. Intel RealSense (laptop, or directly on the server)

`camera.backend: "realsense"` opens the COLOUR stream through `pyrealsense2` (depth / infrared are not used). Index "auto" is not
used because one RealSense appears as several cameras. Laptop: `pip install pyrealsense2`, colour 1280x800 @ 30 on USB 3
(`config/box_pipeline_config_realsense.yaml`). **Directly on the server** (camera plugged into the server, no tunnel and no
`stream_webcam.py`) - never tried yet, test step by step:

```bash
source ~/trse_venv/bin/activate && pip install pyrealsense2
# Linux needs a udev rule so the camera opens without sudo (once, needs sudo):
sudo wget -O /etc/udev/rules.d/99-realsense-libusb.rules \
  https://raw.githubusercontent.com/IntelRealSense/librealsense/master/config/99-realsense-libusb.rules
sudo udevadm control --reload-rules && sudo udevadm trigger     # then unplug and replug the camera (USB 3 port)
python -c "import pyrealsense2 as rs; [print(d.get_info(rs.camera_info.name), d.get_info(rs.camera_info.serial_number), d.get_info(rs.camera_info.usb_type_descriptor)) for d in rs.context().devices]"
python scripts/test_webcam_box.py --config config/box_pipeline_config_server.yaml --expected 20 --inventaris data/rak/inventaris.csv
```

Without the udev rule, `pipeline.start()` fails with "Permission denied" / "failed to set power state". Without `sudo`: run
`test_webcam_box.py` with `sudo -E env "PATH=$PATH"` (temporary) or ask the server admin to install the rule.

## 4. Manual evaluation (test photos never used for training)

```bash
python scripts/eval_box_manual.py --images <test photos>/images --labels <test photos>/labels
```
Results: `results/eval_*/tabel_box.csv` (fill in the `kenapa_gagal` (why it failed) column yourself), `ringkasan.txt`
(formulas + numbers), `vis/` (green TP, red FP, yellow FN). Open the pictures and compare them with the numbers in the table.

## Known limits (from sandbox tests)

- OCR is the slowest part (about 0.7-2 seconds per crop on the sandbox CPU). That is why OCR runs only on size_label crops, at most
  2 crops per frame, and each box is read again at most every 15 frames. Measure again on the laptop.
- MATCH needs the product name to be read. If the product name is not inside any size_label (example Terumo Accuforce: size numbers
  in one block, the product name in another), the system answers CONFIRM, not MATCH. The fix is in the data: give a size_label to
  the product-name block too.
- Blurred / out-of-focus labels cannot be read -> CONFIRM (safe, no guessing).

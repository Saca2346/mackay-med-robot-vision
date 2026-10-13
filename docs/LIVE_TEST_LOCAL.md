# Live test on the laptop (local, RealSense D455, no server)

Everything runs on the laptop: camera, detection, OCR, verification and display. No tunnel, no `stream_webcam.py`, no
Remote Desktop. Progress is tracked in `docs/LIVE_TEST_CHECKLIST.md` (skip parts A5 and B, they are server only). The 20 tests:
`docs/CHECKLIST_DEMO.md`. For one recorded run of everything (with report) use `DEMO.bat` menu S (`docs/TEST_SESSION_GUIDE.md`).

All commands run on the **laptop (cmd)** in the folder `C:\Users\sacab\Downloads\medicine_shelf_vision\medicine_shelf_vision`
(after the folder move also `C:\Users\sacab\Downloads\mackay_vision\code`; the old path keeps working).

**General rules**
- Do not skip stages. **Stop and note it** if there is a MATCH / PICK on a wrong box (the only failure that must never happen).
  A doubtful box must become NEEDS CONFIRMATION.
- Only one program can use the camera: close RealSense Viewer and the previous test window (`q`) before starting a new one.
- Do not open heavy applications during the test (the i3 laptop slows down when hot). All evidence goes to one folder:
  `results\bukti_laporan_20261008\` (recordings `.mp4`, `bacaan_*.csv`, `verify_log.csv`, `screens\`). Keep failed tests as they are.
- No `scp` needed: when finished, tell Claude. The files are read straight from that folder for the slides.

Difference with the server: model v4 (OpenVINO 960 on the Intel iGPU), about 6-8 FPS, OCR on the CPU (about 110-130 ms per box).
These results test **correctness** (never a wrong pick), not top speed. The display is smooth because it does not go through Remote Desktop.

---

## L0. Preparation (once, no boxes in front of the camera)

```bat
cd /d C:\Users\sacab\Downloads\medicine_shelf_vision\medicine_shelf_vision
venv\Scripts\python.exe scripts\cek_usb_realsense.py --detik 5
venv\Scripts\python.exe -m pytest tests -q
venv\Scripts\python.exe scripts\uji_checklist.py
venv\Scripts\python.exe scripts\rak_inventaris.py lihat
```

| Check | Pass condition |
| --- | --- |
| USB | `RealSense D455: USB 3.x <-- GOOD` (if `2.1`: no frames, do not continue) |
| Unit tests | all pass (268 as of 14-10) |
| Text test of 20 requests | 0 failed (19 passed + #17 skipped while K10 is on the shelf; checked 08-10) |
| Inventory = shelf content | 20 boxes; only boxes that are really **not on the shelf** are `diambil` (taken) |

**Check now if it applies:** if the inventory marks a box `diambil` that is in fact on the shelf, put its status back
(you run this, not Claude):

```bat
venv\Scripts\python.exe scripts\rak_inventaris.py kembali K10
```
Then test #17 (`angiolite 4 x 19`) changes from NOT FOUND to READY; note it in the table.

---

## L1. Text demo (no camera, 5 minutes)

```bat
venv\Scripts\python.exe scripts\demo_request.py --sekali "ivascular angiolite 2.5 x 29"
venv\Scripts\python.exe scripts\demo_request.py --sekali "accuforce 2.75 x 15"
venv\Scripts\python.exe scripts\demo_request.py --sekali "terumo xperience 2.5 x 15"
```
Pass: K15 READY; `accuforce 2.75 x 15` NOT FOUND; wrong brand REPEAT.

---

## L2. Box count (camera further away, the whole shelf in view, 10 minutes)

```bat
venv\Scripts\python.exe scripts\test_webcam_box.py --config results\bukti_laporan_20261008\config_uji.yaml --request "angiolite,2.5,29" --expected 20 --record results\bukti_laporan_20261008\uji_live_tahap3.mp4
```
In the window: **`a`** = mark the shelf area (removes glasses / screens detected as boxes), **`s`** = save a screenshot to
`screens\`, **`q`** = quit.

Pass: sharpness > 100; box count 20 of 20 (19 is noted as R-21; 21 = a false object: press `a`).

---

## L3. One request per box (the core test, camera CLOSE)

Bring the camera closer until a box is **>= 28 pixels wide (ideally >= 40)** in the image; run `TEST_REALSENSE.bat` menu D to
measure it. In the 08-10 test with boxes 24 px wide (19 boxes in one frame) the size text was read in 7 % of readings with 0 PICK;
at 36 px it was 34 % and PICK for iVascular. Every time the camera is moved, press `a` again (the shelf area belongs to one framing).

One command per number (number = column `#` in `CHECKLIST_DEMO.md`, 1-16, and 17 while K10 is on the shelf). Example number 5
(accuforce 2.75 x 20):

```bat
TEST_LIVE_LOCAL.bat 5
```
(`UJI_LIVE_LOKAL.bat 5` still works.) The script saves `uji_live_05.mp4` and `bacaan_live_05.csv` to the evidence folder. There is
no automatic time limit: watch it yourself, and if 90 seconds pass without PICK press `q` and note **TIMEOUT** (do not delete it,
it is evidence of a failure).

**Priority order** (look-alikes and the riskiest first): #4, #5, #6 (accuforce), then #1, #17 (the two angiolite, K15 / K10,
almost the same look), then #11, #12 (conqueror), then #8, #9 (nagomi), then the rest by number. The result of each number goes
into table D of `LIVE_TEST_CHECKLIST.md`.

**Negative test** (a combination not on the shelf, must never PICK):
```bat
venv\Scripts\python.exe scripts\test_webcam_box.py --config results\bukti_laporan_20261008\config_uji.yaml --request "accuforce,2.75,15" --inventaris data\rak\inventaris.csv --expected 20 --record results\bukti_laporan_20261008\uji_live_negatif.mp4 --debug-readings results\bukti_laporan_20261008\bacaan_live_negatif.csv
```

Pass: **no PICK on a wrong box**; the right box may be NEEDS CONFIRMATION (safe, note it as material for improvement); the
negative test must not PICK.

---

## L4. Physical pick (optional, changes the inventory)

Back up first, run one request as in L3, pull the right box out by hand, then put it back.
```bat
copy data\rak\inventaris.csv data\rak\inventaris_sebelum_uji_live.csv
```
Pass: after pulling, TAKEN appears and that box is `diambil` in the inventory. Put it back with
`venv\Scripts\python.exe scripts\rak_inventaris.py kembali K15` (replace K15 with the box you pulled).
`DEMO.bat` and `TEST_REALSENSE.bat` menus 4 / 7 / 8 / 9 work on a copy of the inventory, so they need no backup.

---

## L5. Demo rehearsal (15 minutes)

Monday flow: type a request with `demo_request.py` (product, spec, status, rack / level), continue to the camera with `--kamera`,
show one refused case, time the whole flow. `DEMO.bat` menu S records the screen by itself; otherwise use the Snipping Tool
recorder (Win + Shift + R) and select the cmd and camera windows together.
Pass: twice in a row without intervention.

## L6. Collect the evidence

```bat
dir results\bukti_laporan_20261008 /s /b
```
When all numbers have run, send a summary: failed numbers, numbers that were NEEDS CONFIRMATION although the box was right,
box count, and the sharpness number. The files in the evidence folder are used directly for the slides.

## If something goes wrong

| Symptom | Likely cause | Action |
| --- | --- | --- |
| `Cannot open the RealSense` | Viewer / another test window still open, or the cable | close other programs; run `cek_usb_realsense.py` |
| `USB 2.1` | cable / port is not USB 3 | change the cable; no frames on 2.1 |
| 0 PICK, many `needs_confirmation` | text too small in the frame | bring the camera closer (box width >= 28 px; calibration: menu D) |
| Box count 21 | a false object (glass, screen) | key `a` for the shelf area |
| Laptop slows down | heat | pause 5 minutes between tests |

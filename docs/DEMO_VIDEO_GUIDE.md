# Demo video and evidence guide (Monday 12 October)

Goal: leave the demo with video that proves what the system did, including a take where it correctly refuses to
guess. Record **three layers**; the first one is automatic.

## 1. Three layers of evidence

| Layer | What it shows | How | Quality |
| --- | --- | --- | --- |
| **A. Annotated program view** | the camera image with boxes, labels, status line (PICK, TAKEN, NEEDS CONFIRMATION …) | automatic: every menu run saves `*_vis.mp4` (menu 9: `ambil_vis.mp4`, menu 8: `manual_vis.mp4`) | 6 fps (the processing rate): fine as proof, jerky to watch |
| **B. Screen recording of the live window** | the same view in real time at 30 fps, plus the terminal log | Windows: **Win + Alt + R** (Game Bar, records the focused window; **Win + G** shows the controls) or OBS Studio | smooth, best for the demo video |
| **C. Side view of the real shelf** | the hand taking the box, the box leaving the slot | a phone on a stand beside the laptop, 1080p, same take | the proof that the physical action happened |

Keep the files of one take together in one folder, named `take_NN_<request>`.

## 2. Before the first take (10 minutes)

1. Laptop on the charger, lid open, other programs closed (RAM is the limit: 8 GB), mode **Best performance**.
2. D455 in a **USB 3** port (`scripts\cek_usb_realsense.py` must say 3.x).
3. Menu **D** → session `demo_cal`: the calibration card must be PASS (or WARN with box width ≥ 28 px). Keep the camera where it passes.
4. Target boxes **mid-frame**, away from the shelf post (a box at the left edge took ~200 s to read; mid-frame: 17–19 s).
5. Inventory is read from a fresh copy in menus 4 / 7 / 8 / 9, so taking boxes does not change the real inventory.
6. Phone: 1080p, locked focus on the shelf, battery > 50 %, storage free.
7. One dry run of the whole list without recording.

## 3. Shot list (one take per line, 1–2 minutes each)

| # | Request | What the audience should see | Expected result |
| --- | --- | --- | --- |
| 1 | `ivascular angiolite 2.5 x 29` | box found among 20, green PICK, box taken, empty slot verified | PICK ≈ 20–30 s → TAKEN |
| 2 | `angiolite 4 x 19` | the **look-alike** next to it is not confused | PICK on K10 only |
| 3 | `permanent sled bag` | four identical boxes, FEFO picks the earliest expiry | K03 (2029-03-10) |
| 4 | `accuforce 2.75 x 15` | a size that is not on the shelf | NOT FOUND (red), nothing picked |
| 5 | `essential pro 3` | length missing | REPEAT (asks again), nothing picked |
| 6 | a non-iVascular item, e.g. `accuforce 2.5 x 12` | the system does **not** guess: candidate / unsure (blue / yellow), asks for confirmation | NEEDS CONFIRMATION — this is the safety rule working, say so in the video |

Take 6 is the honest one: at shelf distance only one source (the size text) is read for Terumo, Boston Scientific, APT
and OrbusNeich, so the system waits for a second. If the close-up test succeeds before Monday, add a take 7: move the
camera close, the box turns green.

## 4. During a take

- Say the request aloud (or type it) and start the clock: the status line shows seconds since the request.
- Hold the camera still while the status says SEARCHING; do not wave a hand over the shelf.
- Take the box out, then move the hand away for ~5 seconds so the empty slot can be verified (TAKEN).
- Do not cut a failed take: keep it (a refusal or a cancelled PICK is evidence too).

## 5. After the demo — collect

Laptop, folder `results\bukti_laporan_20261008\`:
`*_vis.mp4` (A), `hasil_ambil.csv` (time per item), `verify_log.csv` (every PICK / TAKEN / CANCELLED), `screens\peristiwa\*.jpg`
(automatic screenshot at each event), `bacaan_*` (raw text read). Add the screen recordings (B), the phone clips (C) and the
calibration card of the session. Index them in `results\bukti_arsitektur_20261009\README.md` as a new numbered section.

## 6. If something fails during the demo

| Failure | What to do |
| --- | --- |
| Camera does not open | run `UJI_REALSENSE.bat` option 1 (USB check); fall back to the recorded take with `rs_uji1.mp4` (menu 5) and say it is a replay |
| Everything stays blue / yellow | normal for non-iVascular at distance; show the close-up, or the text demo (`scripts\demo_request.py`) which never needs the camera |
| Laptop lags | close VS Code and the browser; menu 2 (detection only) shows the boxes while the rest recovers |
| Network / server problems | nothing in the demo needs the server |

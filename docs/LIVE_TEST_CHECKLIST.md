# Live-test progress checklist (RealSense D455; local and server versions)

Step-by-step guides: **local (laptop only)** `docs/LIVE_TEST_LOCAL.md`, or the old **Remote Desktop (server)** route
`docs/archive_id/UJI_LIVE_SERVER.md` (Indonesian, superseded by `docs/REMOTE_SERVER_CLAUDE_CODEX.md`). This checklist is shared;
the local version skips parts A5 and B (server only). The 20 tests: `docs/CHECKLIST_DEMO.md`.
Schedule: report Friday 09-10, demo Monday 12-10. Last updated: 08-10. For a full recorded run use `DEMO.bat` menu S
(`docs/TEST_SESSION_GUIDE.md`), which writes its own report.

How to use: tick `[x]` as soon as a stage meets its pass condition, fill in **Date** and **Evidence** (file name in `results/`).
Failed tests are not deleted: note them in the Note column and keep the files. They are the material for the "real difficulties" slide.

## A. Camera and network

| Stage | Pass condition | Status | Date | Evidence / note |
| --- | --- | --- | --- | --- |
| A1. RealSense on USB 3 | `cek_usb_realsense.py` shows `USB 3.x` | [x] | 08-10 | D455 USB 3.2 (was 2.1 with no frames; cable / port replaced) |
| A2. Local stream | frames 1280x800 @ 30 fps, sharpness > 100 | [x] | 08-10 | sharpness 778-790, frame 30 ms |
| A3. Pipeline runs on the laptop (no server) | BOX COUNT counted, no error | [x] | 08-10 | 20 boxes (round 1), 21 (round 2); 0 PICK / 145 s: text too small |
| A4. Camera framing for reading text | box width >= 28 px (ideally >= 40 px), sharpness >= 300; run TEST_REALSENSE.bat menu D (calibration) | [ ] |  |  |
| A5. Tunnel + stream to the server | the server opens `http://127.0.0.1:18090/video.mjpg`, input 10-15 fps | [ ] |  |  |

## B. Server ready (stages 0-2)

| Stage | Pass condition | Status | Date | Evidence / note |
| --- | --- | --- | --- | --- |
| B1. Code + data + model on the server | `git pull`, `data/rak/inventaris.csv`, `models/box_v4s.pt` present | [ ] |  |  |
| B2. Unit tests on the server | `pytest` all pass | [ ] |  |  |
| B3. Text test of 20 requests | `uji_checklist.py`: 0 failed (19 passed + #17 skipped while K10 is on the shelf) | [ ] |  |  |
| B4. Text demo (stage 1) | 3 requests as in the table | [ ] |  |  |
| B5. Video path / HUD (stage 2) | the HUD shows shown, input and max processing fps | [ ] |  |  |

## C. Camera tests (stages 3-4)

| Stage | Pass condition | Status | Date | Evidence / note |
| --- | --- | --- | --- | --- |
| C1. Box count (stage 3) | 20 of 20 (19 noted as R-21), no CAMERA MOVING | [ ] |  | `uji_live_tahap3.mp4` |
| C2. One request per box, priority order | no MATCH / PICK on a wrong box | [ ] |  |  |
| C3. Negative test | `accuforce 2.75 x 15`: no PICK | [ ] |  | `uji_live_negatif.mp4` |
| C4. Physical pick (optional, stage 5) | TAKEN appears, inventory `diambil`, then `kembali` (put back) | [ ] |  | back up `inventaris.csv` first |
| C5. Full demo rehearsal (stage 6) | the flow twice in a row without intervention, screen recorded | [ ] |  |  |

## D. Result per box (fill in during stage 4)

Status: `PICK` (success), `RECHECK` then success, `NEEDS CONFIRMATION` (safe, material for improvement), `TIMEOUT` (failed),
`WRONG` (MATCH on a wrong box: stop the test, send the recording).

| # | Request | Result | Seconds to PICK | Text read correctly? | Evidence (recording / readings) |
| --- | --- | --- | --- | --- | --- |
| 4 | accuforce 2.5 x 12 |  |  |  | `uji_live_04.mp4` / `bacaan_live_04.csv` |
| 5 | accuforce 2.75 x 20 |  |  |  | `uji_live_05.mp4` / `bacaan_live_05.csv` |
| 6 | accuforce 3.5 x 15 |  |  |  | `uji_live_06.mp4` / `bacaan_live_06.csv` |
| 11 | conqueror 3.5 x 15 |  |  |  | `uji_live_11.mp4` / `bacaan_live_11.csv` |
| 12 | conqueror 4 x 8 |  |  |  | `uji_live_12.mp4` / `bacaan_live_12.csv` |
| 8 | nagomi 2.75 x 33 |  |  |  | `uji_live_08.mp4` / `bacaan_live_08.csv` |
| 9 | nagomi 3 x 24 |  |  |  | `uji_live_09.mp4` / `bacaan_live_09.csv` |
| 1 | angiolite 2.5 x 29 (K15) |  |  |  | `uji_live_01.mp4` / `bacaan_live_01.csv` |
| 17 | angiolite 4 x 19 (K10; look-alike of #1) |  |  |  | `uji_live_17.mp4` / `bacaan_live_17.csv` |
| 2 | xperience pro 2.5 x 15 |  |  |  | `uji_live_02.mp4` / `bacaan_live_02.csv` |
| 3 | essential pro 3 x 40 |  |  |  | `uji_live_03.mp4` / `bacaan_live_03.csv` |
| 7 | ryurei 2 x 10 |  |  |  | `uji_live_07.mp4` / `bacaan_live_07.csv` |
| 10 | nc emerge 5.5 x 12 |  |  |  | `uji_live_10.mp4` / `bacaan_live_10.csv` |
| 13 | sapphire 3 x 15 |  |  |  | `uji_live_13.mp4` / `bacaan_live_13.csv` |
| 14 | scoreflex 3 x 15 |  |  |  | `uji_live_14.mp4` / `bacaan_live_14.csv` |
| 15 | agent 2.75 x 30 |  |  |  | `uji_live_15.mp4` / `bacaan_live_15.csv` |
| 16 | permanent sled bag |  |  |  | `uji_live_16.mp4` / `bacaan_live_16.csv` |

Summary (fill in when table D is complete): PICK ___ / 17, NEEDS CONFIRMATION ___, TIMEOUT ___, WRONG ___ (must be 0).

## E. Documentation for the report slides

| Evidence asked for | Source | Status |
| --- | --- | --- |
| Photo / recording of a pick (PICK) | `screens/peristiwa/*.jpg`, `uji_live_*.mp4` | [ ] |
| Example of a **successful** test (log + photo) | `PICK` rows in `verify_log.csv` | [ ] |
| Example of a **failed** test (log + photo) | `TIMEOUT` / `PICK CANCELLED` / `NEEDS CONFIRMATION` rows | [ ] |
| Details of the boxes read on the shelf | `bacaan_live_*.csv` + an enlarged HUD screenshot | [ ] |
| Real difficulties (RealSense USB 2.1, text too small, false detections) | log of 08-10 in part A, RealSense Viewer screenshot | [x] |
| Results sent to the laptop | `hasil_uji_live.tgz` (stage 7) then `scp` | [ ] |

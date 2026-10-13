# Recorded test session (DEMO.bat, menu S)

One menu runs the whole test in a fixed order and records the screen from the first step to the last, so the evidence
shows the cmd window and the camera window together. Every file of a session lands in one folder:
`results\sesi_<date_time>\`.

```bat
cd C:\Users\sacab\Downloads\medicine_shelf_vision\medicine_shelf_vision
DEMO.bat
```
Choose **S**. A minimised window "SCREEN RECORDER" opens: leave it alone, it stops by itself at the end.

## Before you start (5 minutes)

1. Charger in, lid open, power mode Best performance. Close VS Code and the browser (RAM is the limit on this laptop).
2. RealSense D455 on a **USB 3** port, camera on a tripod, the whole shelf in view.
3. All 20 boxes on the shelf. Take a phone photo of the shelf: it is the reference for putting boxes back.

## The six steps and when each one passes

| Step | What happens | Pass when | If it fails |
| --- | --- | --- | --- |
| 1 Hardware check | model file, inventory, free RAM, RealSense type | `READY: 0 FAIL` | FAIL lines say what to fix; you may continue (`y`), it is logged |
| 2 Verification | the automated tests (about 30 s) | `... passed`, no `failed` | stop and report; do not demo on failing code |
| 3 Calibration | live preview with PASS / WARN / FAIL per check, then a 3 s measurement | no FAIL (WARN on box width is allowed) | move the camera and repeat the session |
| 4 Hardware test 1 | detection only, annotated video | HUD shows `boxes 20` and sharpness 300 or more for 10 s | adjust camera / light, press q, repeat |
| 5 Stress test | all 20 boxes in turn: PICK, you take the box, TAKEN, next | each box ends as TAKEN, PICK, NOT FOUND or TIMEOUT (all are logged) | type `lewati` in the cmd window to skip a box; q stops |
| 6 Random requests | you type numbers from the list, the camera verifies each | every request gets an answer | Enter on an empty line ends the session |

Expected results, from the camera tests of 10-13 Oct (do not promise more than this):
* iVascular boxes (numbers 5, 9, 14 in the list) reached PICK at shelf distance.
* Other products usually stay **candidate**: their second source (REF / barcode) is not readable at shelf distance. That is the
  safety rule working (PICK needs two independent sources), not a crash. Bring the camera closer to the label to try.
* Sled Bags (numbers 1, 2, 4, 16): each number asks for one dated box. PICK only after the printed date of that box is read,
  which needs the camera close to the date. Number 0 = any Sled Bag (earliest date first).

## What the session folder contains

| File | Content |
| --- | --- |
| `00_layar_penuh.mp4` | the whole screen, start to end (8 fps, 1280 px) |
| `langkah.txt` | time of every step and its outcome |
| `sesi_<T>_kalibrasi.json/.jpg` | calibration card |
| `04_hw_test1_vis.mp4` | hardware test 1 with boxes drawn |
| `05_stress.csv`, `05_stress_vis.mp4`, `05_stress.mp4`, `05_bacaan.csv` | stress test: result per box, annotated video, raw video, every reading |
| `06_acak_<n>_vis.mp4`, `demo_log.jsonl` | random requests: video per request and the text answers |
| `screens\`, `verify_log.csv` | event pictures (PICK, TAKEN, ...) and the event log of this session only |
| `LAPORAN_SESI.md` | the report written at the end from the files above |

The real inventory (`data\rak\inventaris.csv`) is never changed: the session works on its own copy.

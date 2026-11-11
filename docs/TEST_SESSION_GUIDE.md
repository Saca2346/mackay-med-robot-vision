# Recorded test session (DEMO.bat, menu S)

For the current **one-box visual lock** task, use DEMO menu **3** instead; see
[TARGET_VIEW.md](TARGET_VIEW.md). Menu S below is the longer verified PICK/TAKEN
test and still waits for independent identity sources.

One menu runs the whole test in a fixed order and records the screen from the first step to the last, so the evidence
shows the cmd window and the camera window together. Every file of a session lands in one folder:
`results\demo_<launch_time>\sesi_<date_time>\` (the exact path is printed on screen).

The current default is an **external USB webcam**, using `config/box_pipeline_config_webcam.yaml`: requested
1920 x 1080 colour, MJPG, and the existing v4 OpenVINO model at 736 x 1280. RealSense is not required.

```bat
cd C:\Users\sacab\Downloads\mackay_vision\code
DEMO.bat
```
Choose **S**. A minimised window "SCREEN RECORDER" opens: leave it alone, it stops by itself at the end.

## Before you start (5 minutes)

1. Charger in, lid open, power mode Best performance. Close VS Code and the browser (RAM is the limit on this laptop).
2. Plug in the external USB webcam, close other camera apps, and place it on a tripod with the whole shelf in view.
   Unplug unused cameras, including RealSense. `camera.index: auto` selects the largest resolution; confirm the
   preview is from the external webcam. If needed, use `TEST_WEBCAM.bat` menu **1** to list camera indices and set
   `camera.index` in the webcam config to the correct number before restarting DEMO.
3. All 20 boxes on the shelf. Take a phone photo of the shelf: it is the reference for putting boxes back.
4. A shelf map or crop saved for D455 may not fit the webcam view. If changing camera/view, refresh the shelf map
   using `TEST_WEBCAM.bat` **P** before starting DEMO. In the live preview, use **a** to adjust the shelf area if needed.

## The six steps and when each one passes

| Step | What happens | Pass when | If it fails |
| --- | --- | --- | --- |
| 1 Hardware check | model file, inventory, free RAM, readable webcam frame and actual resolution | `READY: 0 FAIL` | check USB connection, close camera apps, or correct camera.index; low resolution gives a WARN |
| 2 Verification | the automated tests (about 30 s) | `... passed`, no `failed` | stop and report; do not demo on failing code |
| 3 Calibration | live preview with PASS / WARN / FAIL per check, then a 3 s measurement | no FAIL (WARN on box width is allowed) | move the camera and repeat the session |
| 4 Hardware test 1 | detection only, annotated video | HUD shows `boxes 20` and sharpness 300 or more for 10 s | adjust camera / light, press q, repeat |
| 5 Stress test | all 20 boxes in turn: PICK, you take the box, TAKEN, next | each box ends as TAKEN, PICK, NOT FOUND or TIMEOUT (all are logged) | type `lewati` in the cmd window to skip a box; q stops |
| 6 Random requests | you type numbers from the list, the camera verifies each | every request gets an answer | Enter on an empty line ends the session |

Historical results from camera tests of 10-13 Oct (not a guarantee for the current webcam placement):
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
| `grasp.jsonl`, `screens\pick\` | box/barcode pixel coordinates and PICK crops; metric depth/3D fields unavailable with a normal webcam |
| `LAPORAN_SESI.md` | the report written at the end from the files above |

The real inventory (`data\rak\inventaris.csv`) is never changed: the session works on its own copy.
PICK still requires two independent sources; depth is not an identity source. Physical 3D grasp validation is deferred
until a depth-capable setup is available. DEMO menu **5** can replay a selected webcam recording; keep its
`.mp4.times.csv` file beside the video and report replay results separately from live tests.

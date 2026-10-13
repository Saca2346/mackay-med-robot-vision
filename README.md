# Medicine shelf vision (Mackay medicine-box robot)

Vision module of the NSTC / Mackay Memorial Hospital robot that picks catheterisation devices from a shelf. A RealSense D455
looks at the shelf; the pipeline finds every box (YOLO11-OBB), reads it (OCR of the size label, REF catalogue code, GS1 barcode)
and tells the robot which box to pick, only when two independent sources agree.

```
camera (D455, 1280x800) -> detector (YOLO11s-OBB v4, OpenVINO 960) -> tracker / motion
   -> OCR (RapidOCR PP-OCRv6) + REF rules + GS1 barcode -> decision (2 independent sources)
   -> selection flow: CHECKING SHELF -> SEARCHING -> FOUND / PICK -> TAKEN -> DONE
      (or NEEDS CONFIRMATION / NOT FOUND)  -> inventory (FEFO, verified boxes only)
```

## Quick start (laptop, cmd)

```bat
cd C:\Users\sacab\Downloads\medicine_shelf_vision\medicine_shelf_vision
DEMO.bat
```

| Menu | What it does |
| --- | --- |
| S | recorded test session: hardware check, tests, calibration, hardware test 1, 20-box stress test, random requests, report |
| 1 | pre-flight check (model, inventory, RAM, RealSense on USB 3) |
| 2 | 7 text scenes, no camera |
| 3 / L | camera with a box picked by number from the list / show the list |
| 4 | fixed camera scene (Angiolite 2.5 x 29) |
| 6 | all boxes in turn, take each box |

Other menus: `TEST_REALSENSE.bat` (camera checks, recordings, calibration, shelf-data sessions), `TEST_LIVE_LOCAL.bat N`
(one checklist test). The old Indonesian names (`UJI_*.bat`) still work.

## Documentation

| Topic | File |
| --- | --- |
| Recorded test session (start here) | `docs/TEST_SESSION_GUIDE.md` |
| How the pipeline works, keys, colours, flow | `docs/BOX_PIPELINE.md` |
| What was fixed, measured, tried and dropped | `docs/FIXES_LOG_20261008-11.md` |
| Live test on the laptop / checklist / product list | `docs/LIVE_TEST_LOCAL.md`, `docs/LIVE_TEST_CHECKLIST.md`, `docs/CAMERA_TEST_PRODUCT_LIST.md`, `docs/CHECKLIST_DEMO.md` |
| Camera calibration and shelf data | `docs/DATA_RAK_PROTOCOL.md`, `docs/DATA_COLLECTION_PLAN.md` |
| Demo video | `docs/DEMO_VIDEO_GUIDE.md` |
| Server, VS Code Remote-SSH, Claude Code + Codex | `docs/REMOTE_SERVER_CLAUDE_CODEX.md`, `AGENTS.md`, `docs/HANDOFF.md` |
| FY115 review answer (vision part) | `docs/REVIEW_RESPONSE_VISION_115.md` |
| Progress report 09-10 | `docs/PROGRESS_REPORT_20261009.md` |
| Indonesian originals / superseded docs | `docs/id/`, `docs/archive_id/` |

## Tests

```bat
venv\Scripts\python.exe -m pytest tests -q
```
268 tests pass (14-10). The safety rules (two sources, dated requests, no inventory changes in tests) are listed in `AGENTS.md`.

## Data

Hospital data (inventory, box photos, videos) and models are not in this repository (`.gitignore`); the lab server keeps the
master copy. The labelling project with photos, labels and training scripts is the sibling folder `prelabel_size_label_full`
(after the folder move: `mackay_vision\labeling`).

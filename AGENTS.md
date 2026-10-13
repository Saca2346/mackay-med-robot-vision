# Rules for AI coding assistants (Claude Code, Codex) in this repository

Project: vision module of the Mackay / NSTC medicine-box robot. A camera (RealSense D455) finds medicine boxes on a shelf
(YOLO11-OBB), reads them (OCR, REF code, GS1 barcode) and tells the robot which box to pick.

## Safety rules (never change without the project owner's written decision)

1. **MATCH / PICK needs two independent sources** (e.g. size text + REF, or barcode). One source = candidate. Do not relax it.
2. A request for a dated box (`... 2029-03-24`) may only PICK after that printed date was read (`src/box_verifier.py`, `REQUIRED_EXPIRY`).
3. Never mark inventory rows verified or taken on the user's behalf; tests use a copy of `data/rak/inventaris.csv`.
4. Never create new detection classes or put product names in class / label files.
5. Test and replay runs must write their logs to a scratch or session folder, never into `results/bukti_*` (evidence).
   Use `scripts/sesi_config.py` or `scripts/demo_foto_target.py`, which redirect all outputs.

## Repository hygiene

* This repository is public: never commit inventory files, photos, videos, models, serial numbers, passwords or tokens
  (see `.gitignore`). Data lives on the lab server, not on GitHub.
* Code, comments, docs and screen texts are in English. Keep `.bat` files ASCII with CRLF line endings; inside `.bat` use
  labels and `goto` instead of `( ... )` blocks that read `%VAR%` set in the same block.
* Run `venv\Scripts\python.exe -m pytest tests -q` before handing work over; all tests must pass.
* Commit only when the owner asks; one branch per assistant (`claude/<topic>`, `codex/<topic>`), merged after review.

## Coordination between assistants

* Before starting, read `docs/HANDOFF.md` (who works on what, open questions). After finishing, add a short entry there:
  date, assistant, what changed, how it was tested, what is open.
* Do not edit the same files in parallel; claim a topic in `docs/HANDOFF.md` first.
* Results are claimed only with evidence (log, CSV, picture). Say what was not tested.

## Where things are

| Path | Content |
| --- | --- |
| `src/` | pipeline: detector, OCR, verifier (decision rules), session (search / PICK / TAKEN), inventory |
| `scripts/` | entry points: `test_webcam_box.py` (live / replay / photo), `demo_request.py`, calibration, session tools |
| `DEMO.bat`, `UJI_REALSENSE.bat` | menus used in front of the shelf (menu S = recorded test session) |
| `docs/` | guides; start with `docs/TEST_SESSION_GUIDE.md`, `docs/FIXES_LOG_20261008-11.md` |
| `tests/` | automated tests (pytest) |

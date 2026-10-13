# Handoff log (Claude Code, Codex, people)

Newest entry first. One entry per finished piece of work: date, who, what changed, how it was tested, what is open.

## 2026-10-14 (evening) - Claude Code (laptop): audit, server comparison run, commit

* Folder move is DONE (the entry below is outdated on this point): `mackay_vision\code` and `\labeling` are real folders, the old
  paths are junctions. Both repos `git fsck` clean, 268 tests pass. Parent `README.md` updated.
* Server comparison run on the real manifest (`labeling\server_results\langkah7_manifest.txt`, 14 Oct 04:00) with
  `tools/compare_server.py`; report in `labeling\server_results\COMPARE_SERVER_LAPTOP.md`. Server 51 done / 112 auto, laptop
  76 / 87; 104 expiry specs differ in content (77 are `auto` on both sides, so the server changed them, probably
  `rapatkan_label_tanggal.py`); 26 are `auto` on the server but `selesai` on the laptop (the 11 Oct batch rule); 1 the other way.
  Date tools exist only on the server (`review_tanggal.py`, `rapatkan_label_tanggal.py`, `cek_tanggal_terlewat.py`,
  `bandingkan_model.py`, ...); `labels_final_obb/IMG_20260923_132329_751.txt` differs.
* Open: pulling the server copies needs the SSH key of `docs/REMOTE_SERVER_CLAUDE_CODEX.md` step 1 (laptop key is not on the
  server yet). Then merge: server label content + laptop `selesai` status, after a check.
* Missing locally, not needed by the box pipeline: `data/test_assets/synthetic_shelf.mp4` (`scripts/generate_test_assets.py`),
  `data/book_db.sqlite`. The 3-class model and v5/v6 are only on the server.
* Commit: all work since 8 Oct committed on branch `claude/sync-20261014` (not pushed).

## 2026-10-14 (later) - Claude Code (laptop): folders, English, server comparison

* Owner decisions: one parent folder `Downloads\mackay_vision\` with `code\` + `labeling\`; English names with the old names kept as
  one-line wrappers; compare server and laptop label data before choosing a master.
* Folder move NOT done yet: the first attempt with PowerShell Move-Item half-moved the repo (62 files) because a program held `.git`;
  everything was moved back with robocopy and verified (31,967 files, `git fsck` clean, 268 tests pass). The move is now a script the
  owner runs with VS Code / Claude closed: `Downloads\MOVE_TO_MACKAY_VISION.bat` (cmd `move` = rename only, junctions keep old paths).
* English entry points: `TEST_REALSENSE.bat`, `TEST_LIVE_LOCAL.bat`, `TEST_ROBOT_LAPTOP.bat`, `TEST_CAMERA_TO_SERVER.bat`; labelling
  repo `RUN_ON_SERVER.bat` (outputs to `server_results\`). Old `UJI_*.bat` / `SERVER_JALANKAN.bat` forward to them (tested).
* Docs: English `README.md`, `docs/BOX_PIPELINE.md`, `CHECKLIST_DEMO.md`, `LIVE_TEST_LOCAL.md`, `LIVE_TEST_CHECKLIST.md`,
  `CAMERA_TEST_PRODUCT_LIST.md`; Indonesian originals in `docs/id/`, superseded docs in `docs/archive_id/` (references updated).
  Labelling repo: `docs/LABELING_TOOLS_GUIDE.md`, `docs/TRAINING_3_CLASSES.md`, new `README.md`; root clutter moved to `archive\`
  (list in `archive\MOVED_FILES_20261014.csv`) and `server_results\`.
* Server comparison: `server_scripts/langkah7_manifest.sh` (read-only) + `tools/compare_server.py`; tested on a simulated list
  (found exactly the 2 injected differences). Not run on the server yet.

## 2026-10-14 - Claude Code (laptop)

* Sled Bag by expiry: a request may contain a date (`permanent sled bag 2029-03-24`); PICK only after that printed date is read.
  Files: `src/box_verifier.py`, `src/box_inventory.py`, `src/box_demo.py`, `scripts/test_webcam_box.py`, `scripts/demo_request.py`.
  Tested: `tests/test_expiry_request.py` (8 tests) + full suite 268 passed. Not tested live with the camera yet.
* Box list in the cmd: one number per box (Sled Bags by date), 0 = any Sled Bag (`scripts/demo_daftar_kotak.py`).
* Recorded test session: `DEMO.bat` menu S (hardware check, tests, calibration, hardware test 1, 20-box stress test, random
  requests, report). New: `scripts/rekam_layar.py`, `scripts/sesi_config.py`, `scripts/ringkas_sesi.py`. Not run live yet.
* Open: server and laptop label data differ (server: expiry status 51 done / 112 auto; laptop: 76 done). Decide which is master
  before the next training (see `docs/REMOTE_SERVER_CLAUDE_CODEX.md`).
* Open: 13 rows + 4 pictures from Claude's replays (13 Oct 16:17-16:32) are in `results/bukti_laporan_20261008`; cleaned copy in
  `results/demo_20261012/replay_menu5_analisis/`, owner decides whether to apply it.

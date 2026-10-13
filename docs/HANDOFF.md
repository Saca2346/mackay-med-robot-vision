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
* SSH key installed by the owner (`ssh trse-server` works without a password). Server copy pulled read-only to
  `labeling\server_copy_20261014\`. Finding: the server is newer (12 Oct 02:31-02:56) and built on the laptop's 76: it fixed the
  hourglass-as-"8" bug, tightened the iVascular date labels and deliberately reopened 26 photos with missed dates. So the laptop
  took the server's expiry specs, `labels_final_obb` and date tools (laptop copy kept in
  `labeling\archive\laptop_before_server_merge_20261014\`); labels are now identical. Exception: `labeling_progress.csv` is newer
  on the LAPTOP (7 photos re-checked 8 Oct = `selesai`; the server still has `perlu dicek manual`, likely why v6 used 156 photos).
* 33 `New image` photos (28 Sep 16h) put in the label queue: copied to `raw/`, v4 drafts (`box_v4s.pt`, 1280, cpu) via
  `pred_to_spec.py` (note SEDANG DIKERJAKAN, out of training), expiry proposals (19 dates), batch
  `labeling\batches\baru_20260928_16h` with `REVIEW.md`. They are one-box close-ups (~40 boxes, not ~400); v4 drafts include the
  drink bottle and monitor as boxes. Seed set unchanged (51 photos), the 33 are pool "sedang dikerjakan".
* Fixes: `labeling/tools/auto_expiry.py` crashed on a photo with 0 boxes and on very thin tiles; `export_datasets.seed_state`
  let tool-reviewed labels into training without the expiry review (now same rule as specs; verified on simulated input).
  `code/src/box_expiry.py` got the server's hourglass fix (`82029-03-24` was dropped, so a dated Sled Bag request could not PICK);
  2 new tests, 270 pass. The live pipeline is not re-tested with the camera.
* Server sync DONE: the owner ran `labeling\RUN_ON_SERVER.bat langkah8_sync_laptop` (14 Oct 06:26): 4 server files backed up to
  `~/Downloads/cadangan_sebelum_sync_laptop_20261014.tgz`, 33 photos into `raw/` (201 files), 278 files extracted, batch rebuilt.
  Checked with a fresh read-only manifest: 640 of 644 files identical, expiry status 51 / 145 on both sides, the 33 photos
  sha1-identical. Remaining differences are docs only (server keeps 3 Indonesian guides at the root; the laptop root
  `PANDUAN_TRAINING_3KELAS.md` has a newer "Status 11-10" section). From now on edit labels on the server only.
* Push: branch `claude/sync-20261014` is on GitHub in both repos (PRs not opened). The old labelling remote
  `Saca2346/mackay-med-robot-vision-` never existed; the owner created `Saca2346/mackay-med-robot-vision-labeling` (private),
  now `origin` of `labeling\`, with `main` (first commit) and the branch pushed. History checked: code and docs only (72 files).
* Missing locally, not needed by the box pipeline: `data/test_assets/synthetic_shelf.mp4` (`scripts/generate_test_assets.py`),
  `data/book_db.sqlite`. The 3-class model and v5/v6 are only on the server.
* Commit: all work since 8 Oct committed on branch `claude/sync-20261014` in both repos (not pushed). The labelling repo got a
  whitelist `.gitignore` first (it had none; `git add -A` would have added the photos).

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

# Follow-up of 17 October 2026

18 October continuation: [FOCUSED_WEBCAM.md](FOCUSED_WEBCAM.md) documents the
Angiolite-specific YOLO26 webcam launcher, measured 1.51-1.84 s on the latest-session
replays, the slower older-view case, Drive review, and **397 passing tests**.

Latest task: the owner chose **one visual target without waiting for PICK**.
Use `DEMO.bat` **1 -> 3** and [TARGET_VIEW.md](TARGET_VIEW.md). The sections below
record the earlier fixes and outstanding physical verification/model work.
Final review on 18 October: **383 tests passed**; the 120-frame Conqueror replay
completed with a visual lock and no PICK/grasp output. Evidence is in
`results/codex_target_view_20261017/`. Live timing at the current webcam pose remains open.

The earlier local fixes passed **350 tests**. The live model remains v4. No files were staged or
committed: existing Claude/Codex ownership still applies, and the owner decides the split.

Current camera preference: **external USB webcam**. DEMO now uses the existing webcam profile, with requested
1920 x 1080 colour and v4 OpenVINO 736 x 1280. The earlier D455 candidate experiments below remain historical results.
The webcam follow-up passes **354 tests** (33.94 s; `results/codex_followup_20261017/pytest_webcam_final.log`).
Webcam preflight requires no RealSense SDK; launcher smoke checks passed. Physical webcam/shelf testing is still open.

## Completed fixes

* `DEMO.bat` uses the current webcam configuration and creates a fresh `results/demo_TIMESTAMP/` folder with its own
  config and inventory copy. Screens, verification logs, PICK crops and grasp output stay in that folder. Recorded
  menu S sessions are nested below it. Preflight checks the same settings; missing fallback video is reported clearly.
* Colour, aligned depth and intrinsics remain paired while detection and PICK panel OCR run. Missing depth stays
  unavailable; another frame's depth is never substituted. Reconnecting clears the camera's old depth cache.
* Grasp dimensions and neighbouring gaps use both focal lengths for the measured axis. A located barcode is included
  in the no-grip polygons but is never treated as an identity source.
* Rectification of a 45-degree label no longer selects duplicate corners or collapses the barcode coordinates.
* Target lock returns to full search after six readings with only partial identity. A product name alone is not a
  complete source; two identity sources with an unread expiry are no longer displayed as only one source.

Safety is unchanged: PICK requires independent identity evidence; a dated request also needs the printed date.
These fixes have synthetic regression coverage; physical depth/grasp validation is still required.

## YOLO26 candidate: exported, not deployed

The server final model was copied to ignored scratch and its SHA-256 verified:
`bff70e4a116129ca9696343b385f538aa3b9d6cc83add4ab19d3ee19d7d62e33`.
The OpenVINO export has input `[1, 3, 544, 960]`. Artifacts, commands, photo outputs and hashes are under
`results/codex_followup_20261017/yolo26/candidate_cpu/`; `summary.json` contains the diagnostic results.

| Photo | .pt photo input 960 | OpenVINO shelf input 544 x 960 |
| --- | --- | --- |
| Braun 2.75 x 30 | PICK, text + REF 5023243 | No PICK |
| CHROMA 2.75 x 33 | Candidate, no PICK | Candidate, no PICK |

Both profiles detect the target outlines, but the fixed shelf input is not a replacement for the portrait-photo
profile. On the CHROMA image, raw box detections are 33 for square .pt and 45 for fixed input versus 27 Claude-drawn
reference boxes. Those references still require owner review; these are diagnostics, not human-reviewed accuracy.
Fixed .pt/OpenVINO detections match one-to-one at IoU >= 0.5 for all 25 task-1 detections; task 2 has 94 matches with
2 unmatched detections on each side. Do not infer equivalence or deployment readiness from box counts alone.

No speed benchmark was performed: Windows reported AC disconnected, and the owner confirmed the charger was not
connected. The existing separate new-brand task-photo model is retained. Reproduce the bounded diagnostic workflow
with a fresh output folder:

```bat
venv\Scripts\python.exe scripts\prepare_yolo_candidate.py --weights results\codex_followup_20261017\yolo26\source\yolo26s_final.pt --out results\candidate_check_NEW
```

## Owner and lab checklist, in order

1. Connect the laptop charger before any FPS or time-to-PICK comparison. Connect the external USB webcam,
   place it on a tripod facing all 20 boxes, and take a shelf reference photo. Close other camera apps and unplug
   unused cameras. Confirm the preview is the external webcam; auto selection uses the largest resolution.
2. Run `DEMO.bat`, menu **1** for preflight, then **3** for the current visual-target task. Keep the active model unchanged.
   When testing physical verification separately, choose **S** for the recorded PICK/TAKEN session.
   Follow `docs/TEST_SESSION_GUIDE.md`; in the dated Sled Bag checks use requests **2**, **16**, then **0**. Stop on any
   wrong-box PICK. Keep the printed session folder and `LAPORAN_SESI.md` for review.
3. Review Claude's task-photo labels in `results/task_20261015/training_data/preview/`, the 20 reference spines in
   `results/meeting_next/yolo_compare/gt_ref_preview.jpg`, and the 9 inferred positions in
   `results/meeting_next/second_source_audit/inventaris_audit.csv`. Record actual corrections/approval; these have not
   been marked human-reviewed. Full new-brand training still needs this review.
4. Obtain the original CHROMA photo or a closer, sharp view of REF and barcode. A verified CHROMA GTIN entry requires
   the physical box and owner confirmation. Keep Task 2 as candidate until a second source is available.
5. With Leon, inspect webcam box/barcode pixel coordinates and no-grip polygons in `grasp.jsonl` and PICK crops.
   Metric depth and 3D grasp tests are deferred: a normal webcam does not supply them. When D455 is used again,
   validate depth with `TEST_REALSENSE.bat` **G** and `docs/GRASP_OUTPUT.md` before physical grasp trials.
6. Ask the team for a written decision on whether an arm may approach a one-source candidate and scan before PICK.
   The existing two-source PICK rule remains in force. Approaching with one source is not yet approved by this work.
7. After review and charger-powered laptop shelf A/B/FPS plus new-brand checks, decide whether to deploy YOLO26.
   The code/weights preparation is done; hardware validation and deployment approval are still open.

## Validation and ownership

`venv\Scripts\python.exe -m pytest tests -q --basetemp results/codex_followup_20261017/pytest_full_final -p no:cacheprovider`
completed with **350 passed in 38.31 s**. Use a different fresh `--basetemp` when repeating. This avoids the inaccessible
existing Windows pytest temp/cache directories without modifying their permissions. Log:
`results/codex_followup_20261017/pytest_full_final.log`.

Real `cmd.exe` smoke checks passed for launcher startup/exit and the missing-replay guard, using file-based menu input;
logs are in `results/codex_followup_20261017/demo_smoke/`. No camera was opened. DEMO remains ASCII with CRLF endings.

This follow-up adds changes to existing Claude files (`src/box_camera.py`, `src/box_lock.py`, `src/box_grasp.py`,
`src/box_crops.py`, `scripts/test_webcam_box.py`, their relevant tests/docs) as well as the demo launcher/helpers,
`src/box_stream.py`, the new candidate preparation script and new regression tests. Preserve the pre-existing changes
when splitting commits. No inventory master, historical evidence, live model setting, server training or labels were edited.

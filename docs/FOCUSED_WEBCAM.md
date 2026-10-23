# Focused webcam: Angiolite sizes

**Trained stable-outline candidate:** use `FOCUS_WEBCAM_STABLE.bat` and
[STABLE_WEBCAM.md](STABLE_WEBCAM.md). This selects the separately evaluated
100-epoch webcam candidate and the current pixel-based outline hold.

**20 October update:** use `TEST_TARGET.bat` to choose webcam or RealSense, or
`FOCUS_WEBCAM.bat` / `FOCUS_REALSENSE.bat` for the selected camera. Direct CMD
commands, current output paths and measured timing indicators are in
[TARGET_CAMERAS.md](TARGET_CAMERAS.md). New sessions use
`results/target_<camera>_<timestamp>/take_<timestamp>/`; the older `focus_` layout
and measurements below remain historical evidence.

## Current workflow: 19 October, two-source VERIFIED

### Automatic video, CMD and photo evidence

Start `FOCUS_WEBCAM.bat` (or DEMO **F**) and choose **2**, **3** or **4**. Every
camera run automatically creates the following in `results/focus_<timestamp>/`:

| File / folder | Evidence |
| --- | --- |
| `camera_<take>.mp4` and `.times.csv` | Original camera video and actual frame timing |
| `view_<take>.mp4` | Annotated detection/verification video |
| `cmd_<take>.txt` | Command, stdout, stderr, requests/status messages and exit code; also displayed in CMD |
| `photos_<take>/` | Paired original-camera and annotated PNG photographs, per-photo JSON and `index.jsonl` |
| `verification/` | Number/label/spine crops and structured observed fields |
| `readings_<take>.csv`, `target_view.jsonl` | OCR readings, status changes and measured timing |

Photos are automatic on status/evidence changes and at the end, including
unverified/searching results. Press **S** in the camera window for an extra pair.
Photo encoding runs in the background with a bounded queue; exit waits for all
queued photographs and reports write failures rather than silently dropping them.
Press **Q** to finish so the video files can close normally. Type subsequent
requests in the same terminal; transcript capture keeps console input available.
The file list and session folder are printed before the camera opens. A failed or
interrupted run points to its transcript rather than reporting successful completion.

The annotated recording uses a fixed display frame rate; use JSON wall timing or
the raw video's timing CSV to assess the 1-2 s requirement, not its playback speed.
Photographs show the current scene; `verification/reading_N/` retains the actual
earlier OCR inputs. These additions apply to new sessions and cannot reconstruct
terminal output that was not logged in an older live run.

`FOCUS_WEBCAM.bat` (also DEMO **F**) now uses **`--verify-target`**. The older
visual-only measurements below are historical and do not measure verification.

* Blue **CHECKING** means pending evidence. Green **VERIFIED** means the existing
  two-source identity gate passed on a continuous track, including any requested
  printed expiry date. Repeated REF alone never becomes green.
* The same model and sibling shortlist are retained. Angiolite labels are tried
  in their readable orientation first. Every fourth unsuccessful/continuing read
  uses the normal multi-orientation, full-spine and hard-barcode fallback.
  Confidence thresholds, REF rules and verification scores are unchanged.
* After identity verification, missing details continue to be read in the
  background. Loss, stale association or contradictory evidence removes green.
  This view does not emit retrieval commands or update inventory.
* A crop panel sits below the camera image. Actual number crops, full label,
  spine, raw readings and `result.json` are saved under the session's
  `verification/`. Each `reading_N/` contains the reader inputs for that same
  track/epoch, rather than substituting a later camera image.
* `identity_verified` and `all_fields_read` are separate. Unknown brand, GTIN,
  LOT or expiry stays null. REF-derived dimensions are labelled `(REF)` until
  directly parsed from size text. A barcode-like texture is only a saved
  `barcode_candidate.png`; only a decoded polygon is displayed as barcode proof.

Read `target_view.jsonl`: `VERIFIED.wall_seconds` measures the decision from the
first frame/new request after model startup; the following `EVIDENCE_SAVED`
timestamp includes crop export. `seconds` is video time on a replay, not latency.

Charged-laptop replays of the owner's latest 18-Oct recordings (AC status 1 before
and after each run), with video recording enabled:

| Request | Time to VERIFIED | Verified crops ready |
| --- | --- | --- |
| Angiolite 2.5 x 29, first run | 4.837 s | 4.985 s |
| Angiolite 2.5 x 29, repeat | 1.722 s | 1.784 s |
| Angiolite 4 x 19, first run | 1.749 s | 1.818 s |
| Angiolite 4 x 19, repeat | 1.714 s | 1.783 s |
| Absent Angiolite 4 x 24 | No VERIFIED in 60 frames | None |

Final checks with enlarged numeral crops in the panel: 2.5 x 29 **1.698 s**
(crops ready **1.747 s**), 4 x 19 **1.730 s** (crops ready **1.804 s**).
See `panel_final_results.json` / `panel_summary.json` in the same scratch folder.
Full regression suite: **409 passed in 29.08 s** (`pytest_final.log`). Launcher
startup/exit and ASCII/CRLF passed; the master inventory hash is unchanged.

All positive decisions use text + REF. **The consistent 1-2 s requirement is
not met**: the 2.5 numeral has low OCR confidence in some frames. Enlarging its
crop did not lift confidence above the unchanged 0.90 threshold. Barcode/GTIN,
LOT and all requested fields are not yet verified; no run claims completion of
the whole reference diagram. Initial non-optimized diagnostics took about 9.7-10 s
on battery and are not a controlled speed comparison with these AC runs.

Evidence: `results/verified_20261019/final_results.json`, `summary.json`, per-run
`reads.csv`, `view.mp4`, `preview.jpg` and `verification/`. These are a small replay
sample, not a live guarantee or an accuracy benchmark. For the next live run,
keep the charger connected and the camera still; move closer or expose the
barcode-bearing face so the missing details can actually be read. Keep the
entire target in frame. Do not treat green identity alone as complete field verification.

## Earlier visual-only workflow (18 October)

The earlier `--lock-only` workflow used `FOCUS_WEBCAM.bat` / DEMO **F**.
The launcher has no stress test. It uses the existing webcam-fine-tuned YOLO26
OpenVINO model at **544 x 960**, exactly the candidate requested by the owner.
The regular webcam/RealSense profiles and the physical PICK verifier are unchanged.

1. Connect the charger and USB webcam; close other camera applications.
2. Choose **1** for preflight.
3. Choose **2** for `angiolite 2.5 x 29`, or **3** for `angiolite 4 x 19`.
4. Keep the camera window open. Type the next full product/size request in the
   terminal to change targets without loading the models again. **q** in the
   image closes the camera; **o** discards prior readings and reads again.

Menu **4** accepts an arbitrary request, including a size absent from the shelf.
For the recorded 20-box local setup, `angiolite 4 x 24` is absent and must not lock.
It is present in the separate Drive reference photo; absence depends on the scene.

Every launch creates `results/focus_<timestamp>/` with a config, inventory copy,
raw video, annotated video, reading log and `target_view.jsonl`. CHECKING is
provisional; LOCKED requires repeated complete observed support for the requested
size and, if requested, its date. Visual lock never emits PICK, TAKEN or grasp
commands. Two independent sources remain required in the separate physical workflow.

## What changed

* Colour-band comparison tolerates changes in lighting and ranks the same product
  family together. It never supplies identity evidence or decides the size.
* A shortlist of up to two similar cartons is read with two OCR workers, one CPU
  thread each. Both Angiolite sizes can be checked without waiting for a guessed
  sibling to be read twice first. Exhausted/weak appearance hints fall back to
  full search; they do not establish that the requested item is absent.
* The visual reader stops after a complete printed source. It skips barcode
  decoding and extra rotations/full-spine reads whose only purpose is obtaining
  another source for PICK. Missing details, conflicts and requested dates still
  require reading. The ordinary verified reader retains its original policy.
* Existing evidence is re-evaluated when the request changes on continuously
  tracked boxes. Lost or stale tracks cannot reuse their identity. This cache is
  temporary and does not modify the inventory.
* Outline work waits until the target is established. End refinement can use one
  clearly observed side when its same-colour neighbour hides the other; the
  uncertain side stays at the detector line. If a lateral adjustment crosses a
  neighbour, a separate attempt adjusts only the ends. Ambiguous boundaries still
  fall back. These display polygons never replace OCR or grasp geometry.

## Measured on 18 October 2026

Charger connected, requested model, one replay process at a time. These are
**elapsed wall seconds from the first frame after model initialization**, through
repeated matching readings and the first displayed full target outline.
No cached target identity was supplied at startup. Model/camera startup is not
included; this is recorded-video validation, not a physical live-camera guarantee.

| Recording / request | Final wall time to visual lock |
| --- | --- |
| 18 Oct: Angiolite 2.5 x 29, two runs | 1.557 s, 1.507 s |
| 18 Oct: Angiolite 4 x 19, two runs | 1.593 s, 1.841 s |
| 17 Oct: Angiolite 2.5 x 29 | 1.947 s |
| 17 Oct: Angiolite 4 x 19 | 2.115 s |
| 18 Oct: Angiolite 4 x 24, absent | No visual lock in the 60-frame check |

The pre-change 18-Oct replay of 2.5 x 29 took 28.27 s with the same YOLO26
candidate. It read unrelated products and repeatedly searched for more evidence.
This is a small diagnostic comparison, not a latency distribution or an accuracy
benchmark. The 17-Oct recording contributed frames to Claude's webcam fine-tune;
do not describe it as held out. Claude's training/reference outlines still need
owner review. No new training or inventory verification was performed here.

Evidence: `results/codex_speed_20261018/final_results.json`, per-run `run.log`,
`reads.csv`, `target_view.jsonl`, and `view.mp4`. The 1-2 s target was met in the
four latest-session trials; the older view has a 2.115 s case. Different poses,
blur, unreadable detail or many identical designs can still take longer.

Full regression suite: **397 passed in 29.35 s** (`pytest_full.log` in the same
scratch folder). Launcher startup/exit and ASCII/CRLF checks passed. The master
inventory hash remained unchanged; positive replays produced no PICK/grasp files.

## Drive review and remaining validation

The owner's material folder contains product photos, including the
[Angiolite reference](https://drive.google.com/file/d/1tkGPZbkrMuxfy2r7jiThQhxOBOBw73za/view).
A single-image audit read REF sizes from 22 detected cartons: the 4 x 19 request
selected one supported carton, 4 x 24 matched four, and 2.5 x 29 matched none.
This is single-image support, not repeated-read lock or PICK. Training overlap
for this photo is unknown; it is not claimed as independent accuracy validation.
The annotated comparison and readings are in `drive_audit/` under the scratch folder.

The supplied Weekly folder's Vision section contains reports/presentations; the
interactive report recaps June-August work. The speed results above use the actual
local 17/18-Oct recordings, not those historical report claims.

Next, run the focused launcher at the current physical webcam pose. Check both
sizes, an absent size, and loss/reacquisition after moving or occluding a box.
Keep that session's files for review. Bounds still need owner inspection at the
top flaps and touching same-colour sides. This change does not guarantee exact
corners or 1-2 s cold acquisition for an entire shelf of 22 near-identical cartons.

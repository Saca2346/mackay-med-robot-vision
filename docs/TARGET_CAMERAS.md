# Verified target testing: webcam and RealSense

For the 22 October single-carton calibration fix and separately evaluated D455 detector profile,
see [SINGLE_BOX_CAMERAS.md](SINGLE_BOX_CAMERAS.md). It includes the current CMD commands and limits.

20 October trained webcam candidate and pixel-stable outlines: run
`FOCUS_WEBCAM_STABLE.bat`. See [STABLE_WEBCAM.md](STABLE_WEBCAM.md) for the
separate model profile and its recorded validation evidence.

From `C:\Users\sacab\Downloads\mackay_vision\code`, run:

```bat
TEST_TARGET.bat
```

Choose **1** for the ordinary external webcam or **2** for RealSense. Then choose
**1** for preflight, **2** for Angiolite 2.5 x 29, **3** for 4 x 19, or **4** to
enter another request. `FOCUS_WEBCAM.bat` and `FOCUS_REALSENSE.bat` open the same
menu with the camera already selected.

Direct CMD commands:

```bat
venv\Scripts\python.exe scripts\run_target.py --camera webcam --index 1 --request "angiolite 4 x 19"
venv\Scripts\python.exe scripts\run_target.py --camera realsense --request "angiolite 4 x 19"
```

Keep the window open and type the next request in the same terminal to reuse the
loaded models. **S** saves a paired original/annotated photo; **Q** closes and
finishes the recordings. The camera must show both ends of the target with sharp
printed numerals. Connect the charger and close competing camera applications.

| Camera | Input/profile | Detector |
| --- | --- | --- |
| Ordinary webcam | DSHOW, 1920 x 1080, device #1 on this laptop; change `--index` if necessary | Existing webcam-specific YOLO26 544 x 960 |
| RealSense | RealSense SDK colour stream; original RealSense resolution/profile | Existing v4 OpenVINO 544 x 960 |

The two camera profiles share the verified target UI and evidence format. Each
uses its own detector; the webcam fine-tune is not substituted for the RealSense
detector. The current shelf tests use RGB identity verification.

On 21 October the external device was identified by DirectShow as **Logitech
Webcam C930e**. A driver-negotiation issue left the actual format at YUY2 even
though MJPG was requested and `CAP_PROP_FPS` reported 20; measured delivery was
5.007 FPS. The DirectShow opener now reapplies the requested FOURCC after
resolution/FPS negotiation. Through the normal `open_camera` entry, the unchanged
1920 x 1080 / 20 FPS profile delivered MJPG at **19.897 actual FPS**, with 100
frames written and no recorder drops in the short test. RealSense/MSMF opening
is unchanged. Evidence: `results/sled_barcode_ac_20261021/camera_audit/REPORT.md`.
Camera FPS is separate from time to VERIFIED; the new short positive replay
verified at 2.369 s, so this fix alone does not certify the 1–2 s target.

## Output and timing

Each launch creates `results/target_<camera>_<timestamp>/` with a copied inventory
and session config. Every run has a `take_<timestamp>/` containing:

* `camera_raw.mp4` and `.times.csv` for live capture; existing videos remain
  read-only when using `--source` for replay.
* `view.mp4`, `cmd.txt`, `readings.csv` and `target_view.jsonl`.
* `photos/`: paired original/annotated photos and per-photo metadata.
* `verification/`: actual OCR inputs, product/REF and size-band crops, individual
  numerals, panels and `result.json` with observed/derived/missing fields.

Green requires the existing independent-source/date gate. Repeated REF alone
cannot authorize green. The panel distinguishes identity VERIFIED from missing
barcode/GTIN, LOT, expiry or brand. Inventory remains isolated and no retrieval
command is emitted by this view.

The default **2 s target is measured, not forced**. `target_view.jsonl` records
`wall_seconds`, `latency_target_seconds` and `within_target`; a verified result
after 2 s says **over 2s target**. The following `EVIDENCE_SAVED` timestamp also
includes crop export. Timing starts at the first frame/new request after model
initialization. Annotated playback is fixed-rate; use the logs/raw timing CSV.

For a stricter measurement, use `--target-seconds 1`. This changes the timing
indicator only. It never marks an unverified result green or stops waiting for
valid evidence. Typed subsequent requests re-evaluate observations on continuous
tracks; logs label those as cached/warm request timing. **O** starts fresh reads.

Angiolite's focused reader first tries narrow product/REF and size bands, then
uses the detected label geometry when those bands are unreadable. It orders numeral
and unit tokens using their actual positions, and ignores tiny LOT/date fragments
when assigning large size fields. Values still come from OCR at the unchanged
confidence threshold. An unreadable layout uses the original orientation/full
spine reader; requested dates retain the original gate. Reading priority finishes
a supported target before spending more OCR on its siblings. Recognition warmup
runs before timed frames and is discarded, never used as evidence.

Pending crop exports wait until verification or the time target; state photos,
raw recordings and per-read logs still retain pending results. Two OCR workers
are the default; use `--workers 1` if memory is constrained and remeasure latency.
Unreadable, cropped or reflected print can exceed the target. Do not describe this
as a universal 1-2 s guarantee, especially for a camera not yet tested live.

Current validation and source logs: `results/speed_dual_20261020/`. RealSense live
validation is deferred at the owner's request; its profile can be checked using
the existing RealSense recording with `--source`.

Final 20 October sequential webcam checks on AC power: Angiolite 2.5 x 29 reached
VERIFIED in **1.408 and 1.726 s**, with crop export complete at 1.490 and 1.815 s.
Angiolite 4 x 19 took **2.496 and 2.759 s**; absent 4 x 24 remained unverified in
the 40-frame control. The separate RealSense replay took 4.708 wall seconds.
These exclude startup and use fresh reads. Full suite: **433 passed**. The report
and links to the recordings, CMD transcripts, photos and crops are in
`results/speed_dual_20261020/REPORT.md`. The common <=2 s requirement remains open.

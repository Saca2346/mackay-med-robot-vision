# One-by-one shelf-box evidence review

The 20 reference inventory rows are tested as separate requests. A stored date
is retained in the request, including the four Permanent Sled Bag instances.
Inventory status is reference data; it does not mark any new trial verified.
Machine verification still needs the original independent-source/date gates.

Open `results/all_boxes_corrected_20261021/REVIEW.html` in a browser. Each card
has its exact request, machine result, video, CMD transcript, paired photographs,
actual reader inputs and nullable observed fields. Compare the physical box,
its full outline, size/REF sources, and requested printed date. The shelf map and
proposed source spines are assistant references awaiting your review; they never
enter tracking, OCR, matching, or inventory.

Expand **Actual reader crops, observed fields and provenance** to see the saved
spine, OCR input, size band and any located barcode. A barcode candidate image
does not mean a successful decode. Browser videos are H.264 copies of the full
retimed evidence, with original encoded videos and frame timing retained.

Choose **Accept evidence**, **Reject evidence**, or **Need closer image / barcode**
and add notes. Export the owner review CSV or JSON. Decisions start **Pending**,
remain separate from machine results, and never change inventory or issue a
retrieval command. Browser storage is optional; export before closing the page.

The reports distinguish first machine verification, first green frame ready,
the timestamp of that source frame, export readiness and later verification
losses. The <=2 s count requires charger eligibility and both recorded wall and
capture timestamps within the deadline. It is not a fresh live/monitor timing
certificate. Model initialization and complete per-field reading are separate.
All supplied October recordings overlap training; these checks are regression
evidence, not held-out accuracy. Missing GTIN/barcode, LOT, expiry or brand fields
remain unknown. An undecoded barcode candidate is not barcode proof.
The follow-up action CSV separates `sources_at_first_green` from
`last_observed_sources`: later partial detail reads need not repeat every source
that established the original identity.

## Repeating a recorded batch

From `C:\Users\sacab\Downloads\mackay_vision\code`:

```bat
venv\Scripts\python.exe scripts\validate_box_targets.py --source results\outline_stability_20261020\capture\training.mp4 --config config\box_pipeline_config_all_targets.yaml
```

A new folder under `results/` preserves the inputs/model/code hashes and creates
one fresh process/tracker for each target. Runtime source snapshots are retained;
input/model/source hashes are checked before and after each trial. A changed
shared file stops the batch with an `input_drift.json` explanation. The default limit is 80 processed
frames or clip EOF; failed and unverified trials remain visible. Camera capture
timestamps are used rather than the MP4's nominal FPS. No stress menu is involved.

Use `--only K04` for one box or `--only K01,K03,K05,K17` for the dated Sled Bags.
Use a new output folder for a retry after code/model/input changes. `--resume`
accepts only an identical plan and fingerprints and retains completed trials.
The runner copies inventory per take and checks the master reference remains
unchanged. It never writes into `results/bukti_*`.

## Later live testing

Connect the charger and keep the complete carton ends and printed labels in
frame. Run:

```bat
.\FOCUS_ALL_BOXES.bat
```

Select **4** and paste the exact request from the review card, including its date.
**S** saves another photo and **Q** closes and flushes recordings. Typing the next
request keeps models loaded and labels reuse of cached observations; **O** starts
fresh observations for the current request. The original Angiolite launcher
`FOCUS_WEBCAM_STABLE.bat` remains available. RealSense live testing is separate.

The all-target profile enables the tested printed-region path for Angiolite,
Essential Pro and Xperience Pro. Other products retain the full reader. The
precise B544 detector is unchanged. Real printed unit positions can remove
unpaired date digits from numeric-field parsing; raw tokens and selected field
tokens are retained for inspection. Confidence values are not reduced, and
requested values never supply observed numbers. Two consistent different dates
can lower a sibling's reading priority without changing its matching decision.

The final batch also includes Claude's split-decimal parser correction and
verified-track hold correction. The earlier baseline and intermediate reviews
are retained in `results/all_boxes_review_20261020` and
`results/all_boxes_validation_20261020`. They are separate runs and cannot be
combined to select only the fastest result for each box. The intermediate
batch finished before the hold correction; its version note preserves the exact
original core-source hashes and explains why it is not the final-version test.

The 21 October correction also preserves Claude's Terumo REF suffix rule and
rejects invalid numeric interpretations of slash-separated LOT/date fragments
such as `20/0L92` and `20/0192`. Literal sizes such as `3/24` remain valid; a real
conflicting `20 mm` is still evidence against a requested `40 mm`. Original OCR
tokens remain in the files. A separate 160-frame Essential Pro replay checks the
previous later verification loss; it is not substituted for the main batch.

The previous complete 20-case review remains at
`results/all_boxes_final_20261020/REVIEW.html`. The partial first follow-up in
`results/all_boxes_followup_20261021` was deliberately stopped by the source-hash
guard after the second fragment spelling was found. It is not a completed
validation or a final timing result.

The follow-up also includes `PHOTO_REVIEW.html`: separate runs of the full photo
pipeline on existing higher-resolution reference images, with date/size negative
controls. Its legacy offline PICK label is a photo decision, not a robot command.
Those observations never fill fields or replace results from the current webcam
recording. `NEEDS_CLOSER_VIEW.md` lists the remaining capture requirements per box.

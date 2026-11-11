# CVAT and expiry review, 14 October 2026

The supplied `baru_20260928_16.zip` contains CVAT 1.1 XML annotations and 33 JPEGs. Rotations were preserved when converting its rectangles into OBB polygons. The original ZIP was left intact.

The export needed corrections: thin accidental shapes, date-only size panels, missing visible product panels, and a Ryurei spine extending into the background. The server already contained the imported and converted batch when checked. Its reviewed geometry was retained; three missing product/logo panels were added after visual inspection.

## Reviewed master

The server is the label master. At the start of this review it had 99 completed expiry specs and 97 automatic proposals, ahead of the earlier guide's 51/145 count. All 97 remaining photos were reviewed individually using full-resolution spine strips and enlarged date crops. Unreadable or hidden dates were left without a date annotation rather than inferred from neighbouring packages.

The reviewed master now contains:

| Item | Count |
| --- | ---: |
| Completed photo specs | 196 |
| Automatic expiry specs remaining | 0 |
| Box annotations | 1,859 |
| Size-panel annotations | 2,306 |
| Expiry annotations | 1,489 |

Five photos needed additional spine corrections before training. In two shelf photos, the old spine boundaries and date associations crossed neighbouring packages. Ten spine regions were corrected, including Agent and the four distinct Sled Bag dates. The existing geometry gate and its previously accepted photo-border warnings were preserved.

## Validation and evidence

* Server readiness: 196 specs checked; zero new/unaccepted problems, zero crashes; 49 existing accepted border warnings. All 196 photos qualify as seeds. Burst-separated validation folds contain 66/65/65 photos.
* The laptop received 789 master files, verified against server SHA-256 hashes. Its previous labels were backed up first.
* The detector verification evaluator now ignores the expiry class when extracting box/size predictions. A regression test verifies three-class output and both identity thresholds. Full code suite: 271 passed.
* Local audit directory: `../labeling/audit_local/cvat_20261014/` relative to the workspace's `code` directory. It contains visual decisions, proposed overlays, before/after payloads, backups, manifests and the local geometry log. These assets are excluded from Git.
* Server audit directory: `~/Downloads/prelabel_size_label_full/audit/codex_20261014/`. Backups are in `~/Downloads/cadangan_codex_*20261014*` and `~/Downloads/cadangan_sebelum_3kelas_20261014_0952/`.

## Training and remaining validation

Training was started on the server with `tools/mulai_training_3kelas.sh`: three 120-epoch folds and a 200-epoch final model, batch 8. Previous runs were backed up. The completion check requires the final completion message, `best_model.json` with final weights, and `runs/best_final/done`; the earlier `=== model final` message alone only announces the start of final training.

The completed three-fold comparison does **not** pass the replacement gate:

| Class | v6 mAP50-95 / recall (156 photos) | New mAP50-95 / recall (196 photos) | Decision |
| --- | --- | --- | --- |
| box | 0.706 / 0.948 | 0.666 / 0.941 | Fail: mAP decrease exceeds 0.02 |
| size_label | 0.719 / 0.953 | 0.697 / 0.945 | Fail: mAP decrease exceeds 0.02 |
| expiry_label | No baseline | 0.762 / 0.949 | New class |

The datasets differ in size and content; v6 also used five 200-epoch folds, while the prescribed quick run uses three 120-epoch folds. These figures do not isolate the cause of the decrease. The prescribed gate still fails; the model must not replace v4/v6. Final training completed at 200 epochs; the full run took 70 minutes. The new weights are retained as an experiment artifact.

Out-of-fold `eval_verify_detector.py` completed at box/size confidence 0.35/0.45 on the completed fold models before final training finished. Its GT labels were frozen from the reviewed snapshot. It detected 116/123 boxes (94.3%); the remaining seven were missed. Verification produced **zero wrong MATCH** over 1,276 wrong-identity requests, including 87 same-product/different-size requests. Correct-identity requests produced 13 MATCH, 42 candidates and 61 requests for confirmation. This evaluation does not test date-specific live PICK requests.

Results and exact completion status are recorded in `pipeline_status.json`, `model_comparison.log` and `safety_evaluation.log` in the server audit directory.

The owner later confirmed that Claude was also reviewing labels. No further master-label edits were made after that confirmation. Master hashes are monitored. All 1,176 label copies in the three training folds and final dataset matched the saved snapshot; the evaluation GT is frozen from that same snapshot. Legacy fold3/fold4 exports are preserved and excluded from this three-fold run.

The laptop demo still uses v4. Physical D455 shelf tests, date-specific Sled Bag requests (2, 16, 0), and new shelf recordings remain to be performed at the rack. No inventory verification or taken state was changed.

Final completion was verified from the log, `best_model.json`, and the done tag `yolo11s-obb@180|53889695cdf219b0|200|1280`. The downloaded candidate weights match server SHA-256 `c627550672c21e51795e0d9f47c43749c558d93c80378e7cc06c58194b50391b`. Final master hashes still matched the reviewed snapshot. Local outputs include `FINAL_SUMMARY.json`, `pipeline_status.json`, `best_model.json`, `model_3classes_candidate_20261014.pt`, and `baru_20260928_16_reviewed_3classes.zip` in the audit directory. The final training-set score is not used as a replacement for the held-out CV gate.

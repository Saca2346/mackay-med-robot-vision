# Remaining work after 14 Oct 2026: step by step

Indonesian version (the one to work from): `docs/id/PANDUAN_SISA_PEKERJAAN.md`. Work from top to bottom; tick the list at the end.

| # | Task | Where | Time | Done when |
| --- | --- | --- | --- | --- |
| 0 | Install the new review tools on the server | laptop, 1 command | 5 min | output `done` |
| 1 | Correct the boxes of the 33 close-up photos | CVAT on the server, laptop browser | 1.5-2 h | `33 converted, 0 to check` |
| 2 | Review the date labels of 145 photos, then the 3-class training | server terminal | 5-7 h review (6 sessions) + 1.5 h automatic training | `bandingkan_model.py` output sent to Claude |
| 3 | Live test and shelf data in front of the shelf | laptop + D455 | 2-3 h per visit | session folder + `LAPORAN_SESI.md` |

Task 3 does not depend on tasks 1-2.

## Server terminal (tasks 0-2)

VS Code Remote-SSH (`F1` -> `Remote-SSH: Connect to Host...` -> `trse-server`) or `ssh trse-server` in a laptop cmd. Then always:
```bash
cd ~/Downloads/prelabel_size_label_full && source ~/trse_venv/bin/activate
```

## Task 0. Install the new review tools (5 min)

The package is already uploaded. Laptop cmd:
```bat
cd C:\Users\sacab\Downloads\mackay_vision\labeling
RUN_ON_SERVER.bat langkah9_tools_review
```
Last line `done`. It installs `tools/reviewed_to_spec.py` (new: a CVAT result back into the spec, so the date tools and the
training use the corrected boxes) and the updated `tools/review_tanggal.py` (skips photos whose boxes are still a draft, prints
the sheet path).

## Task 1. Correct the boxes of the 33 close-up photos (1.5-2 h)

One box per photo (28 Sep, 16h) in front of a desk, a monitor and a drink bottle. The v4 drafts are often wrong (bottle /
monitor as a box, box only over the coloured part). Only `box` and `size_label` here; dates come in task 2.

1. **CVAT in the laptop browser**: laptop cmd `ssh -N -L 8080:localhost:8080 trse-server` (keep open), browser
   `http://localhost:8080`, log in.
2. **Project** `TRSE box size_label` (labels `box`, `size_label`, type Rectangle) if it does not exist yet.
3. **Task** `baru_20260928_16h`: upload the 33 photos from
   `C:\Users\sacab\Downloads\mackay_vision\labeling\batches\baru_20260928_16h\images` (identical to the server `raw/`, hashes
   checked 14 Oct); Advanced configuration -> Image quality 95.
4. **Drafts**: Actions -> Upload annotations, format CVAT 1.1, `cvat_annotations.xml` from the same batch folder.
5. **Correct** each photo (F / D next / previous, Ctrl+S, Delete, N, rotation handle):
   delete boxes on the bottle, monitor, keyboard, desk or cap; box = the whole visible spine, end to end (up to the photo
   edge when it runs out); one `size_label` on name + size (Terumo two, iVascular the coloured panel, Sled Bag logo +
   "Permanent Sled Bag"); nothing for REF / LOT / barcode / date alone; no `expiry_label`; slivers < 1 % not labelled.
   Most attention: 162600_587 (no box: draw one); 162335_447, 163033_361, 163052_907 (a box collapsed to a line: delete);
   162031_994, 162154_336, 162159_717, 163611_481, 163640_459, 163052_907 (extra box, usually the bottle);
   162552_123, 163517_241, 163605_296, 163710_214 (box too short).
6. **Export**: Actions -> Export task dataset, **Ultralytics YOLO Oriented Bounding Boxes 1.0**, Save images off; download from
   Requests; laptop cmd `scp "%USERPROFILE%\Downloads\FILE.zip" trse-server:~/Downloads/baru_16h_cvat.zip`.
7. **Import** (server):
   ```bash
   python tools/import_reviewed.py ~/Downloads/baru_16h_cvat.zip --dry
   python tools/import_reviewed.py ~/Downloads/baru_16h_cvat.zip
   python tools/reviewed_to_spec.py --batch baru_20260928_16h
   ```
   `KESALAHAN` = nothing written, fix in CVAT; `BELUM DIKOREKSI` = photo not corrected (`--allow-unchanged` only if the draft
   was really right). Done when `33 converted, 0 to check`; `grep -l "SEDANG DIKERJAKAN" label_specs/*.json | wc -l` gives 0.

## Task 2. Date labels, then the 3-class training

145 photos are `auto`: 112 old ones (incl. 26 reopened on 12 Oct) and the 33 of task 1. A visible date without a label is
learned as "not a date": never `ok` when unsure.

1. `python tools/review_tanggal.py` (Ubuntu desktop through Remmina opens the sheet itself; in VS Code Remote-SSH Ctrl+click the
   printed `lembar:` path). Header: `(selesai 51/196, sisa 145)`; `keluar` stops, progress is kept.
2. Sheet: one strip per box, far end left, near end right, box number below; magenta = existing label, cyan = new proposal,
   red frame = PERIKSA (sharp, no label). Expiry = the date next to the hourglass; iVascular: only the middle line
   (LOT top, expiry middle, production bottom).
3. Commands: `tgl N DATE` (date misread), `hapus N` (not an expiry date), `tambah N T1 T2 DATE` (date visible, no label;
   T along the strip 0 = left, 1 = right; optional `V1 V2` across for a tighter label), `buang N` (blurred), `kosong N`
   (red frame, no date visible), `ok`, `lewati`, `keluar`. Suspicious known dates: 2038-01-31, 2037-06-30, 2036-09-30,
   2026-09-30.
4. Pace: 25 photos per session (45-60 min), about 6 sessions. Before training at least all 112 old photos.
5. `CEK=1 bash tools/mulai_training_3kelas.sh` -> `SEMUA SYARAT TERPENUHI`. Open PERIKSA boxes on a `selesai` photo:
   `python tools/set_expiry.py NAME --redo` and review again; too few seeds or a fold < 5: review photos of other dates;
   VRAM: try later.
6. `bash tools/mulai_training_3kelas.sh`, then `tail -3 logs/train_3kelas.log` until "model final" (1-1.5 h; old results are
   backed up first).
7. Send to Claude: `python tools/bandingkan_model.py`, `grep -A6 "Hasil cross-validation" logs/train_3kelas.log`,
   `cat logs/best_model.json`. A new model replaces v6 only if box and size_label mAP50-95 and recall drop by at most 0.02 and
   the safety test (0 wrong MATCH) passes; the demo model v4 stays until then.

## Task 3. In front of the shelf (laptop + D455)

1. **Prepare** (10 min): charger, Best performance, VS Code and browser closed; D455 on USB 3 (`TEST_REALSENSE.bat` menu 1
   shows 3.x), tripod, all 20 boxes, a phone photo of the shelf.
2. **Recorded session**: `cd C:\Users\sacab\Downloads\mackay_vision\code`, `DEMO.bat`, **S**; pass criteria per step in
   `docs/TEST_SESSION_GUIDE.md`. Expected: iVascular (5, 9, 14) PICK at shelf distance, other brands candidate.
3. **Sled Bag dates** (fix of 14 Oct), in step 6: type `2` (K03, 2029-03-10: PICK only after its date is read), `16`
   (K17, 2029-03-12, one digit from K03: PICK K17, never K03), `0` (any Sled Bag: earliest date first). Bring the camera close
   until the date next to the hourglass is clearly readable on the live view, hold 3-5 s. Note per number: right box PICK /
   candidate / WRONG box PICK. **A wrong-box PICK: stop and report at once.**
4. **Shelf data**: `TEST_REALSENSE.bat` -> **D**, sessions `s1_far` (whole shelf) and `s2_mid` (box width >= 40 px, left half
   then right half); calibrate until no FAIL, record 1-2 min (still 3 s per position), `q` stops. Then
   `venv\Scripts\python.exe scripts\ambil_frame_data.py results\data_rak\s1_far.mp4 results\data_rak\s2_mid.mp4`
   (frames in `labeling\raw_d455\`) and tell Claude, who prepares drafts and a CVAT batch as in task 1.
5. **Afterwards**: give Claude the `results\sesi_<date_time>\` folder name for the summary.

## Checklist

- [x] 0 `langkah9_tools_review` -> `done` (14 Oct 06:43, checked: tools installed, queue 51/163)
- [ ] 1 CVAT task, 33 photos corrected, import + `reviewed_to_spec.py` -> `33 converted, 0 to check`
- [ ] 2 date review: session 1 [ ] 2 [ ] 3 [ ] 4 [ ] 5 [ ] 6 [ ]; `CEK=1` passes [ ]; training done [ ]; results sent [ ]
- [ ] 3 session S [ ]; Sled Bag 2 [ ] 16 [ ] 0 [ ]; shelf data `s1_far` [ ] `s2_mid` [ ], frames extracted [ ]

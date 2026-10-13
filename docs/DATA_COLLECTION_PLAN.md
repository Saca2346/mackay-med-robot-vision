# Image data plan for the next training round (14 Oct 2026)

Goal: a detector that works on the **deployment camera** (RealSense D455, 1280x800) at shelf distance, for all 20 boxes,
and an expiry-label finder for identical boxes (Sled Bag).

## 1. What we have (counted on 14 Oct)

| Item | Count | Note |
| --- | --- | --- |
| Labelled photos (`labels_final_obb`) | **163** (104 from 23 Sep, 54 from 28 Sep, 5 from 1 Oct) | phone photos of boxes in cardboard, not the shelf |
| `box` instances | 1,826 | |
| `size_label` instances | 2,277 | |
| `expiry_label` instances | 719, in 76 photos only | 87 photos still in the review pool |
| Unlabelled photos already on disk | **33** (`New image\`, 28 Sep 16h) | never entered the dataset |
| D455 frames from the real shelf | **0** | the biggest gap: every test runs on D455 images |

Measured weak spots that more data should fix:
* 3-class model: `box` mAP50-95 0.564 with ~50 training photos per fold vs 0.706 with ~125 (v6). Fewer photos = clearly worse.
* At shelf distance (box 24-36 px wide) the size text reads 7-34 % of the time; the second source of non-iVascular products is not read.
* Sled Bag dates are unreadable at shelf distance, so the dated request can only PICK from close up.

## 2. Targets

The Ultralytics training guide suggests about 1,500 images and 10,000 labelled objects per class for best results. One
D455 shelf frame holds 20 boxes and about 20-25 size labels, so a few hundred frames reach the object count quickly; the
number of **different conditions** is what limits us, not the number of frames.

| Set | Frames / photos | Objects (approx.) | Why |
| --- | --- | --- | --- |
| A. D455 shelf sessions (menu D) | **400** selected frames | 8,000 box, 9,000 size_label | deployment camera and distance |
| B. D455 close-up of labels (Sled Bag dates, REF, barcode) | **120** | 400 expiry_label, REF areas | dated requests and second source |
| C. The 33 unlabelled phone photos | 33 | ~400 box | already taken, free gain |
| D. Finish the 87 expiry-pool photos | 87 | ~700 expiry_label | the 3-class model was trained on 76 photos |
| E. Negatives: shelf without boxes, other shelves, hands only | **30** (5-10 % of A) | 0 | fewer false boxes (monitor, shelf post were counted) |
| F. New brands (only if the team decides to add them) | 40 per brand | | e.g. CHROMA, SeQuent (see `results/demo_20261012/foto_baru`) |

Total new labelling: about 670 frames. With auto-proposals from v4 plus review (1-2 min per frame), that is 12-20 hours.

## 3. Conditions to vary in set A (each session changes one or two)

| Condition | Values |
| --- | --- |
| Camera distance | far (whole shelf), mid (box 30-40 px), close (3-5 boxes) - sessions s1/s2/s3 of `docs/DATA_RAK_PROTOCOL.md` |
| Shelf arrangement | 5 different orders of the 20 boxes (shuffle between sessions; photo each arrangement) |
| Light | room lights on / off, daylight morning / afternoon, a lamp from the side (glare on plastic) |
| Camera angle | straight, 10-15 deg left / right, slightly from above |
| Occlusion and gaps | one or two boxes missing, a hand in front, a box leaning, a box put back at the image edge |
| Background | monitor behind the shelf on / off (it was detected as a box), cardboard, wall |

## 4. Rules for capturing (so the frames are usable)

1. Run the calibration first (menu D does it). No FAIL; record the calibration card with the session.
2. Camera still at each position for 2-3 s, then move slowly; blurred frames are dropped by `scripts/ambil_frame_data.py`.
3. Never put patient data, faces, screens with personal data or hospital paperwork in view.
4. Do not stamp dates or text on the pictures; keep the original files.
5. One folder per session; the session name says distance and condition (e.g. `s4_mid_lightsoff`).
6. Keep validation honest: whole sessions go to validation (never frames of the same session in train and val).

## 5. Labelling rules (same as the labelling guide, short form)

* `box`: the spine only, as a rotated box along the spine; partly hidden boxes labelled where visible; slivers under 1 % of the image width skipped.
* `size_label`: the cluster of product name + size, always inside its box. Terumo: two (name+size, TERUMO logo block). iVascular: the coloured panel. Sled Bag: logo + "Permanent Sled Bag".
* `expiry_label`: the printed date next to the hourglass icon.
* Never a new class, never product names as class names. REF, LOT, barcode alone are not labelled.

## 6. Order of work

1. This week: sets C and E (no new photos needed for C), and 2 shelf sessions of set A (far + mid) with the current shelf.
2. Next week: the remaining set A sessions with changed arrangements and light; set B close-ups.
3. Then finish set D, retrain on the server (same cross-validation as v6 so the numbers are comparable), and compare per class against v4/v6 before using the new model.

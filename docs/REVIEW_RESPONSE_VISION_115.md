# Response to the second-round review (FY115 project), vision part

Source: `(計畫主持人_林顯易)115年度「智慧機器人關鍵技術研發與場域應用專案計畫」複審意見.docx` (project: robot assistant for picking,
placing and opening medical devices in cardiac catheterisation). This file answers only the comments that concern the vision
module and says, for each answer, what is **already measured** and what is **planned**. Items for the arm, AMR, opening and
regulation teams are listed at the end for the PI.

## 1. "If picking accuracy cannot reach 100 %, what is the solution?"

Design answer: the system is built so that an uncertain identification never becomes a pick. Accuracy is enforced by rules,
and every remaining doubt is handed to a person.

| Layer | What it does | Status / evidence |
| --- | --- | --- |
| Two independent sources | A box is accepted (MATCH / PICK) only when two independent sources agree, e.g. printed size text + REF catalogue code, or the GS1 barcode. One source = "candidate", never picked. | In use. Evaluation on 123 labelled boxes: **0 wrong MATCH** in 1,353 readings (v4), 1,309 (v5), 1,287 (v6) |
| Look-alike protection | Boxes of the same product with another size (e.g. Angiolite 4 x 19 vs 2.5 x 29) are rejected by the size + REF rules | Automated tests; live tests 8-13 Oct: no confusion |
| Identical boxes | Four identical Sled Bags differ only by the printed expiry date; a request for one date picks only after that date is read | Implemented 14 Oct (tests); live test pending |
| Inventory cross-check | Only boxes registered and verified by staff can be requested; expired boxes are refused; FEFO order | In use |
| "Needs confirmation" | Unreadable or contradictory boxes are shown in yellow and reported to the operator instead of being guessed | In use |
| Post-pick check | After the pick, the camera verifies the slot is empty (TAKEN) or the box was only moved (cancel and search again) | In use, live-tested 13 Oct |
| Human confirmation | The message to the robot software carries `needs_doctor_confirmation = true`; the final check stays with the clinician | In use (message format) |

Honest limits today: at shelf distance only one brand (iVascular) reaches the two-source PICK; other products stay
"candidate" because their second source is too small to read. Plan: a close-up view for the second source (camera on the arm
or a second camera), more training data from the real camera (see `docs/DATA_COLLECTION_PLAN.md`), and measured
error rates per product reported each quarter. The residual risk is the time cost of a manual confirmation, not a wrong device.

## 2. "Electromagnetic interference in the high-radiation environment: voice commands and visual accuracy"

| Point | Answer | Status |
| --- | --- | --- |
| Camera image | The camera works on visible / infrared light; EMI affects the electronics and cables, not the optics. Measures: shielded USB 3 cables kept short, ferrite cores, the processing computer outside the radiation zone, and a watchdog that stops picking if frames drop or the image quality checks fail (sharpness, motion, box count). | Image-quality checks and the frame watchdog exist; EMI test in the cath lab is planned |
| Voice request | Voice is never the only channel: every spoken request is shown as text with the understood product and size and must be confirmed (touch / pedal) before the robot moves; the text input path already exists. | Text path in use; voice front-end belongs to the HRI team |
| Validation | Repeat the recorded test session (DEMO.bat menu S) inside the cath lab with fluoroscopy on / off and compare detection and reading rates. | Planned for the field phase |

## 3. Packaging and opening challenges (vision side)

Vision supports opening by locating the pouch, its seal and the tear notch. Plan: an extra detection / segmentation step for
pouch edge, seal and notch on close-up images, with depth from the D455 for grasp points. Not started; depends on the opening
team's chosen method and needs its own labelled data.

## 4. Dual-arm grasping and trial scenarios (vision side)

Today the vision output is a rotated 2D box per spine plus the decision; for grasping we will add the 3D pose (D455 depth) of
the chosen box and its neighbours. Trial scenarios that the vision module can already script and record: single request, list of
requests, random requests, box moved instead of taken, box put back at the image edge, identical boxes with different dates.

## 5. Sharing methods, code and data with the National Robotics Research Center

* Code: the vision repository is on GitHub (public, code only), with automated tests and English documentation.
* Data: labelled images in YOLO-OBB format with class rules, label review records and calibration cards per recording session.
  Shared after removing anything hospital-specific (inventory, patient or staff data are never in the images).
* Methods: the evaluation scripts and the cross-validation protocol, so results can be reproduced.

## For the PI: comments outside the vision module

| Comment | Team |
| --- | --- |
| Regulatory plan: ISO 12100, ISO 14971, ISO 13482, ISO 3691-4 (in addition to IRB, device classification, ISO 13485, IEC 60601, IEC 62443) | PI / regulation (vision provides its hazard list: wrong device, wrong expiry, missed recall) |
| Dust-free opening of flexible film pouches without debris | opening / arm team |
| Preparing the operating room within four years: scope per year | PI |
| FPGA only near commercialisation | PI / hardware |
| Cooperation with Taiwan Health Network Platform on human data | PI |

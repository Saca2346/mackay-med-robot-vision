# Architecture roadmap: from the current pipeline to a stronger one

Status: **proposal, nothing here is built or tested** (24-27 Oct 2026, written by Claude Code from the measured results of this
project). Every option names the measured problem it answers, what it costs, the risk, and the evidence that would prove it.
The rules of `AGENTS.md` stay: two independent sources for PICK (plus the owner exception for dated Sled Bags), no new detection
classes, no product names in class files, no inventory writes, no hospital data on GitHub, the doctor confirms, no LLM in the decision.

## 1. Where we are (baseline A)

```mermaid
flowchart LR
  CAM[Webcam C930e / D455] --> CAL[Framing gate: sharpness, still, roll, margins]
  CAL --> DET[YOLO26s-OBB model B, 544x960, OpenVINO]
  DET --> TRK[Tracker + target lock + outline refinement]
  TRK --> OCR[RapidOCR on size label / label band / spine]
  OCR --> PAR[Parsers: size, REF tables, dates, GTIN]
  PAR --> FUS[Evidence atoms, integer weights, window of 4 reads]
  FUS --> GATE[Date gate + two-source rule]
  GATE --> VIEW[Green / blue / none + crops panel]
  VIEW --> GR[object_points + scene cloud npz]
  GR --> LEON[GraspGen-X, arm]
```

Measured limits that drive the roadmap (all from this project, see `docs/HANDOFF.md`):

| Problem | Number | Where it comes from |
|---|---|---|
| D455 cannot read the Accuforce REF | 0 of 90 live crops (spine about 47 px) | optics: too few pixels |
| Webcam REF is intermittent | K16 green in 3 of 5 live batches; K07 read `RM2612` instead of `RM2512` in 4 of 13 reads | OCR character confusions (T/7, 6/5) |
| Live time varies widely | K06 1.50 to 7.42 s, K21 2.14 to 6.16 s over 5 batches; 9 of 35 within 2 s | one OCR pass about 0.2 to 1.5 s on an i3 laptop, read order, luck of the first frames |
| Depth is sparse on spines | about 5 % valid pixels at 0.9 m (every preset) | active stereo on glossy, narrow, parallel surfaces |
| Evidence weights are hand-set | W = 2/1/1/1, "score >= 2" | no calibrated probability of a wrong green |
| Dated Sled Bags rely on digits | one misread digit can turn the date of box A into the date of box B | per-box decision, no shelf-level constraint |
| Few negatives | 0 false green of 88 only bounds the rate to about 3.4 % (95 %) | n too small for a safety claim |

## 2. Target architecture (candidate B, "evidence-ledger pipeline")

```mermaid
flowchart TB
  subgraph Perception
    WIDE[Shelf camera: wide view] --> D1[OBB / segmentation detector]
    D1 --> LOC[Shelf map: slots, tracks]
  end
  subgraph Reading
    LOC --> ROI[Field locators: label band, REF line, date, barcode]
    WRIST[Wrist camera: close-up on demand] --> ROI
    ROI --> REC[Field recognizers with constrained decoding]
  end
  subgraph Evidence
    REC --> LED[Evidence ledger per box: field, value, source, confidence, frame]
    LED --> BAY[Calibrated fusion: likelihood ratios, SPRT stop rule]
    LOC --> JOINT[Shelf-level assignment: one request, one box]
    BAY --> JOINT
  end
  JOINT --> DEC[Decision: green / blue / none, with P(wrong) estimate]
  DEC --> HUM[Doctor confirmation, TM software]
  DEC --> POSE[Segmented cloud or model-based pose]
  POSE --> GRASP[GraspGen-X, arm]
  LED --> AUD[Append-only audit log, owner review pages]
```

The four ideas that change the system (the rest of this document is detail):

1. **Close-up second stage** (wrist camera, "look closer on demand") instead of asking one far camera to read everything.
2. **Constrained recognizers** for the fields we already know the grammar of (REF, date, size) instead of generic OCR plus repairs.
3. **Calibrated sequential fusion** (likelihood ratios, a stopping rule) instead of integer weights and a fixed window.
4. **Shelf-level joint decision** (each request maps to exactly one box; identical siblings must differ in date or size).

## 3. Options by layer (with equivalent alternatives)

Priority: **P1** next 2 to 3 weeks, **P2** next month, **P3** later or only if P1/P2 are not enough.

### 3.1 Reading (answers: REF intermittent, D455 optics, timing)

| Option | Idea | Gain | Cost / risk | Evidence needed | Pri |
|---|---|---|---|---|---|
| **R1 Close-up stage** | Far camera finds the box; the wrist camera (Gemini 305, 4 to 100 cm) or the arm moves 15 to 25 cm closer and reads REF, date and the DataMatrix | removes the optical limit; barcode becomes a real second source (webcam read 4 of 4 DataMatrix at 169 px) | needs the arm (Farhan, Makata), hand-eye calibration, motion time, a collision-safe approach | live test on 11 boxes: REF read rate per box at close range; added seconds | P1 (design), P2 (test) |
| **R2 Constrained decoding** | Recognizer restricted to a grammar: `DC-RM` + diameter code + length + `HHW`, dates `YYYY-MM-DD`, size from the catalogue. Implement as a lexicon-constrained beam over the recognizer's character posteriors, or a fine-tuned recognizer with a small output alphabet per field | replaces ad-hoc maps (T to 7, 6 to 5) by one principled rule; a wrong-looking code cannot be produced | needs per-character posteriors from the OCR (RapidOCR returns text and a score only); or fine-tune | held-out crops: REF exact-match rate, wrong-REF rate (must stay 0) | P1 |
| **R3 Fine-tune the recognizer on our crops** | Fine-tune PP-OCR recognition (or PARSeq) on the label-band, date and REF crops we already saved, with the owner-reviewed text as target | reads our fonts and blur; REF read rate up | needs reviewed text labels (owner), about 1,000+ crops; overfitting to the shelf; labels made by models are not ground truth | K-fold by session, never by frame; report per field | P2 |
| **R4 Multi-frame stacking** | Align the last n crops of the same box (they are already tracked) and average or select the sharpest per region before OCR; learned super-resolution only as a display aid | cheap gain on the narrow spine | alignment error; learned SR can invent characters, so never feed it to the decision | A/B with the same frames: REF read rate, no new wrong reads | P2 |
| **R5 Field-first reading** | Skip whole-spine OCR: locate the label band or REF line by geometry/colour (Terumo band already works) and run only the recognizer on that strip | fewer OCR calls (the logo skip already cut 5 to 4); lower latency | per-brand locator rules grow | latency per request over 5 batches | P1 |
| **R6 VLM as offline auditor** | A vision-language model (e.g. Florence-2) reads crops on the server to find disagreements with the pipeline, never inside the decision | finds label errors and OCR failures at scale | server GPU; hallucination; the 5 Oct meeting says no LLM in the decision | audit report only | P3 |

Equivalent alternatives for the recognizer: PP-OCRv5/v3 fine-tuned (stay inside the current runtime, easiest), PARSeq (strong
accuracy, permuted autoregressive decoding, needs its own export to OpenVINO), Tesseract (fallback, weaker on small print).

### 3.2 Fusion and decision (answers: hand-set weights, one-digit date risk, few negatives)

| Option | Idea | Gain | Cost / risk | Evidence needed | Pri |
|---|---|---|---|---|---|
| **F1 Calibrated likelihood ratios** | Per source and field, estimate from labelled reads the probability that a read value is right, and combine as log-likelihood ratios; green when P(wrong) is below a target | a number with a meaning ("P(wrong) < 0.1 %") and a principled reason for each weight | needs many labelled reads per source; independence is an assumption (two reads from one crop are not independent) | held-out calibration curve (reliability diagram) and the matrix below | P2 |
| **F2 Sequential stopping rule (SPRT)** | Stop reading as soon as the log-likelihood ratio crosses a threshold for "this box" or "not this box" | the time-to-green becomes adaptive; fewer wasted frames; error rates are controlled by the thresholds | needs F1; thresholds must be validated | time to decision and error rate on a sealed test set | P2 |
| **F3 Shelf-level joint assignment** | Treat all boxes and the inventory slots as an assignment problem (Hungarian or min-cost): each box gets one identity; two boxes cannot both be "Sled Bag 2029-03-12" | stops a misread date from matching two boxes or the wrong one; uses the inventory as a constraint, not as evidence | needs a trusted slot map (owner review); the inventory may be stale | replay with injected digit errors: false green before and after | P1 |
| **F4 Conformal guarantee** | Wrap the decision score in conformal prediction to obtain a distribution-free coverage statement on a calibration set | an honest statement of the error rate with finite samples | needs exchangeability (same shelf, same setup); more data | reported with the matrix | P3 |
| **F5 Safety envelope** | Keep the two-source rule as a hard outer gate; the learned fusion may only make a decision stricter than the rule, never looser, until the owner approves | the new fusion cannot raise false green | slightly more blue | A/B: false green must not rise | P1 (principle) |

### 3.3 Detection and segmentation (answers: outline, point cloud for GraspGen-X)

| Option | Idea | Gain | Cost / risk | Evidence needed | Pri |
|---|---|---|---|---|---|
| **D1 Keep YOLO26s-OBB** | Current model B | best measured on our 3 folds (0.746) and shelf (100 % perfect frames) | OBB is only a rectangle | baseline | now |
| **D2 Instance segmentation head** (YOLO26-seg or a tiny mask head) | A mask per spine instead of a rectangle | exact spine pixels for the depth mask and the cloud; tighter outline | new labels (masks) or semi-automatic masks from SAM 2 reviewed by the owner; slower | mask IoU on a held-out day; effect on `object_points` count | P2 |
| **D3 SAM 2 as a label tool** | Use SAM 2 to propose masks and to propagate them over video frames for labeling | cuts labeling time | proposals are not ground truth; owner review stays | labelling time and mask quality on a sample | P2 |
| **D4 RT-DETR / D-FINE** | End-to-end transformer detectors, no NMS | simpler post-processing; strong accuracy | heavier on an i3; no OBB variant by default | same 3 folds and recipe as the 16 Oct comparison, laptop FPS | P3 |

### 3.4 3D and grasp hand-off (answers: 5 % valid depth)

| Option | Idea | Gain | Cost / risk | Evidence needed | Pri |
|---|---|---|---|---|---|
| **G1 Closer depth camera** | Gemini 305 (depth 4 to 100 cm) on the wrist | dense depth on the spine at pick distance | purchase / mount / calibration; focus type of the colour sensor not stated | valid-pixel share at 20 to 40 cm | P1 (test) |
| **G2 Model-based pose** | Use the OBB corners, the shelf plane and the carton dimensions from the catalogue to compute the box pose by PnP; depth only checks it | a grasp pose without dense depth | catalogue dimensions needed (not in the inventory yet); carton tilt | pose error against a measured box | P2 |
| **G3 Depth completion** | Fill sparse depth with a monocular metric depth model constrained by the sparse points | denser cloud | the model can invent geometry; GraspGen-X needs real surfaces | compare with a scanned box | P3 |
| **G4 6D pose foundation models** | FoundationPose or similar on a CAD or scanned model | full 6D pose | per-product 3D models, GPU | not justified for flat cartons | P3 |
| **G5 Segmented cloud contract** | Keep the `.npz` hand-off; add the mask-based `object_points` when D2 exists; wait for Leon on frame, units and pose format | the interface is already defined (`docs/GRASP_OUTPUT.md`) | needs Leon's confirmation | one run of GraspGen-X on our file | P1 (Leon) |

### 3.5 Latency and runtime (answers: 1.5 to 7.4 s)

| Option | Idea | Gain | Risk | Pri |
|---|---|---|---|---|
| **L1 INT8 quantization (NNCF) of detector and recognizer** | OpenVINO post-training quantization | lower latency on the i3 iGPU/CPU | accuracy drop must be measured per field | P1 |
| **L2 Read scheduling** | Read the rarely-read fields first for the requested product (REF line before the whole spine), cancel stale reads, reuse the last box crops | most of the live variance (first read) | more rules | P1 |
| **L3 Process layout** | Camera thread, detector, OCR workers and the view as separate processes with a frame-drop policy | smoother UI, less contention | complexity | P2 |
| **L4 Better laptop or the server for OCR** | OCR on a stronger machine over the LAN | latency | network and safety dependency | P3 |

### 3.6 System, data and safety architecture

| Option | Idea | Why | Pri |
|---|---|---|---|
| **S1 Evidence ledger** | Append-only record per box: field, value, source, confidence, frame id, time, model hash; the decision is a function of the ledger | traceability and replayable decisions; review pages are views of it | P1 |
| **S2 Sealed test protocol** | A test day, a shelf layout and a lighting condition never used for training, tuning or labelling; negatives generated systematically | an honest error bound; stops the replay-vs-training overlap | P1 |
| **S3 Data and model registry** | Dataset and model versions with hashes (DVC or plain manifests), fold definitions by session | reproducible numbers | P2 |
| **S4 Active learning loop** | Pick the least-certain reads for owner labelling | spends owner time where it helps | P2 |
| **S5 Process interface** | One message schema from vision to TM and to the arm (request, decision, pose, evidence id); a ROS 2 node wrapper if the team standardises on ROS 2 | stops per-pair formats | P2 |
| **S6 Defence in depth** | Independent channels: shelf camera decision, close-up confirmation, scanner at the basket before release, a watchdog, fail-safe default (blue or stop) | one failed channel never causes a wrong pick | P1 (principle) |
| **S7 Standards review** | Check the lifecycle and risk files against the standards that may apply to hospital robots and software (risk management, software lifecycle, functional safety of robot control) | the hospital partner will ask | P3 |

## 4. Recommended staging

**Phase 1 (2 to 3 weeks), no new hardware, no new model:**
F3 shelf-level assignment, F5 safety envelope, R5 field-first reading, R2 constrained decoding of REF and dates (needs a way to get
character posteriors; otherwise a lexicon re-score of the top-k strings), L1 quantization test, L2 read scheduling, S1 ledger,
S2 sealed protocol. Test G1 with a borrowed Gemini 305 if available.

**Phase 2 (about a month):** R1 close-up stage with the arm team, R3 recognizer fine-tune (after owner-reviewed text labels),
R4 stacking A/B, F1+F2 calibrated sequential fusion, D2/D3 masks and G2 model-based pose, S3 to S5.

**Phase 3 (only if needed):** D4 transformer detectors, F4 conformal wrapper, G3/G4 depth completion and pose models, R6 VLM auditor, S7.

**Acceptance criteria (measured with the project's own confusion-matrix protocol, never replay-only):**
- False green must not rise against baseline A on the same sealed set; target 0 in at least 1,000 independent negative requests
  (95 % upper bound about 0.3 %).
- Sized and dated boxes: green in at least 95 % of requests on the sealed set for each camera used.
- Time to green: report the median and the 90th percentile over at least 5 batches; claim "within 2 s" only for the percentile that holds.
- No decision may rest on labels, slots or crops that were produced by models or Claude and not reviewed by the owner.

## 4a. Comparison of the whole architectures

Six candidates. **A** is what runs today (measured). **A+** is A with only the Phase 1 changes. **B** is the target of section 2.
**C**, **D** and **E** are equivalent or alternative designs shown for comparison. Ratings are my judgement from this project's
numbers and the literature, not measurements, except where the cell says "measured".

| | **A** current rules pipeline | **A+** A with Phase 1 only | **B** evidence-ledger + close-up | **C** learned embedding identification | **D** VLM-centred reading and decision | **E** server-offloaded reading |
|---|---|---|---|---|---|---|
| Idea | Detector, OCR, hand-set weights, two-source rule | A plus shelf-level assignment, constrained REF/date decoding, field-first reading, quantization, ledger | A+ plus close-up second stage, calibrated sequential fusion, masks, model-based pose | Match each spine to a gallery of catalogue images with DINOv2-type embeddings; use it as another source | A vision-language model reads the crop and proposes the identity | Same as A or A+, OCR and fusion on a stronger machine over the LAN |
| Evidence status | **Measured** (replay 11/11 webcam, live webcam 33/35 sized green over 5 batches, D455 4/7) | Hypothesis, each part cheap to test | Hypothesis, needs the arm team and new data | Hypothesis; look-alike siblings differ only in size digits | Hypothesis; conflicts with "no LLM in the decision" | Hypothesis |
| Reads REF on the D455 | No (0 of 90 crops) | No (optics) | **Yes, if the close-up reaches enough pixels** (untested) | Not a REF source | Unknown | No (optics) |
| Webcam REF reliability | Intermittent (K16 3 of 5) | Better (constrained decoding, stacking), size unmeasured | Best (close-up plus constraints) | n/a | Unknown | Same as A+ |
| False-green control | Hard rule, 0 of 88 negatives, bound about 3.4 % | Same rule plus the assignment constraint (should only lower it) | Calibrated P(wrong) per decision, still gated by the hard rule | Adds a source that is not independent of appearance cues | Hard to bound; model outputs are not calibrated to our shelf | Same as the base it offloads |
| Time to green | 1.5 to 7.4 s live (measured), 9 of 35 within 2 s | Lower and less variable (fewer OCR calls, quantization), size unmeasured | Adaptive (stopping rule) but adds approach time for the close-up | Fast (one embedding) but only for coarse identity | Slow on an i3, needs a server | Depends on the network; adds a failure mode |
| Extra hardware | None | None | Wrist camera or arm motion (Gemini 305 untested) | None | Server GPU | Server or stronger laptop on the LAN |
| New data / labels | None | Few (reviewed text of REF and date crops) | Masks, reviewed text labels, calibration set | Gallery photos per product | Prompt sets, audit data | None |
| Explainability for the doctor and audit | High (every atom is shown) | High, plus the ledger | High, plus probabilities and the ledger | Medium (a similarity score) | Low (free text) | High |
| Effort | 0 (exists) | About 2 to 3 weeks | About 1 to 2 months with the arm team | 2 to 3 weeks for a first version | 1 to 2 months and a policy change | 1 to 2 weeks |
| Main risk | Optics, intermittent REF, wide time variance | Gains smaller than hoped | Arm motion and hand-eye calibration; schedule | Siblings look identical; false sense of a third source | Hallucinated labels in the decision | Safety dependency on the network |
| Verdict | Baseline | **Do first** | **Target** | Optional low-priority source for brand-level checks only | Not for the decision; offline auditor only | Fallback if the laptop is too slow after A+ |

How to use this table: build A+ first because it is cheap and measurable against A; decide on B only after A+ shows how much of the
REF and timing problem is software and how much is optics; keep C, D and E out of the decision path unless a test shows a gain that
holds on a sealed set with the false-green target of section 4.

## 5. What this document does not claim

- None of the gains above are measured yet; they are hypotheses with a test attached.
- Numbers quoted in section 1 are from this project's recordings and live batches; replays overlap the training data and the
  owner review of identities is still pending for most batches.
- Hardware choices (Gemini 305, a wrist camera) depend on the arm team; focus type and depth quality at 20 to 40 cm are untested.

## 6. References (titles checked against the arXiv pages on 27 Oct 2026)

- PARSeq: Bautista and Atienza, *Scene Text Recognition with Permuted Autoregressive Sequence Models*, arXiv:2207.06966.
- PP-OCRv3: Li et al., *PP-OCRv3: More Attempts for the Improvement of Ultra Lightweight OCR System*, arXiv:2206.03001.
- RT-DETR: Zhao et al., *DETRs Beat YOLOs on Real-time Object Detection*, arXiv:2304.08069.
- D-FINE: Peng et al., *D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement*, arXiv:2410.13842.
- SAM 2: Ravi et al., *SAM 2: Segment Anything in Images and Videos*, arXiv:2408.00714.
- DINOv2: Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, arXiv:2304.07193.
- Depth Anything V2: Yang et al., arXiv:2406.09414.
- FoundationPose: Wen et al., *FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects*, arXiv:2312.08344.
- Conformal prediction: Angelopoulos and Bates, *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty
  Quantification*, arXiv:2107.07511.
- Florence-2: Xiao et al., *Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks*, arXiv:2311.06242.
- Wald, A. (1945). Sequential tests of statistical hypotheses. *Annals of Mathematical Statistics*, 16(2), 117-186 (SPRT).
- OpenVINO NNCF (post-training quantization): https://docs.openvino.ai/ (search "NNCF post-training quantization").
- NVlabs GraspGen: https://github.com/NVlabs/GraspGen (the project's earlier read of GraspGen-X: arXiv:2606.00998).

"""
Multi-tier shelf search.

Extends the design doc's Stage 1 idea (section 7: "wide camera locates the
region, then hand off to Stage 2 for precise detection") to a shelf that has
several physical tiers/levels — exactly the scenario you described: one rak
with 4 tingkat, holding a mix of pill bottles, tablets, and boxed medicines
of different shapes, and the task is "which tingkat has paracetamol?" before
the arm ever tries to reach for anything.

Deliberately narrow scope: this module answers ONE question — which tier
(if any) contains the target SKU — and nothing about grasping. Picking is a
separate task per your note, handled later once the arm is positioned at the
returned tier and Stage 2 (wrist camera, full-confidence verification) takes
over.

Accuracy note (also per your message): detection/verification run on a wide,
possibly distant shot of the whole shelf is expected to be less reliable than
a close-up — smaller pixels-on-target, more motion blur, OCR more likely to
fail. That's fine here: the bar for THIS stage is "good enough to pick a
tier", not "certain identification" — certainty is Stage 2's job. So this
scanner accepts color+shape matches (not just strict OCR matches) by default;
tune `shelf.accept_color_shape_matches` in the config if that's too loose for
your actual packaging.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.utils import crop_box


@dataclass
class TierCandidate:
    tier_index: int
    box: tuple
    detector_score: float
    sku: str | None
    name: str | None
    match_kind: str  # "ocr_match" | "color_shape_match" | "no_match"


@dataclass
class ShelfScanResult:
    target_sku: str
    found: bool
    tier_index: int | None
    best: TierCandidate | None
    all_candidates: list = field(default_factory=list)

    def tier_summary(self) -> dict[int, list[TierCandidate]]:
        """Every candidate grouped by tier, for logging/debugging/UI — lets you
        see e.g. 'tier 0: nothing recognized, tier 1: MED-002, tier 2: MED-001 (target)'."""
        out: dict[int, list[TierCandidate]] = {}
        for c in self.all_candidates:
            out.setdefault(c.tier_index, []).append(c)
        return out


def compute_tier_bounds(frame_height: int, cfg: dict) -> list[tuple[float, float]]:
    n = int(cfg.get("tiers", 4))
    explicit = cfg.get("tier_bounds")
    if explicit:
        return [(r0 * frame_height, r1 * frame_height) for r0, r1 in explicit]
    step = frame_height / n
    return [(i * step, (i + 1) * step) for i in range(n)]


def assign_tier(box: tuple, tier_bounds: list[tuple[float, float]]) -> int | None:
    _, y1, _, y2 = box
    cy = (y1 + y2) / 2.0
    for idx, (t0, t1) in enumerate(tier_bounds):
        if t0 <= cy < t1 or (idx == len(tier_bounds) - 1 and cy >= t0):
            return idx
    return None


class ShelfScanner:
    def __init__(self, detector, verifier, cfg: dict):
        """detector: YoloOnnxDetector. verifier: ObjectVerifier (already wired to the DB)."""
        self.detector = detector
        self.verifier = verifier
        self.cfg = cfg

    def _acceptable(self, match_kind: str) -> bool:
        if match_kind == "ocr_match":
            return True
        if match_kind == "color_shape_match":
            return bool(self.cfg.get("accept_color_shape_matches", True))
        return False

    def scan_frame(self, frame: np.ndarray) -> list[TierCandidate]:
        """Run detection + verification once over a full shelf frame, tag every
        candidate with its tier. Use scan_frames() below to sweep several shots
        (e.g. a pan across a long shelf) and merge results."""
        tier_bounds = compute_tier_bounds(frame.shape[0], self.cfg)
        detections, _ = self.detector.infer(frame)
        candidates = []
        for det in detections:
            tier_idx = assign_tier(det.box, tier_bounds)
            if tier_idx is None:
                continue
            crop = crop_box(frame, det.box)
            vr = self.verifier.verify(crop)
            candidates.append(
                TierCandidate(
                    tier_index=tier_idx,
                    box=det.box,
                    detector_score=det.score,
                    sku=vr.sku,
                    name=vr.name,
                    match_kind=vr.confidence,
                )
            )
        return candidates

    def scan_frames_for_ground_truth_boxes(
        self, frame: np.ndarray, boxes: list[tuple], tier_bounds_override=None
    ) -> list[TierCandidate]:
        """Same as scan_frame, but verifies a caller-supplied list of boxes instead
        of running the detector — used by tests/demos where ground-truth box
        locations are known (e.g. synthetic test assets), so the test is not at
        the mercy of a not-yet-trained detector's recall on synthetic images."""
        tier_bounds = tier_bounds_override or compute_tier_bounds(frame.shape[0], self.cfg)
        candidates = []
        for box in boxes:
            tier_idx = assign_tier(box, tier_bounds)
            if tier_idx is None:
                continue
            crop = crop_box(frame, box)
            vr = self.verifier.verify(crop)
            candidates.append(
                TierCandidate(
                    tier_index=tier_idx,
                    box=box,
                    detector_score=1.0,
                    sku=vr.sku,
                    name=vr.name,
                    match_kind=vr.confidence,
                )
            )
        return candidates

    def find_target(self, all_candidates: list[TierCandidate], target_sku: str) -> ShelfScanResult:
        hits = [c for c in all_candidates if c.sku == target_sku and self._acceptable(c.match_kind)]
        if not hits:
            return ShelfScanResult(target_sku, False, None, None, all_candidates)

        def rank(c: TierCandidate):
            kind_rank = 1 if c.match_kind == "ocr_match" else 0
            return (kind_rank, c.detector_score)

        best = max(hits, key=rank)
        return ShelfScanResult(target_sku, True, best.tier_index, best, all_candidates)

    def scan(self, frame: np.ndarray, target_sku: str) -> ShelfScanResult:
        candidates = self.scan_frame(frame)
        return self.find_target(candidates, target_sku)

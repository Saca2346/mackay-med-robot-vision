"""Fast evidence acquisition for visual tracking; never a physical PICK policy."""
from __future__ import annotations

import time

import numpy as np

from src.box_evidence import gather, gather_full_box, redecide
from src.box_inventory import SIG_PARTS, similarity
from src.box_verifier import CANDIDATE, CONFIRM, Evidence, combined_score


def focused_order(groups, ids, tracker, lock, need=2, required_expiry=None):
    """Read a small group of similar designs, not one guessed sibling at a time.

    Two consistent exclusions are needed before dropping a box. If colour gives
    no shortlist (or it is exhausted), return None for the existing full search.
    Appearance affects scheduling only; exact size still comes from evidence.
    """
    from src.box_verifier import EXPIRED, IGNORED
    visible = [tid for tid in ids if tid >= 0 and tid in tracker.tracks]
    best = max((tracker.state(t).hint for t in visible), default=0)
    candidates = []
    supported = []
    def date_priority(state):
        if required_expiry is None:
            return 0
        dates = [set(ds) for ds in list(state.dates)[-need:]]
        readings = list(state.evs)[-need:]
        if len(readings) == len(dates):
            # A decoded GS1 date is already an observation of this track. Without
            # it, a supported wrong-date sibling can monopolize the shortlist
            # when barcode success means no printed-date OCR was needed.
            # This changes read order only; no dates/sources are added to state.
            for ds, ev in zip(dates, readings):
                ds.update(c.expiry for c in getattr(ev, "codes", ())
                          if c.source == "barcode" and c.expiry is not None)
        if len(dates) < need or any(not ds for ds in dates):
            return 1  # not enough observations to prefer a dated sibling
        if all(required_expiry in ds for ds in dates):
            return 0
        common = set(dates[0]).intersection(*dates[1:])
        return 2 if common and all(required_expiry not in ds for ds in dates) else 1
    for tid in visible:
        state = tracker.state(tid)
        support = combined_score(state.support[-1][0])[0] if state.support else 0
        observed = (support >= 1 and not state.support[-1][1] and state.history
                    and state.history[-1][0] not in (IGNORED, EXPIRED))
        date_rank = date_priority(state)
        # A reading that supports the request outranks the colour guess: on the D455 (22-10, K10 / K21) the target's
        # hint was below the shortlist after its first matching reading and every other box was read first. Not a
        # sibling whose own printed date already differs twice (Sled Bag K03: the 03-24 / 03-12 bags took every read).
        if state.hint < max(.68, best - .10) and not (observed and date_rank < 2):
            continue
        recent = list(state.history)[-need:]
        if len(recent) >= need and all(d in (IGNORED, EXPIRED) for d, _ in recent):
            continue
        if state.reads >= 6 and not any(combined_score(atoms)[0] >= 1 for atoms, _ in state.support):
            continue
        rank = (date_rank, -support, state.reads, -state.hint, tid)
        candidates.append(rank)
        if observed and date_rank < 2:
            supported.append(rank)
    if supported:
        # Finish the observed target before spending more OCR on other siblings.
        # This is scheduling only, not a new exclusion or verification rule.
        return [min(supported)[-1]]
    return [r[-1] for r in sorted(candidates)[:2]] if candidates else None


def reuse_request(tracker, request, cfg, catalog):
    """Re-evaluate observed evidence on continuous tracks; never reuse a stale identity."""
    for track in tracker.tracks.values():
        state = track["state"]
        if state.stale or track.get("missed", 0) or track.get("was_out"):
            state.mark_stale()
            continue
        state.redecide(lambda ev: redecide(ev, request, cfg, catalog))
        state.next_ocr_frame = 0


def colour_similarity(a, b):
    """Lighting-tolerant reading priority, preserving the order of colour bands.

    Ignore brightness and saturation bins and soften one hue bin in either
    direction. Requiring the printed identity still distinguishes same-design
    sizes; this score must never count as identity support.
    """
    if a is None or b is None or a.shape != b.shape or a.size != SIG_PARTS * 18:
        return similarity(a, b)

    def bands(s):
        h = s.reshape(SIG_PARTS, 18)[:, :12]
        h = h + .5 * (np.roll(h, 1, axis=1) + np.roll(h, -1, axis=1))
        return h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-8)

    aa, bb = bands(a), bands(b)
    relaxed = max(float(np.mean(np.sum(aa * v, axis=1))) for v in (bb, bb[::-1]))
    return max(similarity(a, b), relaxed)


def read_visual_evidence(sl_crops, box_crop, too_far, reader, cfg, req, catalog=None):
    """Stop after complete visual support, retaining conflicts and dated requests.

    Barcode decoding and searches solely for an additional independent source
    are omitted here. The standard verifier/scoring is still used, with no
    reduced PICK threshold. Callers must never attach SelectionSession.
    """
    started = time.perf_counter()
    if too_far:
        ev, count = Evidence(too_far=True), 0
    else:
        ev, count = gather(sl_crops, box_crop, reader, cfg, req, visual=True)
        decision, _, _, sup = redecide(ev, req, cfg, catalog)
        score, _ = combined_score(sup[0])
        missing_date = req.expiry is not None and req.expiry not in ev.dates
        near = box_crop is not None and min(box_crop.shape[:2]) >= cfg.get("verify", {}).get("full_box_min_width_px", 40)
        if near and decision in (CANDIDATE, CONFIRM) and (score < 1 or sup[1] or missing_date):
            count += gather_full_box(ev, box_crop, reader, cfg, product=req.product)[0]
    decision, reason, key, ev.support = redecide(ev, req, cfg, catalog)
    return ev, decision, reason, key, (time.perf_counter() - started) * 1000, count

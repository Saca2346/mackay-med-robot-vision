"""
Target lock (meeting 15-10: "turn off the reading of the other boxes, keep one green box, 1-2 s").

The inventory knows what the requested box looks like (spine colour signature, src/box_inventory.signature) and where
it stood (slot order, left to right). The lock picks the ONE box on screen that looks most like it, and only that box
is read (OCR text, REF, barcode); the other boxes are still detected and tracked (cheap: one network pass for all) but
not read. Reading is what costs time (0.5-1 s per box on the laptop CPU), so one box instead of 20 is the speed-up.

Safety is unchanged: PICK still needs two independent sources on the locked box (src/box_verifier). The lock only
chooses the reading order. A locked box whose reading says "not the target" is dropped and the next most similar box
is locked; with no similar box left (moved, replaced, not in the inventory, unreadable from here) the full search
reads every box as before, so nothing is missed that the full search would find.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.box_geometry import to_poly
from src.box_verifier import EXPIRED, IGNORED, MATCH, combined_score

LOCK_MIN = 0.80          # spine looks this much like the requested inventory box (same scale as the reading hint)
POS_MIN = 0.50           # the box in the target's mapped slot must still look at least this much like it
PLACE_MATCH = 1.5        # shelf map: a mapped box and a detected box this many box widths apart may be the same box
PLACE_TOL = 0.5          # ... the requested box's place (after the camera shift) within half a box width
PLACE_DY = 0.10          # ... and within this share of the frame height (another shelf level is further)
TIE = 0.03               # similarities this close count as equal: the slot position decides (identical designs)
MAX_READS = 6            # readings of one locked box without a decision -> stop locking, read every box
WAIT_FRAMES = 15         # frames without any similar box -> full search


@dataclass
class TargetLock:
    tid: int | None = None             # box number (tracker) that is locked now
    box_id: str = ""                   # inventory box it looks like
    sim: float = 0.0
    since: float | None = None         # when the current lock was set
    first: float | None = None         # when the first lock was set (time to lock)
    rejected: set = field(default_factory=set)
    fallback: str = ""                 # why the full search took over ("" = locking)
    reads0: int = 0                    # TrackState.reads of the locked box when it was locked
    waited: int = 0                    # frames with hints but no similar box
    n_shelf: int = 0                   # boxes on the shelf when it was mapped (slots are counted among all of them)
    min_similarity: float = LOCK_MIN   # reading priority only; never identity evidence
    one_source: bool = False           # locked box keeps matching on one source only: needs a closer look
    code_doubt: bool = False           # locked box: size text matches, a code (REF / GTIN) disagrees
    log: list = field(default_factory=list)   # (time, text) lock changes, printed by the caller

    @property
    def active(self) -> bool:
        return not self.fallback

    def _say(self, now, text):
        self.log.append((now, text))

    def _stop(self, now, why):
        self.fallback, self.tid = why, None
        self.one_source = self.code_doubt = False
        self._say(now, f"full search: {why}")

    def update(self, now: float, groups, ids, tracker, need: int, order_of: dict, target_ids=(), places=None,
               frame_size=None) -> None:
        """Once per frame, after the reading hints (TrackState.hint / hint_id) are up to date. order_of = shelf map
        {inventory box: slot, 1 = leftmost} of the verified boxes (scripts/rak_inventaris.py peta); places = shelf map
        {inventory box: (x, y, width) as fractions of the frame}; target_ids = the inventory boxes that fit the
        request, FEFO order; frame_size = (width, height) px."""
        if self.fallback:
            return
        if self.tid is not None:
            if self.tid not in tracker.tracks:
                self._say(now, f"#{self.tid} lost from view - choosing again")
                self.tid = None
            else:
                st = tracker.state(self.tid)
                last = st.history[-1][0] if st.history else None
                if st.stable(need) == MATCH:
                    self.one_source = self.code_doubt = False
                    return
                atoms = frozenset().union(*(a for a, _ in st.support))
                score, _ = combined_score(atoms)
                recent = [a for a, _ in list(st.support)[-need:]]
                if last != EXPIRED and st.stable(need) == IGNORED and recent \
                        and all({"tp", "td", "tl"} <= a for a in recent):
                    # the big SIZE TEXT says it is the requested box and only a small code (REF / GTIN) disagrees:
                    # a misread code digit (D455 16-10: Nagomi 2.75 x 33 REF read ...2711 twice -> the right box
                    # was dropped, PICK 43 s instead of 24 s). Keep reading this box; PICK still needs a window of
                    # readings without any contradiction.
                    if not self.code_doubt:
                        self._say(now, f"#{self.tid}: size text matches, a code disagrees - reading it again")
                        self.code_doubt = True
                    return
                if st.stable(need) in (IGNORED, EXPIRED) or (last == EXPIRED):
                    # "not the target" in `need` consistent readings moves the lock. One reading is not enough: at
                    # shelf distance one REF digit is misread now and then (D455, 16-10: 4 of 180 spine readings,
                    # e.g. Nagomi 3.0 x 24 read DE-R51024 = 1.0 x 24), which must not throw away the right box.
                    self._say(now, f"#{self.tid} is not the target ({st.last_reason[:40]}) - next similar box")
                    self.rejected.add(self.tid)
                    self.tid = None
                elif st.reads - self.reads0 >= MAX_READS and score == 0:
                    # A product name / partial size can match every sibling. Only a complete matching source
                    # justifies staying here; otherwise let the full search read the other boxes.
                    self._stop(now, f"#{self.tid} gave no complete matching source in {MAX_READS} readings")
                    return
                else:
                    # matching but still one source (e.g. size text read, REF / barcode too small from here): the
                    # lock stays - reading other boxes does not help; the camera has to come closer (meeting 15-10:
                    # lock -> move closer -> barcode)
                    self.one_source = st.reads - self.reads0 >= MAX_READS and score == 1
                    return
        xs = sorted((to_poly(g.box.poly).mean(axis=0)[0], t) for g, t in zip(groups, ids) if t >= 0)
        pos = {t: k + 1 for k, (_, t) in enumerate(xs)}
        wide = {t for g, t in zip(groups, ids) if t >= 0 and _width(g.box.poly) >= 16}   # look measurable (hints)
        if any(tracker.state(t).hint_frame < 0 for t in wide if t not in self.rejected):
            return                                     # look of every box not measured yet (first frames)
        # 0 place: where each mapped box stood in the camera frame (shelf map). The camera may have shifted a little
        #   and the detector may miss boxes (16-10: 17-18 of 20 with another model): the shift is the median offset of
        #   the mapped boxes to their nearest detected box, then the box nearest the requested box's mapped place is
        #   locked if it still looks like a requested box.
        if places and frame_size:
            hit = self._by_place(groups, ids, tracker, places, target_ids, frame_size)
            if hit is not None:
                return self._lock(now, hit[0], hit[1], tracker.state(hit[0]).hint, tracker, "its place in the shelf map")
        # 1 position: the whole shelf is in view (as many boxes as when it was mapped), so the k-th box from the left
        #   stands in slot k. The box in the slot of the requested box (FEFO first) is locked if it still looks like
        #   a requested box.
        n_shelf = self.n_shelf or max(order_of.values(), default=0)
        if order_of and len(pos) == n_shelf:
            for bid in target_ids:
                slot = order_of.get(bid)
                t = next((t for t, k in pos.items() if k == slot), None)
                if t is None or t in self.rejected or tracker.state(t).hint < POS_MIN:
                    continue
                return self._lock(now, t, bid, tracker.state(t).hint, tracker, f"slot {slot} of the shelf map")
        # 2 look: the most similar box on screen (twins: the one nearest its mapped slot)
        cands = []
        pending_hint = False
        for t in pos:
            if t in self.rejected:
                continue
            st = tracker.state(t)
            if st.hint >= self.min_similarity:
                slot = order_of.get(st.hint_id)
                off = abs(pos[t] - slot) if slot else 99
                cands.append((st.hint, off, t, st.hint_id))
        if not cands:
            if self.rejected and not pending_hint:
                self._stop(now, "every similar box was checked")
            elif not pending_hint:
                self.waited += 1
                if self.waited >= WAIT_FRAMES:
                    self._stop(now, "no box on screen looks like the requested one")
            return
        best = max(c[0] for c in cands)
        near = [c for c in cands if c[0] >= best - TIE]
        sim, off, tid, bid = min(near, key=lambda c: (c[1], -c[0]))
        self._lock(now, tid, bid, sim, tracker, "look" + (f", {off} slot(s) from its mapped slot" if off < 99 else ""))

    def _by_place(self, groups, ids, tracker, places, target_ids, frame_size):
        W, H = frame_size
        det = [(t, to_poly(g.box.poly).mean(axis=0) / (W, H)) for g, t in zip(groups, ids) if t >= 0]
        if len(det) < 3:
            return None
        tids = [t for t, _ in det]
        for k, bid in place_candidates(np.array([p for _, p in det], np.float32), places, target_ids):
            t = tids[k]
            if t not in self.rejected and tracker.state(t).hint >= POS_MIN:
                return t, bid
        return None

    def _lock(self, now, tid, bid, sim, tracker, why):
        self.tid, self.box_id, self.sim, self.since, self.one_source = tid, bid, float(sim), now, False
        self.code_doubt = False
        self.reads0 = tracker.state(tid).reads
        if self.first is None:
            self.first = now
        self._say(now, f"#{tid} locked as {bid} by {why} (look {sim:.2f})")


def place_candidates(pts, places: dict, target_ids):
    """Shelf map -> the detected box standing at each requested box's mapped place, FEFO order: yields (index into
    pts, inventory box). pts = detected spine centres (N, 2) as fractions of the frame; places = {box: (x, y, width)}
    fractions. The camera shift is the median offset of the mapped boxes to their nearest detected box."""
    if len(pts) < 3:
        return
    offs = []
    for x, y, w in places.values():
        d = pts - (x, y)
        k = int(np.argmin(np.hypot(d[:, 0], d[:, 1])))
        if abs(d[k, 0]) < PLACE_MATCH * w and abs(d[k, 1]) < PLACE_DY:
            offs.append(d[k])
    if len(offs) < 3:
        return                                            # camera moved too far from the mapped view
    sx, sy = np.median(np.array(offs), axis=0)
    for bid in target_ids:
        p = places.get(bid)
        if p is None:
            continue
        x, y, w = p
        d = pts - (x + sx, y + sy)
        ok = np.abs(d[:, 1]) < PLACE_DY
        if not ok.any():
            continue
        k = int(np.argmin(np.where(ok, np.abs(d[:, 0]), np.inf)))
        if abs(d[k, 0]) < PLACE_TOL * w:
            yield k, bid


def _width(poly) -> float:
    p = to_poly(poly)
    return float(min(np.linalg.norm(p[1] - p[0]), np.linalg.norm(p[2] - p[1])))


def lock_order(lock: TargetLock | None, pick) -> list | None:
    """Box numbers to read in this frame while locking (None = no lock: normal reading order)."""
    if lock is None or not lock.active:
        return None
    out = [t for t in (pick, lock.tid) if t is not None]
    return list(dict.fromkeys(out))


def as_slot_map(boxes, active=("terverifikasi",)) -> dict:
    """Verified inventory boxes -> {box_id: slot (1 = leftmost)} for the boxes with a slot from the shelf map."""
    out = {}
    for b in boxes:
        try:
            slot = int(float(b.order))
        except (TypeError, ValueError):
            continue
        if slot > 0 and b.status in active:
            out[b.box_id] = slot
    return out


def as_place_map(boxes, active=("terverifikasi",)) -> dict:
    """Verified inventory boxes -> {box_id: (x, y, width)} (fractions of the frame) from the shelf map."""
    out = {}
    for b in boxes:
        try:
            v = [float(s) for s in (getattr(b, "pos", "") or "").split()]
        except ValueError:
            continue
        if len(v) == 3 and b.status in active:
            out[b.box_id] = tuple(v)
    return out


def lock_xy(groups, ids, tid):
    """Centre (px) of the locked box in this frame, for the screen; None if not visible."""
    for g, t in zip(groups, ids):
        if t == tid:
            return np.asarray(to_poly(g.box.poly).mean(axis=0))
    return None

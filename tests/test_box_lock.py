"""pytest tests/test_box_lock.py -q  (target lock: read only the box that looks like / stands where the target is)"""
from types import SimpleNamespace

from src.box_lock import MAX_READS, WAIT_FRAMES, TargetLock, as_slot_map, lock_order
from src.box_verifier import CANDIDATE, IGNORED, MATCH, READING


class St:
    def __init__(self, hint=0.0, hint_id="", measured=True):
        self.hint, self.hint_id, self.hint_frame = hint, hint_id, (5 if measured else -1000)
        self.history, self.support, self.reads, self.last_reason = [], [], 0, ""
        self._stable = READING

    def stable(self, need):
        return self._stable

    def read(self, decision, atoms=frozenset(), reason=""):
        self.history.append((decision, None))
        self.support.append((atoms, False))
        self.reads += 1
        self.last_reason = reason
        if decision in (IGNORED, MATCH):
            self._stable = decision
        elif decision == CANDIDATE:
            self._stable = CANDIDATE


class Tracker:
    def __init__(self, states):
        self.tracks = {t: {"state": s} for t, s in states.items()}

    def state(self, t):
        return self.tracks[t]["state"]


def shelf(n, hints=None, ids=None):
    """n boxes side by side (x = 50 * k), box numbers 101.. in a shuffled order, hints {rank: (sim, box_id)}."""
    ids = ids or [100 + ((7 * k) % n) + 1 for k in range(n)]
    groups = [SimpleNamespace(box=SimpleNamespace(poly=[[50 * k, 0], [50 * k + 40, 0], [50 * k + 40, 400],
                                                        [50 * k, 400]])) for k in range(n)]
    states = {t: St(*(hints or {}).get(k + 1, (0.3, "K99"))) for k, t in enumerate(ids)}
    return groups, ids, Tracker(states)


def test_whole_shelf_in_view_locks_the_box_in_the_mapped_slot():
    groups, ids, tr = shelf(20, {13: (0.98, "K07"), 4: (0.97, "K07")})
    lock = TargetLock(n_shelf=20)
    lock.update(0.1, groups, ids, tr, 2, {"K07": 13, "K04": 7}, ["K07"])
    assert lock.tid == ids[12] and lock.box_id == "K07" and "slot 13" in lock.log[-1][1]
    assert lock_order(lock, None) == [ids[12]]


def test_part_of_the_shelf_in_view_locks_by_look():
    groups, ids, tr = shelf(12, {5: (0.91, "K07"), 9: (0.85, "K07")})
    lock = TargetLock(n_shelf=20)
    lock.update(0.1, groups, ids, tr, 2, {"K07": 13}, ["K07"])
    assert lock.tid == ids[4] and "look" in lock.log[-1][1]


def test_no_lock_until_every_box_was_looked_at():
    groups, ids, tr = shelf(5, {2: (0.95, "K07")})
    tr.state(ids[3]).hint_frame = -1000
    lock = TargetLock()
    lock.update(0.1, groups, ids, tr, 2, {}, ["K07"])
    assert lock.tid is None and lock.active


def test_a_reading_saying_not_the_target_moves_the_lock():
    groups, ids, tr = shelf(6, {2: (0.95, "K07"), 5: (0.90, "K07")})
    lock = TargetLock()
    lock.update(0.1, groups, ids, tr, 2, {}, ["K07"])
    first = lock.tid
    st = tr.state(first)
    st.history.append((IGNORED, None))          # ONE reading "other size": a misread REF digit -> lock stays
    st.support.append((frozenset(), True))
    st.reads += 1
    lock.update(0.4, groups, ids, tr, 2, {}, ["K07"])
    assert lock.tid == first and not lock.rejected
    st.read(IGNORED, reason="other size 2.75 x 20")  # consistent: stable IGNORED -> next similar box
    lock.update(0.5, groups, ids, tr, 2, {}, ["K07"])
    assert lock.tid == ids[4] and first in lock.rejected
    tr.state(ids[4]).read(IGNORED)
    lock.update(0.9, groups, ids, tr, 2, {}, ["K07"])
    assert not lock.active and "every similar box" in lock.fallback


def test_nothing_looks_like_the_target_falls_back_to_reading_every_box():
    groups, ids, tr = shelf(6)
    lock = TargetLock()
    for k in range(WAIT_FRAMES):
        lock.update(0.1 * k, groups, ids, tr, 2, {}, ["K07"])
    assert not lock.active and lock_order(lock, None) is None


def test_one_source_keeps_the_lock_and_asks_for_a_closer_look():
    groups, ids, tr = shelf(6, {3: (0.95, "K07")})
    lock = TargetLock()
    lock.update(0.1, groups, ids, tr, 2, {}, ["K07"])
    st = tr.state(lock.tid)
    for _ in range(MAX_READS):
        st.read(CANDIDATE, frozenset({"tp", "td", "tl"}))
        lock.update(1.0, groups, ids, tr, 2, {}, ["K07"])
    assert lock.active and lock.tid == ids[2] and lock.one_source


def test_nothing_matching_read_on_the_lock_falls_back():
    groups, ids, tr = shelf(6, {3: (0.95, "K07")})
    lock = TargetLock()
    lock.update(0.1, groups, ids, tr, 2, {}, ["K07"])
    st = tr.state(lock.tid)
    for _ in range(MAX_READS):
        st.read(READING)
    lock.update(2.0, groups, ids, tr, 2, {}, ["K07"])
    assert not lock.active


def test_product_name_only_does_not_hold_the_lock_on_a_sibling_forever():
    # Siblings share the product name: that alone does not establish the requested size.
    groups, ids, tr = shelf(6, {3: (0.95, "K07"), 5: (0.90, "K07")})
    lock = TargetLock()
    lock.update(0.1, groups, ids, tr, 2, {}, ["K07"])
    st = tr.state(lock.tid)
    for _ in range(MAX_READS):
        st.read(CANDIDATE, frozenset({"tp"}))
    lock.update(2.0, groups, ids, tr, 2, {}, ["K07"])
    assert not lock.active and lock_order(lock, None) is None
    assert not lock.one_source


def test_complete_text_split_across_readings_still_holds_one_source_lock():
    groups, ids, tr = shelf(6, {3: (0.95, "K07")})
    lock = TargetLock()
    lock.update(0.1, groups, ids, tr, 2, {}, ["K07"])
    st = tr.state(lock.tid)
    for k in range(MAX_READS):
        st.read(CANDIDATE, frozenset({"tp", "td"} if k % 2 else {"tl"}))
    lock.update(2.0, groups, ids, tr, 2, {}, ["K07"])
    assert lock.active and lock.tid == ids[2] and lock.one_source


def test_candidate_with_two_sources_is_not_reported_as_one_source():
    # Identity may have two sources while the requested printed expiry still needs reading.
    groups, ids, tr = shelf(6, {3: (0.95, "K07")})
    lock = TargetLock()
    lock.update(0.1, groups, ids, tr, 2, {}, ["K07"])
    st = tr.state(lock.tid)
    for _ in range(MAX_READS):
        st.read(CANDIDATE, frozenset({"tp", "td", "tl", "ref"}))
    lock.update(2.0, groups, ids, tr, 2, {}, ["K07"])
    assert lock.active and lock.tid == ids[2] and not lock.one_source


def test_pick_box_is_read_first_while_locked():
    lock = TargetLock(tid=7)
    assert lock_order(lock, 3) == [3, 7] and lock_order(lock, 7) == [7]
    assert lock_order(None, 3) is None


def test_slot_map_uses_only_verified_boxes_with_a_slot():
    boxes = [SimpleNamespace(box_id="K01", order="3", status="terverifikasi"),
             SimpleNamespace(box_id="K02", order="0", status="terverifikasi"),
             SimpleNamespace(box_id="K03", order="5", status="draf"),
             SimpleNamespace(box_id="K04", order="x", status="terverifikasi")]
    assert as_slot_map(boxes) == {"K01": 3}


def test_mapped_place_survives_missed_detections_and_a_small_camera_shift():
    # map: 20 boxes 50 px apart in a 1000 px wide frame; now 3 boxes are not detected and the camera moved 12 px
    from src.box_lock import as_place_map
    places = {f"K{k:02d}": ((50 * k + 20) / 1000, 200 / 800, 40 / 1000) for k in range(20)}
    keep = [k for k in range(20) if k not in (2, 9, 15)]
    groups = [SimpleNamespace(box=SimpleNamespace(poly=[[50 * k + 12, 0], [50 * k + 52, 0], [50 * k + 52, 400],
                                                        [50 * k + 12, 400]])) for k in keep]
    ids = [300 + k for k in keep]
    tr = Tracker({t: St(0.95, "K13") for t in ids})
    lock = TargetLock(n_shelf=20)
    lock.update(0.1, groups, ids, tr, 2, {}, ["K13"], places, (1000, 800))
    assert lock.tid == 313 and "place" in lock.log[-1][1]
    # the requested box itself was not detected -> never a place lock on its neighbour (only the look rule is left)
    lock2 = TargetLock(n_shelf=20)
    lock2.update(0.1, groups, ids, tr, 2, {}, ["K09"], places, (1000, 800))
    assert "place" not in lock2.log[-1][1] and "look" in lock2.log[-1][1]
    boxes = [SimpleNamespace(box_id="K01", pos="0.5 0.25 0.04", status="terverifikasi"),
             SimpleNamespace(box_id="K02", pos="", status="terverifikasi"),
             SimpleNamespace(box_id="K03", pos="0.1 0.2 0.04", status="draf")]
    assert as_place_map(boxes) == {"K01": (0.5, 0.25, 0.04)}


def test_size_text_matching_keeps_the_lock_when_only_a_code_disagrees():
    # D455 16-10: Nagomi 2.75 x 33, REF misread ...2711 in two readings while the size text read 2.75 / 33
    groups, ids, tr = shelf(6, {3: (0.97, "K13")})
    lock = TargetLock()
    lock.update(0.1, groups, ids, tr, 2, {}, ["K13"])
    st = tr.state(lock.tid)
    for _ in range(2):
        st.read(IGNORED, frozenset({"tp", "td", "tl"}), reason="REF: other size")
    lock.update(1.0, groups, ids, tr, 2, {}, ["K13"])
    assert lock.tid == ids[2] and not lock.rejected and "code disagrees" in lock.log[-1][1]
    # the size TEXT itself says another size -> really not the target
    st.read(IGNORED, frozenset({"tp"}), reason="text: length 24")
    st.read(IGNORED, frozenset({"tp"}), reason="text: length 24")
    lock.update(2.0, groups, ids, tr, 2, {}, ["K13"])
    assert ids[2] in lock.rejected

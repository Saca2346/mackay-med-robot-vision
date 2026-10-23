"""Fast visual reading preserves exact-size distinctions and verifier policy."""
from types import SimpleNamespace as NS
import datetime as dt

import cv2
import numpy as np
import pytest

from src.box_evidence import gather
from src.box_inventory import signature
from src.box_ocr import OcrResult, OcrToken
from src.box_verifier import CANDIDATE, CONFIRM, IGNORED, Evidence, Request, TrackState, combined_score
from src.box_visual_read import colour_similarity, focused_order, read_visual_evidence, reuse_request
from src.config import load_config, resolve


@pytest.fixture
def cfg():
    return load_config(resolve("config/box_pipeline_config_webcam.yaml"))


class Reader:
    def __init__(self, words):
        self.words, self.calls = words, 0

    def read_orient(self, image, rotation):
        self.calls += 1
        return OcrResult([OcrToken(t, .99) for t in self.words])


def test_visual_ref_stops_redundant_orientations_without_barcode_or_extra_source(cfg, monkeypatch):
    import src.box_evidence as evidence
    def forbidden(*args, **kwargs):
        pytest.fail("Visual mode attempted barcode decode or a second-source-only scan")
    monkeypatch.setattr(evidence, "decode_gs1", forbidden)
    monkeypatch.setattr("src.box_visual_read.gather_full_box", forbidden)
    reader = Reader(["SCCDSR14150250029"])
    crop = np.zeros((60, 600, 3), np.uint8)
    ev, decision, _, _, _, count = read_visual_evidence([crop], crop, False, reader, cfg, Request("angiolite", 2.5, 29))
    assert decision == CANDIDATE and combined_score(ev.support[0])[0] == 1
    assert reader.calls == count == 1


@pytest.mark.parametrize("diameter,length", [(4, 19), (4, 24), (2.5, 19)])
def test_same_design_other_size_never_supports_requested_target(cfg, diameter, length):
    reader = Reader(["angiolite", "SCCDSR14150250029"])
    crop = np.zeros((60, 600, 3), np.uint8)
    ev, decision, *_ = read_visual_evidence([crop], crop, False, reader, cfg, Request("angiolite", diameter, length))
    assert decision == IGNORED
    assert ev.support[1]
    assert combined_score(ev.support[0])[0] < 1


def test_conflicting_text_and_ref_are_not_hidden_by_fast_exit(cfg, monkeypatch):
    monkeypatch.setattr("src.box_visual_read.gather_full_box", lambda *a, **kw: (0, ""))
    reader = Reader(["angiolite", "4 mm", "19 mm", "SCCDSR14150250029"])
    crop = np.zeros((60, 600, 3), np.uint8)
    ev, decision, *_ = read_visual_evidence([crop], crop, False, reader, cfg, Request("angiolite", 2.5, 29))
    assert decision in (IGNORED, CONFIRM) and ev.support[1]


def test_standard_gather_still_decodes_barcode(cfg, monkeypatch):
    import src.box_evidence as evidence
    calls = []
    monkeypatch.setattr(evidence, "decode_gs1", lambda crop, **kw: calls.append(kw) or [])
    crop = np.zeros((60, 600, 3), np.uint8)
    gather([crop], crop, Reader(["angiolite", "2.5 mm", "29 mm"]), cfg, Request("angiolite", 2.5, 29))
    assert calls == [{"hard": True}]


def test_dated_visual_request_still_searches_for_missing_date(cfg, monkeypatch):
    calls = []
    monkeypatch.setattr("src.box_visual_read.gather_full_box", lambda *a, **kw: calls.append(True) or (1, ""))
    crop = np.zeros((60, 600, 3), np.uint8)
    ev, *_ = read_visual_evidence([crop], crop, False, Reader(["SCCDSR14150250029"]), cfg,
                                  Request("angiolite", 2.5, 29, dt.date(2028, 7, 31)))
    assert calls == [True] and not ev.dates


def test_colour_bands_handle_lighting_but_cannot_supply_identity():
    crop = np.zeros((40, 240, 3), np.uint8)
    crop[:] = (180, 175, 80)
    crop[:, :40] = crop[:, -40:] = (130, 30, 180)
    a = signature(crop)
    darker = cv2.convertScaleAbs(crop, alpha=.55)
    assert colour_similarity(a, signature(darker)) > .95
    assert colour_similarity(a, signature(darker[:, ::-1])) > .95
    assert colour_similarity(a, signature(np.full_like(crop, 160))) < .68


def states():
    records = {i: TrackState(hint=h) for i, h in [(1, .76), (2, .74), (3, .4)]}
    return records, NS(tracks={i: {"state": s} for i, s in records.items()}, state=records.__getitem__)


def test_sibling_shortlist_reads_both_and_does_not_drop_after_one_disagreement():
    ss, tr = states()
    assert set(focused_order([], [1, 2, 3], tr, None)) == {1, 2}
    ss[1].add(IGNORED, ("other",), (frozenset(), True))
    assert 1 in focused_order([], [1, 2, 3], tr, None)
    ss[1].add(IGNORED, ("other",), (frozenset(), True))
    assert focused_order([], [1, 2, 3], tr, None) == [2]
    assert focused_order([], [-1], tr, None) is None


def test_exhausted_shortlist_returns_to_full_search():
    ss, tr = states()
    for state in ss.values():
        for _ in range(2):
            state.add(IGNORED, ("other",), (frozenset(), True))
    assert focused_order([], [1, 2, 3], tr, None) is None


def test_observed_target_gets_next_read_before_unresolved_sibling():
    ss, tr = states()
    ss[2].add(CANDIDATE, ("target",), (frozenset({"ref"}), False))
    assert focused_order([], [1, 2, 3], tr, None) == [2]
    # One-source support remains one source; scheduling never upgrades it.
    assert not ss[2].combined_match(2)


def test_observed_support_outranks_a_colour_hint_that_left_the_box_out():
    # D455 22-10 (K10 / K21): the target's colour hint was below the shortlist after its first matching reading, so
    # the next reads went to every other box. The reading is evidence; the hint only an appearance guess.
    ss, tr = states()
    ss[3].add(CANDIDATE, ("target",), (frozenset({"tp", "td", "tl"}), False))
    assert focused_order([], [1, 2, 3], tr, None) == [3]
    assert not ss[3].combined_match(2)                    # still one source: read order only
    ss[3].add(IGNORED, ("other",), (frozenset(), True))   # evidence against it: back to the shortlist
    assert set(focused_order([], [1, 2, 3], tr, None)) == {1, 2}


def test_low_hint_siblings_with_other_dates_do_not_block_the_full_search():
    # D455 22-10, K03 (2029-03-10): the 03-24 and 03-12 bags (name read, low colour hint) took every read.
    ss, tr = states()
    wanted, other = dt.date(2029, 3, 10), dt.date(2029, 3, 24)
    for tid in (1, 2):                                    # high-hint shortlist exhausted
        for _ in range(2):
            ss[tid].add(IGNORED, ("other",), (frozenset(), True))
    bag = (frozenset({"tp", "td", "tl"}), False)
    ss[3].add(CANDIDATE, ("bag",), bag, Evidence(dates=(other,)))
    assert focused_order([], [1, 2, 3], tr, None, required_expiry=wanted) == [3]     # one other date: read it again
    ss[3].add(CANDIDATE, ("bag",), bag, Evidence(dates=(other,)))
    assert focused_order([], [1, 2, 3], tr, None, required_expiry=wanted) is None    # twice: back to full search


def test_two_consistent_other_dates_allow_reading_an_unresolved_identical_sibling():
    ss, tr = states()
    wanted, other = dt.date(2029, 3, 24), dt.date(2029, 3, 10)
    ev = Evidence(dates=(other,))
    ss[1].add(CANDIDATE, ("bag",), (frozenset({"tp", "td", "tl"}), False), ev)
    # A single different date is insufficient to deprioritize the current box.
    assert focused_order([], [1, 2], tr, None, required_expiry=wanted) == [1]
    ss[1].add(CANDIDATE, ("bag",), (frozenset({"tp", "td", "tl"}), False), ev)
    assert focused_order([], [1, 2], tr, None, required_expiry=wanted)[0] == 2
    for _ in range(2):
        ss[2].add(CANDIDATE, ("bag",), (frozenset({"tp", "td", "tl"}), False), Evidence(dates=(wanted,)))
    assert focused_order([], [1, 2], tr, None, required_expiry=wanted) == [2]
    # A printed date is scheduling evidence, not a second identity source.
    assert not ss[2].combined_match(2)


def test_request_change_redecides_real_evidence_and_discards_stale_tracks(cfg):
    from src.box_ref import parse_refs
    from src.box_target_view import source_ready
    ss, tr = states()
    for state in ss.values():
        for _ in range(2):
            state.add(IGNORED, ("other",), (frozenset(), True),
                      Evidence(refs=parse_refs("SCCDSR14150400019")))
    tr.tracks[2]["missed"] = 1
    reuse_request(tr, Request("angiolite", 4, 19), cfg, None)
    assert source_ready(ss[1])
    assert not source_ready(ss[2])
    reuse_request(tr, Request("angiolite", 2.5, 29), cfg, None)
    assert not source_ready(ss[1])


def test_repeated_decoded_other_date_does_not_starve_unread_sibling():
    from src.box_barcode import parse_gs1
    from src.box_verifier import MATCH
    ss, tr = states()
    wanted = dt.date(2029, 3, 24)
    code = parse_gs1('(01)08714729896845(17)290310(10)39525412')
    ev = Evidence(codes=[code])
    for count in range(2):
        ss[1].add(MATCH, ('bag',), (frozenset({('bc', code.gtin)}), False), ev)
        order = focused_order([], [1, 2], tr, None, required_expiry=wanted)
        assert order[0] == (1 if count == 0 else 2)
    # Ranking never supplies the missing identity or date to the unread track.
    assert not ss[2].combined_match(2) and not ss[2].evs


@pytest.mark.parametrize('source,expiry', [('ocr', dt.date(2029, 3, 10)), ('barcode', None)])
def test_unobserved_or_ocr_gtin_date_does_not_exclude_sibling(source, expiry):
    from src.box_barcode import Gs1Code
    from src.box_verifier import MATCH
    ss, tr = states()
    code = Gs1Code('08714729896845', expiry=expiry, source=source)
    for _ in range(2):
        ss[1].add(MATCH, ('bag',), (frozenset({('bc', code.gtin)}), False), Evidence(codes=[code]))
    assert focused_order([], [1, 2], tr, None, required_expiry=dt.date(2029,3,24)) == [1]

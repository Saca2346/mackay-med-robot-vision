"""Green focused verification requires independent evidence and preserves its provenance."""
import datetime as dt
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from src.box_target_evidence import observed_fields, read_target_evidence, save_evidence
from src.box_target_view import draw_target, verified_ready
from src.box_verifier import CANDIDATE, EXPIRED, Evidence, TrackState, Request
from src.box_ref import parse_refs
from src.box_ocr import OcrToken, OcrResult
from src.config import load_config, resolve


def state_with(atoms):
    state = TrackState()
    for _ in range(2):
        state.add(CANDIDATE, ("target",), (frozenset(atoms), False))
    return state


def test_repeated_ref_alone_never_turns_green():
    assert not verified_ready(state_with({"ref"}))
    assert verified_ready(state_with({"tp", "td", "tl", "ref"}))


@pytest.mark.parametrize("change", ["stale", "conflict", "expired", "date", "gtin"])
def test_green_gate_rejects_uncertain_sources(change, monkeypatch):
    import src.box_verifier as verifier
    monkeypatch.setattr(verifier, "REQUIRED_EXPIRY", None)
    st = state_with({"tp", "td", "tl", "ref"})
    date = None
    if change == "stale":
        st.mark_stale()
    elif change == "conflict":
        st.support[-1] = (st.support[-1][0], True)
    elif change == "expired":
        st.history[-1] = (EXPIRED, ("target",))
    elif change == "date":
        date = dt.date(2029, 3, 24)
        st.expiry = dt.date(2029, 3, 12)
    else:
        st.support[0] = (frozenset({("bc", "111")}), False)
        st.support[1] = (frozenset({("bc", "222")}), False)
    assert not verified_ready(st, required_expiry=date)


def test_observed_fields_do_not_copy_requested_size_inventory_or_stale_lot():
    st = TrackState()
    st.gtin, st.lot, st.expiry = "stale_gtin", "stale_lot", dt.date(2029, 3, 24)
    ev = Evidence(refs=parse_refs("SCCDSR14150250029"))
    st.add(CANDIDATE, ("target",), (frozenset({"ref"}), False), ev)
    fields = observed_fields(st)
    assert fields["diameter_mm"]["value"] == "2.5"
    assert fields["diameter_mm"]["observations"] == [{"value": "2.5", "source": "ref_parse"}]
    assert all(fields[k]["value"] is None for k in ("GTIN", "LOT", "expiry", "brand"))


def test_report_saves_reader_inputs_and_does_not_call_located_letters_a_barcode(tmp_path, monkeypatch):
    import src.box_crops as crops
    st = state_with({"ref"})
    ev = Evidence(refs=parse_refs("SCCDSR14150250029"))
    ev.capture = {"spine": np.full((40, 240, 3), 117, np.uint8),
                  "labels": [np.full((120, 30, 3), 77, np.uint8)]}
    st.evs[-1] = ev
    monkeypatch.setattr(crops, "locate_barcode", lambda image: (np.float32([[0, 10], [30, 10], [30, 50], [0, 50]]), .8))
    pnl, path = save_evidence(tmp_path, 1, Request("angiolite", 4, 19), 7, st, False, None, "video")
    report = json.loads((path / "result.json").read_text())
    assert report["fields"]["diameter_mm"]["value"] == "2.5"  # observed REF, not requested 4
    assert not report["identity_verified"] and not report["all_fields_read"]
    assert not report["barcode_decoded"] and not report["retrieval_authorized"]
    assert report["evidence"][0]["barcode_crop"] == "not_confirmed"
    assert (path / "reading_2/size_label_1.png").exists()
    assert (path / "reading_2/spine_ocr_input.png").exists()
    assert tuple(pnl[0, 0]) == (255, 220, 30)


def test_fast_orientation_keeps_two_sources_and_checks_other_size(monkeypatch):
    import src.box_barcode as barcode
    cfg = load_config(resolve("config/box_pipeline_config_focus.yaml"))
    monkeypatch.setattr(barcode, "decode_gs1", lambda *a, **k: [])
    words = ["angiolite", "2.5 mm", "29 mm", "SCCDSR14150250029"]
    reader = NS(read_once=lambda image: OcrResult([OcrToken(w, .99) for w in words]))
    crop = np.zeros((300, 50, 3), np.uint8)
    ev, decision, *_ = read_target_evidence([crop], crop, False, reader, cfg, Request("angiolite", 2.5, 29))
    assert decision == "match" and {"tp", "td", "tl", "ref"} <= ev.support[0]
    assert ev.capture["labels"][0] is crop
    ev, decision, *_ = read_target_evidence([crop], crop, False, reader, cfg, Request("angiolite", 4, 19))
    assert decision != "match" and ev.support[1]


def test_fast_reader_periodically_searches_full_spine_and_barcode(monkeypatch):
    import src.box_target_evidence as module
    calls = []
    ev = Evidence()
    monkeypatch.setattr(module, "read_evidence", lambda *a: calls.append(True) or (ev, CANDIDATE, "", None, 0, 0))
    cfg = load_config(resolve("config/box_pipeline_config_focus.yaml"))
    read_target_evidence([], None, False, None, cfg, Request("angiolite", 2.5, 29), attempt=3)
    assert calls == [True]


def test_verified_outline_is_green_and_pending_outline_is_blue():
    frame = np.zeros((120, 180, 3), np.uint8)
    poly = np.float32([[60, 5], [85, 5], [85, 115], [60, 115]])
    for flag, color in ((False, (255, 220, 30)), (True, (0, 210, 0))):
        result = draw_target(frame, poly, "test", "status", verified=flag, verifying=True)
        assert tuple(result[100, 60]) == color


def test_background_export_freezes_evidence_and_returns_original_context(monkeypatch):
    import threading
    import src.box_target_evidence as module
    entered, release = threading.Event(), threading.Event()
    state = NS(values=[1], image=np.full((4, 4), 7))
    def save(snapshot):
        entered.set()
        assert release.wait(5)
        return snapshot
    monkeypatch.setattr(module, "save_evidence", save)
    writer = module.EvidenceWriter()
    try:
        assert writer.submit(("old request", 7, 0), state)
        assert entered.wait(5)
        state.values.append(2)
        state.image[:] = 99
        assert writer.drain() is None
        assert not writer.submit(("new request", 9, 0), state)
        release.set()
        context, snapshot = writer.drain(wait=True)
        assert context == ("old request", 7, 0)
        assert snapshot.values == [1] and (snapshot.image == 7).all()
    finally:
        release.set()
        writer.close()


def test_background_export_close_surfaces_write_failures(monkeypatch):
    import src.box_target_evidence as module
    def fail(*args):
        raise OSError("disk full")
    monkeypatch.setattr(module, "save_evidence", fail)
    writer = module.EvidenceWriter()
    writer.submit("request", object())
    with pytest.raises(OSError, match="disk full"):
        writer.close()


def test_verified_detail_reads_disable_partial_exit_without_mutating_shared_config(monkeypatch):
    import src.box_spine_read as spine
    cfg = load_config(resolve("config/box_pipeline_config_focus.yaml"))
    cfg.setdefault("visual", {})["partial_spine_read"] = True
    modes = []
    def read(box, reader, settings, req, catalog):
        modes.append(settings["visual"]["partial_spine_read"])
        return (Evidence(), CANDIDATE, "", None, 0, 0), []
    monkeypatch.setattr(spine, "read_spine_bands", read)
    for verified in (False, True):
        read_target_evidence([], np.zeros((50, 700, 3), np.uint8), False, None, cfg,
                             Request("angiolite", 4, 19), identity_verified=verified)
    assert modes == [True, False]
    assert cfg["visual"]["partial_spine_read"] is True


@pytest.mark.parametrize("text,has_code,expected", [
    ("Permanent Sled Bag 2029-03-24", True, "match"),
    ("Permanent Sled Bag 2029-03-24", False, "candidate"),
    ("Accuforce", True, "needs_confirmation"),
])
def test_no_size_spine_reads_real_name_and_code_without_repeated_orientations(monkeypatch, text, has_code, expected):
    import src.box_target_evidence as module
    import src.box_barcode as barcode
    from src.box_catalog import Catalog
    cfg = load_config(resolve("config/box_pipeline_config_all_targets.yaml"))
    catalog = Catalog(resolve(cfg["verify"]["catalog_csv"]))
    code = barcode.parse_gs1("(01)08714729896845(17)290324(10)39500350")
    monkeypatch.setattr(barcode, "decode_gs1", lambda *a, **k: [code] if has_code else [])
    monkeypatch.setattr(module, "read_evidence", lambda *a: pytest.fail("unnecessary repeated OCR"))
    seen = []
    def read(image):
        seen.append(image)
        return OcrResult([OcrToken(text, .99)])
    box = np.zeros((60, 900, 3), np.uint8)
    ev, decision, *_, count = read_target_evidence([], box, False, NS(read_once=read), cfg,
                                                 Request("permanent sled bag"), catalog)
    assert decision == expected
    assert len(seen) == count == 1 and seen[0] is box
    assert ev.capture["bands"][0]["image"] is box
    assert ev.spine == text
    if expected == "match":
        assert {"tp", "td", "tl", ("bc", code.gtin)} <= ev.support[0]


@pytest.mark.parametrize("options", [{"attempt": 3}, {"identity_verified": True}])
def test_no_size_spine_retains_periodic_full_reader_and_detail_search(monkeypatch, options):
    import src.box_target_evidence as module
    calls = []
    monkeypatch.setattr(module, "read_evidence", lambda *a: calls.append(True) or
                        (Evidence(), CANDIDATE, "", None, 0, 0))
    cfg = load_config(resolve("config/box_pipeline_config_all_targets.yaml"))
    read_target_evidence([], np.zeros((50, 500, 3), np.uint8), False, None, cfg,
                         Request("permanent sled bag"), **options)
    assert calls == [True]


def test_no_size_spine_low_confidence_date_is_not_an_observed_expiry(monkeypatch):
    import src.box_barcode as barcode
    monkeypatch.setattr(barcode, "decode_gs1", lambda *a, **k: [])
    cfg = load_config(resolve("config/box_pipeline_config_all_targets.yaml"))
    reader = NS(read_once=lambda image: OcrResult([OcrToken("Permanent Sled Bag", .99),
                                                  OcrToken("2029-03-24", .5)]))
    ev, *_ = read_target_evidence([], np.zeros((50, 500, 3), np.uint8), False, reader, cfg,
                                 Request("permanent sled bag", expiry=dt.date(2029, 3, 24)))
    assert not ev.dates
    assert not ev.codes


@pytest.mark.parametrize("product,attempt", [("permanent sled bag", 0), ("permanent sled bag", 3),
                                           ("angiolite", 0), ("angiolite", 3)])
def test_disabled_barcode_never_calls_decoder_in_fast_or_full_readers(monkeypatch, product, attempt):
    import src.box_barcode as barcode
    import src.box_evidence as evidence
    def fail(*a, **k):
        pytest.fail("barcode/QR decoder called while disabled")
    monkeypatch.setattr(barcode, "decode_gs1", fail)
    monkeypatch.setattr(evidence, "decode_gs1", fail)
    cfg = load_config(resolve("config/box_pipeline_config_all_targets.yaml"))
    cfg["verify"]["decode_barcodes"] = False
    words = ["Permanent Sled Bag", "2029-03-24"] if product == "permanent sled bag" else [
        "angiolite", "4 mm", "19 mm", "SCCDSR14150400019"]
    reader = NS(read_once=lambda image: OcrResult([OcrToken(w, .99) for w in words]),
                read_orient=lambda image, rotation: OcrResult([OcrToken(w, .99) for w in words]))
    req = Request(product) if product == "permanent sled bag" else Request(product, 4, 19)
    box = np.zeros((50, 700, 3), np.uint8)
    ev, decision, reason, key, *_ = read_target_evidence([box], box, False, reader, cfg, req, attempt=attempt)
    assert not ev.codes and ev.capture["decode_barcodes"] is False
    st = TrackState()
    for _ in range(2):
        st.add(decision, key, ev.support, ev)
    assert verified_ready(st) is (product == "angiolite")  # printed size + independent REF


def test_no_barcode_export_skips_locator_and_retains_observed_ocr(tmp_path, monkeypatch):
    import src.box_crops as crops
    from src.box_verifier import Identity
    def fail(*a, **k):
        pytest.fail("barcode locator called while disabled")
    monkeypatch.setattr(crops, "pick_crops", fail)
    ev = Evidence(text=Identity(product="permanent sled bag", product_ok=True))
    ev.capture = {"spine": np.zeros((50, 300, 3), np.uint8), "labels": [], "decode_barcodes": False}
    st = state_with({"tp", "td", "tl"})
    st.evs[-1] = ev
    _, path = save_evidence(tmp_path, 1, Request("permanent sled bag"), 1, st, False, None, "video")
    report = json.loads((path / "result.json").read_text())
    assert not report["identity_verified"] and not report["barcode_decoded"]
    assert report["fields"]["product"]["value"] == "permanent sled bag"
    assert report["evidence"][0]["barcode_crop"] == "disabled"
    assert not list(path.rglob("barcode*.png"))


def test_disabled_barcode_date_retry_uses_actual_orientation_but_is_one_observation(monkeypatch):
    import src.box_barcode as barcode
    monkeypatch.setattr(barcode, "decode_gs1", lambda *a, **k: pytest.fail("disabled decoder"))
    cfg = load_config(resolve("config/box_pipeline_config_all_targets.yaml"))
    cfg["verify"]["decode_barcodes"] = False
    inputs = []
    def read(image):
        inputs.append(image.copy())
        return OcrResult([OcrToken("Permanent Sled Bag", .99),
                          OcrToken("2029-03-24", .5 if len(inputs) == 1 else .99)])
    image = np.arange(50*700*3, dtype=np.uint8).reshape(50, 700, 3)
    ev, decision, _, key, _, count = read_target_evidence([], image, False, NS(read_once=read), cfg,
                              Request("permanent sled bag", expiry=dt.date(2029, 3, 24)))
    assert count == 2 and len(inputs) == 2
    assert np.array_equal(inputs[1], image[::-1, ::-1])
    assert ev.dates == (dt.date(2029, 3, 24),)
    assert len(ev.capture["bands"]) == 2 and not ev.codes
    state = TrackState()
    state.add(decision, key, ev.support, ev)
    assert state.reads == 1 and not verified_ready(state)


def test_date_retry_crops_observed_pixels_and_never_fills_the_requested_date():
    from src.box_target_evidence import date_region_retry
    image = np.arange(50*700*3, dtype=np.uint8).reshape(50, 700, 3)
    token = OcrToken('202--24', .7, np.float32([[164,1],[244,1],[244,23],[164,23]]))
    seen = []
    def read(patch):
        seen.append(patch.copy())
        return OcrResult([OcrToken('2029-03-24', .999)])
    bands = date_region_retry(image, [token], NS(read_once=read))
    assert len(bands) == 1
    assert seen[0].shape[1] < image.shape[1] / 4
    assert np.array_equal(bands[0]['image'], image[:33,154:254][::-1,::-1])
    assert bands[0]['tokens'][0].text == '2029-03-24'
    assert date_region_retry(image, [OcrToken('REF12345', .8, token.poly)], NS(read_once=read)) == []

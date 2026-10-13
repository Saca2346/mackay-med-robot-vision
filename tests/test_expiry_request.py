"""pytest tests/test_expiry_request.py -q  (request for ONE dated box among identical ones, e.g. the 4 Sled Bags)"""
import datetime as dt
import importlib.util

import pytest

import src.box_verifier as bv
from src.box_demo import HABIS, SIAP, lookup
from src.box_inventory import STATUS_DIAMBIL, STATUS_OK, InvBox, Inventory
from src.box_verifier import CANDIDATE, MATCH, TrackState, parse_request, pick_fefo

TODAY = dt.date(2026, 10, 14)
SLED = [("K01", "2029-04-26"), ("K03", "2029-03-10"), ("K05", "2029-03-24"), ("K17", "2029-03-12")]


def inv(tmp_path, taken=()):
    i = Inventory(tmp_path / "inventaris.csv")
    i.boxes = {k: InvBox(k, "permanent sled bag", expiry=e, status=STATUS_DIAMBIL if k in taken else STATUS_OK)
               for k, e in SLED}
    return i


@pytest.fixture(autouse=True)
def _reset_flag():
    yield
    bv.REQUIRED_EXPIRY = None


def test_date_in_the_request_is_read_and_not_taken_as_a_size():
    r = parse_request("boston scientific permanent sled bag exp 2029-03-24")
    assert (r.product, r.diameter, r.length, r.expiry) == ("permanent sled bag", None, None, dt.date(2029, 3, 24))
    r = parse_request("angiolite 2.5 x 29 2028-07-31")
    assert (r.diameter, r.length, r.expiry) == (2.5, 29.0, dt.date(2028, 7, 31))
    assert parse_request("permanent sled bag").expiry is None


def test_impossible_date_is_refused_not_guessed():
    with pytest.raises(ValueError, match="valid date"):
        parse_request("permanent sled bag 2029-02-30")


def test_dated_request_picks_that_box_not_the_fefo_one(tmp_path):
    assert lookup("permanent sled bag", inv(tmp_path), {}, TODAY).box.box_id == "K03"          # FEFO
    res = lookup("permanent sled bag 2029-03-24", inv(tmp_path), {}, TODAY)
    assert res.box.box_id == "K05" and res.stock == ["K05"]


def test_date_not_in_stock_and_taken_dated_box_are_reported(tmp_path):
    res = lookup("permanent sled bag 2029-05-01", inv(tmp_path), {}, TODAY)
    assert res.status == HABIS and "2029-05-01" in res.reason
    res = lookup("permanent sled bag 2029-03-24", inv(tmp_path, taken={"K05"}), {}, TODAY)
    assert res.status == HABIS and res.reason == "already taken: K05"
    assert lookup("permanent sled bag 2029-03-12", inv(tmp_path, taken={"K05"}), {}, TODAY).status != HABIS


def _matched(expiry=None):
    st = TrackState()
    for _ in range(2):
        st.add(MATCH, ("permanent sled bag",))
    st.expiry = expiry
    return st


def test_without_a_date_nothing_changes():
    assert _matched().stable(2) == MATCH


def test_dated_request_never_matches_without_that_printed_date():
    bv.REQUIRED_EXPIRY = dt.date(2029, 3, 24)
    assert _matched(None).stable(2) == CANDIDATE                       # date not read (shelf distance)
    assert _matched(dt.date(2029, 3, 10)).stable(2) == CANDIDATE        # another Sled Bag
    assert _matched(dt.date(2029, 3, 24)).stable(2) == MATCH


def test_pick_goes_to_the_box_with_the_requested_date_even_if_another_is_earlier():
    bv.REQUIRED_EXPIRY = dt.date(2029, 3, 24)
    states = {1: _matched(dt.date(2029, 3, 10)), 2: _matched(dt.date(2029, 3, 24)), 3: _matched(None)}
    assert pick_fefo(states, 2) == 2


def test_camera_request_keeps_the_date():
    spec = importlib.util.spec_from_file_location("demo_request", "scripts/demo_request.py")
    dr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dr)
    assert dr.camera_request(parse_request("permanent sled bag 2029-03-24")) == "permanent sled bag 2029-03-24"
    assert dr.camera_request(parse_request("angiolite 2.5 x 29")) == "angiolite,2.5,29"
    assert parse_request(dr.camera_request(parse_request("angiolite 2.5 x 29 2028-07-31"))).expiry == dt.date(2028, 7, 31)

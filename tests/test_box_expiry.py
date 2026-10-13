"""pytest tests/test_box_expiry.py -q  (printed expiry dates on the spine, src/box_expiry.py)"""
import datetime as dt

import pytest

from src.box_expiry import find_dates, trusted_expiry
from src.box_verifier import TrackState

D = dt.date


@pytest.mark.parametrize("text, want", [
    ("Boston Scientific Permanent Sled Bag 2029-03-10", [D(2029, 3, 10)]),
    ("Permanent Sled Bag 2029 - 03 - 12 ##", [D(2029, 3, 12)]),
    ("2029.03.24", [D(2029, 3, 24)]),
    ("2O29-O4-26", [D(2029, 4, 26)]),            # O read instead of 0
    ("2029-03-1O 2029-03-10", [D(2029, 3, 10)]),  # same date twice -> once
    ("2029 03 24", [D(2029, 3, 24)]),             # dashes lost
    ("82029-09-21", [D(2029, 9, 21)]),            # hourglass icon read as a leading "8"
    ("Permanent Sled Bag 82029-03-24", [D(2029, 3, 24)]),
])
def test_find_dates_reads_printed_date(text, want):
    assert find_dates(text) == want


@pytest.mark.parametrize("text", [
    "", "Perimansat Rled Bag", "Accuforce 2512", "DC-RM2512HHW",
    "(01)08714729896845(17)290426(10)39718940",   # GS1 date: handled by src/box_barcode.py, not here
    "2029-13-10", "2029-02-30",                    # impossible month / day
    "1999-03-10", "2.5 x 29 mm",
])
def test_find_dates_ignores_non_dates(text):
    assert find_dates(text) == []


def test_track_keeps_dates_per_read_and_forgets_them_when_stale():
    class Ev:
        dates = (D(2029, 3, 10),)
    st = TrackState()
    st.add("candidate", ("k",), None, Ev())
    st.add("candidate", ("k",), None, None)
    assert list(st.dates) == [(D(2029, 3, 10),), ()]
    st.expiry, st.expiry_src = D(2029, 3, 10), "text"
    st.mark_stale()
    assert not st.dates and st.expiry is None


# Minimum safety contract for trusted_expiry: whatever rule is chosen, these must hold.
A, B = D(2029, 3, 10), D(2029, 3, 12)          # K03 and K17: one digit apart


def test_trusted_expiry_needs_more_than_one_read():
    assert trusted_expiry([(A,)], need=2) is None


def test_trusted_expiry_agreeing_reads():
    assert trusted_expiry([(A,), (), (A,)], need=2) == A


def test_trusted_expiry_conflicting_dates_are_not_trusted():
    assert trusted_expiry([(A,), (B,), (A,)], need=2) is None


def test_trusted_expiry_must_exist_in_inventory():
    assert trusted_expiry([(A,), (A,)], need=2, allowed={B}) is None
    assert trusted_expiry([(A,), (A,)], need=2, allowed={A, B}) == A

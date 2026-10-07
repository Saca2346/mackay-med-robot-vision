"""pytest tests/test_box_verifier.py -q"""
import pytest

from src.box_detector import Det, group_labels
from src.box_geometry import rotated_rect_poly
from src.box_ocr import OcrToken
from src.box_verifier import (CONFIRM, IGNORED, MATCH, READING, Request, SimpleTracker, TrackState, decide,
                              parse_identity, parse_request)

CATALOG = ["angiolite", "xperience pro", "essential pro", "accuforce", "ultimaster nagomi", "conqueror nc pro",
           "nc emerge", "agent", "sapphire", "scoreflex", "ryurei"]


def toks(*items, conf=0.99):
    return [OcrToken(t, conf) if isinstance(t, str) else OcrToken(*t) for t in items]


def test_parse_request():
    r = parse_request("Angiolite,2.5,29")
    assert (r.product, r.diameter, r.length) == ("angiolite", 2.5, 29)
    r = parse_request("xperience pro 2.5x15")
    assert (r.product, r.diameter, r.length) == ("xperience pro", 2.5, 15)
    with pytest.raises(ValueError):
        parse_request("2.5")


def test_accuforce_real_ocr_tokens():
    # token persis seperti keluaran RapidOCR pada crop size_label asli (Terumo Accuforce)
    ident = parse_identity(toks("20", "2.75", "<L>", "mm", "TERUMO", "Accuforce"), CATALOG)
    assert (ident.product, ident.diameter, ident.length) == ("accuforce", 2.75, 20)
    assert ident.product_ok and ident.diameter_ok and ident.length_ok


def test_angiolite_ignores_ref_lot_dates_and_comma_decimal():
    t = toks("iVascular", "angiolite", "SCCDSR14150250029", "STENT", "2,5", "mm", "STENT", "29", "mm",
             "LOT", "251631", "2028-07-31", "2025-08-01")
    ident = parse_identity(t, CATALOG)
    assert (ident.product, ident.diameter, ident.length) == ("angiolite", 2.5, 29)


def test_misread_ref_start_is_not_a_diameter():
    # replay uji_tripod 30-09: REF "SCCDSR14150400019" terbaca "5CCD5A14150400019" di sebelah "mm" -> dulu diameter 5
    ident = parse_identity(toks("angiolite", "4", "19", "2028-09-21", "5CCD5A14150400019", "mm"), CATALOG)
    assert ident.diameter is None and ident.diameter_hint == 4 and (ident.length, ident.length_ok) == (19, True)
    # angka ukuran yang menempel ke satuannya / pemisah tetap terbaca
    for words, want in ((("angiolite", "2.5x29"), (2.5, 29)), (("angiolite", "2,5mm", "29rnm"), (2.5, 29)),
                        (("accuforce", "2.75X20MM"), (2.75, 20))):
        ident = parse_identity(toks(*words), CATALOG)
        assert (ident.diameter, ident.length) == want, words


def test_ocr_typo_still_matches_catalog():
    ident = parse_identity(toks("anqiolite", "2.5", "19"), CATALOG)
    assert ident.product == "angiolite"


def test_integer_diameter_and_length():
    ident = parse_identity(toks("essential", "pro", "BCDPR14N150300040", "3", "mm", "40", "mm"), CATALOG)
    assert (ident.product, ident.diameter, ident.length) == ("essential pro", 3, 40)


def test_ambiguous_numbers_are_not_guessed():
    ident = parse_identity(toks("angiolite", "2.5", "19", "29"), CATALOG)
    assert ident.length is None and any("several lengths" in n for n in ident.notes)
    ident = parse_identity(toks("angiolite", "accuforce", "2.5", "19"), CATALOG)
    assert ident.product is None


def test_units_that_are_not_mm_are_ignored():
    ident = parse_identity(toks("accuforce", "2.75", "20", "mm", "RBP", "22 atm", "140 cm"), CATALOG)
    assert (ident.diameter, ident.length) == (2.75, 20)


def test_decisions():
    req = Request("angiolite", 2.5, 29)
    assert decide(parse_identity(toks("angiolite", "2.5", "29"), CATALOG), req)[0] == MATCH
    assert decide(parse_identity(toks("accuforce", "2.75", "20"), CATALOG), req)[0] == IGNORED
    assert decide(parse_identity(toks("angiolite", "2.5", "19"), CATALOG), req)[0] == IGNORED
    assert decide(parse_identity(toks("angiolite", "2.5"), CATALOG), req)[0] == CONFIRM
    low = toks("angiolite", "2.5", ("29", 0.55))
    assert decide(parse_identity(low, CATALOG), req)[0] == CONFIRM


def test_stable_needs_consistent_readings():
    st = TrackState()
    assert st.stable(2) == READING
    st.history.append((MATCH, ("angiolite", 2.5, 29)))
    assert st.stable(2) == READING
    st.history.append((MATCH, ("angiolite", 2.5, 29)))
    assert st.stable(2) == MATCH
    st.history.append((IGNORED, ("angiolite", 2.5, 19)))
    assert st.stable(2) == CONFIRM          # bacaan berubah -> ragu -> minta konfirmasi


def test_tracker_keeps_id_when_box_moves_slightly():
    tr = SimpleTracker()
    a = rotated_rect_poly(100, 100, 200, 40, 5)
    b = rotated_rect_poly(300, 100, 200, 40, 5)
    ids1 = tr.update([a, b])
    ids2 = tr.update([rotated_rect_poly(304, 102, 200, 40, 5), rotated_rect_poly(103, 101, 200, 40, 5)])
    assert ids2 == [ids1[1], ids1[0]]


def test_group_labels():
    big = Det("box", 0.9, rotated_rect_poly(200, 200, 400, 60, 0))
    small = Det("box", 0.9, rotated_rect_poly(300, 200, 100, 60, 0))      # di dalam "big"
    lab_in_small = Det("size_label", 0.8, rotated_rect_poly(300, 200, 30, 20, 0))
    lab_edge = Det("size_label", 0.8, rotated_rect_poly(200, 229, 30, 20, 0))  # pusat di dalam big, sebagian keluar
    lab_beside = Det("size_label", 0.8, rotated_rect_poly(200, 245, 30, 20, 0))  # di luar big (label kotak sebelah)
    lab_far = Det("size_label", 0.8, rotated_rect_poly(900, 900, 30, 20, 0))
    groups, orphans = group_labels([big, small, lab_in_small, lab_edge, lab_beside, lab_far])
    g_big, g_small = groups
    assert lab_in_small in g_small.labels and lab_edge in g_big.labels
    assert orphans == [lab_beside, lab_far]                                  # tidak dipasangkan ke kotak terdekat


def test_parse_request_in_plain_language():
    from src.box_verifier import describe_request, parse_request
    cases = {
        "pilih ivascular angiolite diameter 4 mm panjang 19": ("angiolite", 4.0, 19.0),
        "angiolite,2.5,29": ("angiolite", 2.5, 29.0),
        "angiolite 2.5x29": ("angiolite", 2.5, 29.0),
        "ambil terumo accuforce 2,75 x 20": ("accuforce", 2.75, 20.0),
        "nagomi 3/24": ("ultimaster nagomi", 3.0, 24.0),
        "Terumo Ultimaster Nagomi Ø2.75 L33": ("ultimaster nagomi", 2.75, 33.0),
        "cari angiolit panjang 29 diameter 2.5": ("angiolite", 2.5, 29.0),          # salah ketik + urutan terbalik
        "boston scientific permanent sled bag": ("permanent sled bag", None, None),
        "xperience 2.5 15": ("xperience pro", 2.5, 15.0),
    }
    for text, want in cases.items():
        r = parse_request(text)
        assert (r.product, r.diameter, r.length) == want, text
    assert describe_request(parse_request("pilih ivascular angiolite diameter 4 mm panjang 19")) == \
        "angiolite (ivascular), diameter 4 mm, panjang 19 mm"


def test_parse_request_rejects_unclear_requests():
    import pytest
    from src.box_verifier import parse_request
    for bad in ("pilih angiolite 2.5",                # panjang tidak ada
                "terumo angiolite 2.5 x 29",          # merek tidak cocok dengan produk
                "pilih kotak biru 3 x 20",            # produk tidak dikenal
                "angiolite 29 x 2.5"):                # tertukar: diameter 29 mm tidak masuk akal
        with pytest.raises(ValueError):
            parse_request(bad)

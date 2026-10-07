"""pytest tests/test_box_verify_multi.py -q

Verifikasi banyak sumber (barcode GS1, teks ukuran, kode REF). String GTIN / REF di sini dibaca dari foto
dataset (lihat data/gtin_catalog.csv dan src/box_ref.py)."""
import datetime as dt

import pytest

from src.box_barcode import find_gtins_in_text, gs1_date, gtin_valid, parse_gs1
from src.box_catalog import Catalog, CatalogEntry
from src.box_ocr import OcrToken
from src.box_ref import parse_refs
from src.box_verifier import (CANDIDATE, CONFIRM, EXPIRED, IGNORED, MATCH, READING, Evidence, Request, TrackState,
                              decide_multi, parse_identity, parse_request, pick_fefo)

TODAY = dt.date(2026, 9, 26)
NAMES = ["angiolite", "xperience pro", "essential pro", "accuforce", "ultimaster nagomi", "agent", "nc emerge",
         "permanent sled bag"]


def catalog(verified=True):
    c = Catalog()
    for g, p, d, l in (("04987350720245", "accuforce", 2.75, 20), ("04987350720061", "accuforce", 2.5, 12),
                       ("05413206252510", "ultimaster nagomi", 2.75, 33), ("08714729896845", "permanent sled bag", None, None)):
        c.upsert(CatalogEntry(gtin=g, product=p, diameter_mm=d, length_mm=l, verified=verified))
    return c


def code(text):
    return parse_gs1(text, "Code 128")


def text(*toks):
    return parse_identity([OcrToken(t, 0.99) for t in toks], NAMES)


# ------------------------------------------------------------------ GS1
def test_gtin_check_digit():
    for g in ("08714729896845", "04987350720245", "05413206252510", "04987350720061"):
        assert gtin_valid(g)
    assert not gtin_valid("08714729896846")          # satu digit salah selalu ketahuan
    assert not gtin_valid("0871472989684")           # 13 digit


def test_parse_gs1_fields_and_dates():
    c = parse_gs1("(01)05413206252428(17)280131(10)260206(93)002222")
    assert (c.gtin, c.expiry, c.lot) == ("05413206252428", dt.date(2028, 1, 31), "260206")
    assert gs1_date("290400") == dt.date(2029, 4, 30)                     # DD 00 = akhir bulan
    assert parse_gs1("(01)08714729896846(17)290426") is None             # check digit salah


def test_gtin_from_ocr_text():
    found = find_gtins_in_text("(01)O5413206252510(17)270630(10)250725")   # O dibaca sebagai 0
    assert [c.gtin for c in found] == ["05413206252510"] and found[0].expiry == dt.date(2027, 6, 30)
    assert found[0].source == "ocr"
    assert find_gtins_in_text("(01)05413206252511(17)270630") == []


# ------------------------------------------------------------------ REF
@pytest.mark.parametrize("ref,expected", [
    ("SCCDSR14150250029", ("angiolite", 2.5, 29)),
    ("SCCDSR14150400019", ("angiolite", 4.0, 19)),
    ("BCPR14N150250015", ("xperience pro", 2.5, 15)),
    ("BCDPR14N150300040", ("essential pro", 3.0, 40)),
    ("DC-RM2720HHW", ("accuforce", 2.75, 20)),
    ("DC-RM3515HHW", ("accuforce", 3.5, 15)),
    ("DC-RM2512HHW", ("accuforce", 2.5, 12)),
    ("DE-RS3024ASM", ("ultimaster nagomi", 3.0, 24)),
    ("DE-RS2733ASM", ("ultimaster nagomi", 2.75, 33)),
    ("REF DC-RM27 2OHHW", ("accuforce", 2.75, 20)),                     # spasi + O dibaca untuk 0
])
def test_ref_codes_from_photos(ref, expected):
    hits = parse_refs(ref)
    assert [(h.product, h.diameter, h.length) for h in hits] == [expected]


def test_ref_ignores_lot_and_dates():
    assert parse_refs("LOT 251023 2028-09-30 (01)04987350720245") == []


def test_ref_with_impossible_diameter_is_a_misread():
    # rekaman 30-09: "250" terbaca "259", "400" terbaca "409" -> dulu jadi bantahan palsu untuk kotak yang benar
    assert parse_refs("SCCDSR14150259029") == []
    assert parse_refs("SCCDSR14150409019") == []
    assert [(h.diameter, h.length) for h in parse_refs("SCCDSR14150275018")] == [(2.75, 18)]


# ------------------------------------------------------------------ katalog
def test_catalog_roundtrip(tmp_path):
    c = catalog(verified=False)
    c.path = tmp_path / "cat.csv"
    c.verify("04987350720245", "tester")
    c.save()
    d = Catalog(tmp_path / "cat.csv")
    assert d.get("04987350720245").verified and not d.get("04987350720061").verified
    assert d.get("04987350720245").key() == ("accuforce", 2.75, 20.0)
    assert d.gtins_for("Accuforce", 2.75, 20) == ["04987350720245"]
    assert d.get("08714729896845").diameter_mm is None


# ------------------------------------------------------------------ keputusan
REQ = Request("accuforce", 2.75, 20)


def run(ev, req=REQ, cat=None, **kw):
    return decide_multi(ev, req, catalog(True) if cat is None else cat, today=TODAY, **kw)[:2]


def test_verified_barcode_alone_is_match():
    assert run(Evidence(codes=[code("(01)04987350720245(17)280930(10)251023")]))[0] == MATCH


def test_unverified_catalog_is_never_used():
    d, reason = run(Evidence(codes=[code("(01)04987350720245(17)280930")]), cat=catalog(False))
    assert d == CONFIRM and "belum diverifikasi" in reason


def test_unknown_gtin_is_confirm():
    d, reason = run(Evidence(codes=[code("(01)05413206252428(17)280131")]))
    assert d == CONFIRM and "belum ada di katalog" in reason


def test_barcode_of_other_size_is_ignored():
    assert run(Evidence(codes=[code("(01)04987350720061(17)280930")]))[0] == IGNORED     # 2.5 x 12


def test_barcode_and_text_disagree_is_confirm():
    ev = Evidence(text=text("Accuforce", "2.5", "20", "mm"), codes=[code("(01)04987350720245(17)280930")])
    assert run(ev)[0] == CONFIRM


def test_expired_always_wins():
    ev = Evidence(text=text("Accuforce", "2.75", "20", "mm"), codes=[code("(01)04987350720245(17)250930")])
    assert run(ev)[0] == EXPIRED


def test_text_alone_is_only_candidate():
    assert run(Evidence(text=text("Accuforce", "2.75", "20", "mm")))[0] == CANDIDATE


def test_text_plus_ref_is_match():
    ev = Evidence(text=text("Accuforce", "2.75", "20", "mm"), refs=parse_refs("DC-RM2720HHW"))
    assert run(ev)[0] == MATCH


def test_text_and_ref_disagree_is_confirm():
    ev = Evidence(text=text("Accuforce", "2.75", "20", "mm"), refs=parse_refs("DC-RM2515HHW"))
    assert run(ev)[0] == CONFIRM


def test_ref_of_other_product_is_ignored():
    assert run(Evidence(refs=parse_refs("SCCDSR14150250029")))[0] == IGNORED


def test_two_gtins_in_one_box_is_confirm():
    ev = Evidence(codes=[code("(01)04987350720245(17)280930"), code("(01)04987350720061(17)280930")])
    assert run(ev)[0] == CONFIRM


def test_too_far_is_confirm():
    d, reason = run(Evidence(too_far=True))
    assert d == CONFIRM and "terlalu jauh" in reason


def test_ocr_gtin_counts_one_point():
    ocr_code = find_gtins_in_text("(01)04987350720245(17)280930")
    assert run(Evidence(codes=ocr_code))[0] == CANDIDATE
    assert run(Evidence(codes=ocr_code, text=text("Accuforce", "2.75", "20", "mm")))[0] == MATCH


def test_product_only_request():
    req = parse_request("Permanent Sled Bag")
    assert req.diameter is None and str(req) == "permanent sled bag"
    assert run(Evidence(codes=[code("(01)08714729896845(17)290426(10)39718940")]), req=req)[0] == MATCH
    assert run(Evidence(codes=[code("(01)04987350720245(17)280930")]), req=req)[0] == IGNORED


def test_same_product_other_size_never_matches():
    """Kasus paling berbahaya: produk sama, ukuran beda -> tidak boleh MATCH dari sumber mana pun."""
    for ev in (Evidence(text=text("Accuforce", "2.5", "12", "mm"), refs=parse_refs("DC-RM2512HHW")),
               Evidence(codes=[code("(01)04987350720061(17)280930")], text=text("Accuforce", "2.5", "12", "mm")),
               Evidence(text=text("Accuforce", "2.75", "12", "mm"))):
        assert run(ev)[0] != MATCH


def test_split_decimal_is_uncertain_not_contradicting():
    """Bacaan OCR nyata foto 67 dengan arah salah: '2' '75' 'mm' '2' 'mm' untuk '2.75 mm 20 mm'."""
    ident = text("2", "75", "mm", "2", "mm", "<L>")
    assert not ident.diameter_ok and ident.length is None      # tidak lagi "diameter 2 / panjang 75"
    ev = Evidence(text=ident, refs=parse_refs("DC-RM2720HHW"), codes=[code("(01)04987350720245(17)280930")])
    assert run(ev)[0] == MATCH                                  # barcode + REF cocok, teks ragu tidak menghalangi
    assert text("2 75 mm").diameter_ok is False


def test_diameter_symbol_read_as_8():
    """Bacaan nyata Nagomi foto 35: '8' '3.0' 'mm' '<L>' '24' 'mm' untuk '⌀ 3.0 mm <L> 24 mm'."""
    ident = text("Ultimaster Nagomi", "8", "3.0", "mm", "<L>", "24", "mm")
    assert (ident.diameter, ident.length) == (3.0, 24) and ident.diameter_ok and ident.length_ok
    ident = text("83.0", "mm", "24", "mm")
    assert ident.diameter == 3.0
    ident = text("8", "mm", "40", "mm")                     # "8" tanpa angka desimal sesudahnya tetap angka
    assert ident.diameter == 8.0


def test_parsing_errors_found_on_dataset():
    """Bacaan OCR nyata yang dulu menghasilkan diameter salah (benchmark 89 kotak)."""
    ident = text("2.75-20", "5", "G")                              # foto 53: '⌀2.75 - 20' Terumo + '5' lepas
    assert ident.diameter is None or not ident.diameter_ok or ident.diameter == 2.75
    ident = text("2.75-20")
    assert (ident.diameter, ident.length) == (2.75, 20)
    ident = text("1150300040", "026-06-15", "0-06-14")             # potongan tanggal -> bukan 6 / 15
    assert ident.diameter is None and ident.length is None
    ident = text("10", "mm")                                       # satu angka tidak boleh jadi keduanya
    assert not ident.diameter_ok and not ident.length_ok
    assert text("15", "2", "2029-09-21").diameter is None          # foto 50: '2' sisa '2,5' terpotong
    assert text("4", "mm", "19", "mm").key() == (None, 4, 19)      # diameter bulat iVascular tetap terbaca
    ident = text("STENT", "1", "mm", "19", "mm")                   # ① terbaca '1'
    assert ident.diameter is None and ident.length == 19
    assert text("2028-09-30", "2.5", "mm", "15", "mm").key() == (None, 2.5, 15)
    ident = text("275", "3")                                       # foto 38: '2.75' tanpa titik + '3' lepas
    assert not ident.diameter_ok
    ident = text("LOT 2512587", "19", "2028-09-21", "REF SCCDSR14150400019", "2025-09")   # foto 71: bukan diameter 9
    assert ident.diameter is None and ident.length == 19
    assert text("2.512", "mm").diameter is None                    # foto 36: '2.5' + '12' menempel


class FakeReader:
    def __init__(self, words):
        self.words = words

    def read_once(self, img):
        from src.box_ocr import OcrResult
        return OcrResult(tokens=[OcrToken(w, 0.99) for w in self.words])


def test_product_from_spine_completes_text_and_matches_with_ref():
    import numpy as np
    from src.box_evidence import gather_full_box
    cfg = {"catalog": NAMES, "ocr": {"min_confidence": 0.8}}
    ev = Evidence(text=text("4", "mm", "19", "mm"), refs=parse_refs("SCCDSR14150400019"))
    req = Request("angiolite", 4, 19)
    assert run(ev, req=req)[0] == CANDIDATE
    gather_full_box(ev, np.zeros((40, 400, 3), np.uint8), FakeReader(["REF", "angiolite", "RX"]), cfg)
    assert ev.text.product == "angiolite" and ev.text.product_ok
    assert run(ev, req=req)[0] == MATCH


def test_spine_product_conflicting_with_size_label_product_is_dropped():
    import numpy as np
    from src.box_evidence import gather_full_box
    cfg = {"catalog": NAMES, "ocr": {"min_confidence": 0.8}}
    ev = Evidence(text=text("Accuforce", "2.75", "20", "mm"))
    gather_full_box(ev, np.zeros((40, 400, 3), np.uint8), FakeReader(["Ultimaster", "Nagomi"]), cfg)
    assert ev.text.product is None and not ev.text.product_ok
    assert run(ev)[0] != MATCH


# ------------------------------------------------------------------ konsistensi + FEFO
def test_stable_states():
    st = TrackState()
    st.history.append((CANDIDATE, ("k",)))
    assert st.stable(2) == READING
    st.history.append((MATCH, ("k",)))
    assert st.stable(2) == CANDIDATE
    st.history.append((MATCH, ("k",)))
    assert st.stable(2) == MATCH
    st.history.append((EXPIRED, ("expired", "g")))
    assert st.stable(2) == EXPIRED


def test_pick_fefo_prefers_earliest_expiry():
    def matched(exp):
        st = TrackState(expiry=exp)
        st.history.extend([(MATCH, ("k",)), (MATCH, ("k",))])
        return st
    states = {1: matched(dt.date(2029, 3, 24)), 2: matched(dt.date(2028, 9, 30)), 3: matched(None), 4: TrackState()}
    assert pick_fefo(states, 2) == 2
    assert pick_fefo({4: TrackState()}, 2) is None


# ------------------------------------------------------------------ bukti gabungan antar bacaan
from src.box_session import pick_status  # noqa: E402
from src.box_verifier import EMPTY, support  # noqa: E402


def read(st, ev, req=REQ):
    d, _, k = decide_multi(ev, req, catalog(True), today=TODAY)
    st.add(d, k, support(ev, req, catalog(True), today=TODAY))
    return d


def test_text_in_one_read_and_ref_in_another_combine_to_match():
    st = TrackState()
    assert read(st, Evidence(text=text("Accuforce", "2.75", "20", "mm"))) == CANDIDATE
    assert read(st, Evidence(refs=parse_refs("DC-RM2720HHW"))) == CANDIDATE
    assert st.stable(2) == MATCH                                   # dulu: CANDIDATE sampai 2 bacaan lengkap berturut-turut
    assert pick_status(st)[0] == "solid"


def test_partial_text_pieces_and_ref_combine():
    st = TrackState()
    read(st, Evidence(text=text("Accuforce", "2.75")))             # diameter saja
    read(st, Evidence(text=text("Accuforce", "20", "mm"), refs=parse_refs("DC-RM2720HHW")))
    assert st.stable(2) == MATCH


def test_one_read_alone_never_combines():
    st = TrackState()
    read(st, Evidence(text=text("Accuforce", "2.75", "20", "mm"), refs=parse_refs("DC-RM2720HHW")))
    st.add(CONFIRM, EMPTY, (frozenset(), False))                    # bacaan kosong tidak ikut dihitung
    assert st.stable(2) == READING


def test_any_contradiction_in_window_blocks_combining():
    st = TrackState()
    read(st, Evidence(text=text("Accuforce", "2.75", "20", "mm")))
    assert read(st, Evidence(text=text("Accuforce", "2.5", "20", "mm"))) == IGNORED     # sekali terbaca 2.5
    read(st, Evidence(refs=parse_refs("DC-RM2720HHW")))
    assert st.stable(2) != MATCH
    for _ in range(3):                                                   # bacaan bertentangan keluar dari jendela
        read(st, Evidence(refs=parse_refs("DC-RM2720HHW")))
    assert st.stable(2) != MATCH                                         # teks sebelum pertentangan juga sudah keluar
    read(st, Evidence(text=text("Accuforce", "2.75", "20", "mm")))
    assert st.stable(2) == MATCH


def test_ref_of_other_size_blocks_combining():
    st = TrackState()
    read(st, Evidence(text=text("Accuforce", "2.75", "20", "mm")))
    read(st, Evidence(refs=parse_refs("DC-RM2720HHW")))
    read(st, Evidence(refs=parse_refs("DC-RM2515HHW")))
    assert st.stable(2) != MATCH and pick_status(st)[0] == "tahan"       # satu bantahan: tahan, JANGAN diambil
    read(st, Evidence(text=text("Accuforce", "2.5", "15", "mm")))
    assert pick_status(st)[0] == "batal"                                  # dua bantahan: batal


def test_locked_pick_stays_solid_through_partial_and_empty_reads():
    st = TrackState()
    read(st, Evidence(text=text("Accuforce", "2.75", "20", "mm")))
    read(st, Evidence(refs=parse_refs("DC-RM2720HHW")))
    assert pick_status(st)[0] == "solid"
    for ev in (Evidence(text=text("Accuforce")), Evidence(refs=parse_refs("DC-RM2720HHW")),
               Evidence(text=text("Accuforce", "20", "mm")), Evidence(text=text("Accuforce"))):
        read(st, ev)                                                      # bacaan sebagian: bukan bantahan
        assert pick_status(st)[0] == "solid"
    st.add(CONFIRM, EMPTY, (frozenset(), False))                          # satu bacaan kosong
    assert pick_status(st)[0] == "solid"


def test_single_misread_is_outvoted_by_following_reads():
    st = TrackState()
    read(st, Evidence(text=text("Accuforce", "2.75", "20", "mm")))
    read(st, Evidence(refs=parse_refs("DC-RM2720HHW")))
    read(st, Evidence(text=text("Accuforce", "2.75", "25", "mm")))       # salah baca 20 -> 25
    assert pick_status(st)[0] == "tahan"
    read(st, Evidence(text=text("Accuforce", "2.75", "20", "mm")))
    read(st, Evidence(refs=parse_refs("DC-RM2720HHW")))
    assert pick_status(st)[0] == "solid"


def test_two_different_gtins_never_combine():
    c = catalog(True)
    c.upsert(CatalogEntry(gtin="05413206252428", product="accuforce", diameter_mm=2.75, length_mm=20, verified=True))
    st = TrackState()
    for g in ("04987350720245", "05413206252428"):
        ev = Evidence(codes=[code(f"(01){g}(17)280930")])
        d, _, k = decide_multi(ev, REQ, c, today=TODAY)
        st.add(d, k, support(ev, REQ, c, today=TODAY))
    assert not st.combined_match(2)


def test_stale_clears_combined_evidence():
    st = TrackState()
    read(st, Evidence(text=text("Accuforce", "2.75", "20", "mm")))
    read(st, Evidence(refs=parse_refs("DC-RM2720HHW")))
    st.mark_stale()
    assert st.stable(2) == READING and not st.combined_match(2)
    read(st, Evidence(refs=parse_refs("DC-RM2720HHW")))
    assert st.stable(2) != MATCH                                       # bacaan lama tidak ikut lagi


def test_other_product_text_is_contradiction():
    atoms, contra = support(Evidence(text=text("Angiolite", "2.75", "20", "mm")), REQ, catalog(True), today=TODAY)
    assert contra and "tp" not in atoms


def test_unverified_barcode_gives_no_support():
    atoms, contra = support(Evidence(codes=[code("(01)04987350720245(17)280930")]), REQ, catalog(False), today=TODAY)
    assert not atoms and not contra


# ------------------------------------------------------------------ diameter bulat iVascular tanpa "mm" (audit 30-09)
def test_ivascular_whole_diameter_only_supports_never_contradicts():
    from src.box_ref import parse_refs
    # token asli rekaman tripod (K10 Angiolite 4x19): "4" terbaca 1,00 tapi dulu tidak pernah dipakai
    t = text("852152", "angiolite", "4", "19", "2028-09-21", "SCCD5R14150400019", "mm", "2025-09-22")
    assert not t.diameter_ok and t.diameter_hint == 4 and t.length == 19 and t.length_ok
    ev = Evidence(text=t, refs=parse_refs("SCCD5R14150400019"))
    assert run(ev, req=Request("angiolite", 4.0, 19.0))[0] == MATCH             # teks + REF: 2 bukti
    assert run(Evidence(text=t), req=Request("angiolite", 4.0, 19.0))[0] == CANDIDATE   # teks saja: tetap 1 bukti
    # potongan "3,5" terbaca "3": petunjuk yang tidak cocok TIDAK membantah permintaan 3.5x15
    cut = text("accuforce", "3", "15", "2028-01-31")
    assert cut.diameter_hint == 3 and run(Evidence(text=cut), req=Request("accuforce", 3.5, 15.0))[0] == CANDIDATE
    atoms, contra = support(Evidence(text=cut), Request("accuforce", 3.5, 15.0))
    assert not contra and "td" not in atoms
    atoms, contra = support(ev, Request("angiolite", 4.0, 19.0))
    assert {"tp", "td", "tl", "ref"} <= atoms and not contra


def test_broken_date_fragments_are_not_sizes():
    t = text("angiolite", "4", "19", "22-505-2022", "12-50-B221")              # tanggal / LOT terbaca rusak
    assert t.length == 19 and t.length_ok and not any("several" in n for n in t.notes)
    assert text("accuforce", "2.75-20").key()[1:] == (2.75, 20)                 # format ukuran Terumo tetap terbaca
    assert text("15", "2", "2029-09-21").diameter_hint is None                 # sisa "2,5" terpotong: bukan petunjuk

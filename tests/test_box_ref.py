"""pytest tests/test_box_ref.py -q  (REF codes as the RealSense D455 actually reads them, 08-10)"""
import pytest

from src.box_ref import parse_refs


@pytest.mark.parametrize("text", [
    "SCCDSR14150250029",          # as printed (phone photos)
    "SCCDSA14150250029",          # D455: R read as A
    "SCCD5A14150250029",          # ... and S as 5
    "SCCDSN14150250029",          # R read as N
    "SCCOSR14150250029",          # D read as O
    "10-00-5202P 29 25 SCCDSA14150250029 anle",   # real line from bacaan_uji1 (label upside down)
])
def test_angiolite_ref_survives_d455_misreads(text):
    assert [(h.product, h.diameter, h.length) for h in parse_refs(text)] == [("angiolite", 2.5, 29.0)]


def test_sibling_k10_keeps_its_own_size():
    # K10 (angiolite 4 x 19) sits next to K15: a relaxed prefix must never turn it into 2.5 x 29
    assert [(h.diameter, h.length) for h in parse_refs("SCCDSA14150400019")] == [(4.0, 19.0)]


@pytest.mark.parametrize("text", ["SCCXSA14150250029", "BCPRSA14150250029", "SCCDSA1415025"])
def test_not_an_angiolite_ref(text):
    assert not [h for h in parse_refs(text) if h.product == "angiolite"]


def test_printed_date_is_not_glued_to_a_ref_prefix():
    # stress test 15-10 K07: "DC-RM" + expiry "2028-09-30" read as REF DC-RM2028 = accuforce 2.0 x 28 (false contradiction)
    for t in ("DC-RM 2028-09-30", "DC-RM2028-09-30", "DC-RM 2028-0", "DC-RM2 2028-00"):
        assert parse_refs(t) == [], t
    assert [(h.product, h.diameter, h.length) for h in parse_refs("DC-RM2512HHW 2028-09-30")] == [("accuforce", 2.5, 12.0)]
    assert [(h.product, h.diameter, h.length) for h in parse_refs("SCCDSR14150250029 2028-07-31")] == [("angiolite", 2.5, 29.0)]


# 15-10: brands of the mentor's task photos. B. Braun codes from the official ordering table, Chroma from the photo.
@pytest.mark.parametrize("text, size", [
    ("REF 5023243", (2.75, 30.0)),               # the task 1 target box
    ("5O23243", (2.75, 30.0)),                   # 0 read as O
    ("5023262 LOT 25H12809", (2.5, 40.0)),
    ("5023226", (3.5, 20.0)),
    ("5023266", (3.5, 40.0)),
    ("5023207", (4.0, 10.0)),
])
def test_braun_sequent_ref_table(text, size):
    assert [(h.product, h.diameter, h.length) for h in parse_refs(text)] == [("sequent please neo", *size)]


@pytest.mark.parametrize("text", ["5023258", "5023290", "5023275"])   # diameter code 5 / 8 or length code 9 / 7: no such box
def test_braun_impossible_codes_are_not_read(text):
    assert not [h for h in parse_refs(text) if h.product == "sequent please neo"]


@pytest.mark.parametrize("text, size", [
    ("REF BCR-2733", (2.75, 33.0)),              # the task 2 target box
    ("BCR-4014", (4.0, 14.0)), ("BCR3514", (3.5, 14.0)), ("8CR-2536", (2.5, 36.0)), ("BCR-2209", (2.25, 9.0)),
])
def test_chroma_ref(text, size):
    assert [(h.product, h.diameter, h.length) for h in parse_refs(text)] == [("chroma", *size)]


def test_chroma_unknown_diameter_code_is_not_read():
    assert not parse_refs("BCR-2833")


@pytest.mark.parametrize("text", ["NEFE50232243", "NE502324246 2027-04-16", "50232262 T 25H12809", "BCR-27333"])
def test_digit_inserted_into_a_short_code_is_not_another_size(text):
    # task 1 photo 15-10: real D455-like misreads of 5023243 (2.75 x 30), 5023246 (3.5 x 30), 5023262 (2.5 x 40)
    assert parse_refs(text) == []


def test_braun_code_glued_to_the_ref_word_is_read():
    assert [(h.diameter, h.length) for h in parse_refs("E5023243 (01)04046963821906")] == [(2.75, 30.0)]
    assert [(h.diameter, h.length) for h in parse_refs("REF 5023243 Lo25G01809 2027-06-30")] == [(2.75, 30.0)]


@pytest.mark.parametrize("text", [
    "2.75-2 Accuforce DCRM272 5- 20 261023 2028-09-30 TERU HHW TERUMO",   # K16 (2.75 x 20): REF piece + "5" = 2.75 x 25
    "Accuforce 3.51 5 15 260226 DC- g 2029-01-31 DC-RM3516HHW TERUMO",     # K21 (3.5 x 15) read 3.5 x 16
    "TERUMO 1-10 DE-RS1024A5M 9 2028-01-31",                               # K18 (3.0 x 24) read 1.0 x 24
    "DE-RS2731ASM", "DE-RS2713ASM",                                        # K13 (2.75 x 33) read 31 / 13
])
def test_terumo_size_not_in_the_makers_table_is_not_read(text):
    # D455 shelf clip 16-10 (scripts/second_source_audit.py): these were the wrong REFs; none is a size Terumo makes
    assert parse_refs(text) == []


@pytest.mark.parametrize("code,size", [
    ("DC-RM2720HHW", ("accuforce", 2.75, 20.0)), ("DC-RM3530HHW", ("accuforce", 3.5, 30.0)),
    ("DC-RM5015HSW", ("accuforce", 5.0, 15.0)), ("DC-RR2010HHW", ("ryurei", 2.0, 10.0)),
    ("DC-RR1210HH", ("ryurei", 1.25, 10.0)), ("DE-RS2733ASM", ("ultimaster nagomi", 2.75, 33.0)),
    ("DE-RS4550ASM", ("ultimaster nagomi", 4.5, 50.0)),
])
def test_terumo_sizes_from_the_makers_table_are_read(code, size):
    assert [(h.product, h.diameter, h.length) for h in parse_refs(code)] == [size]


@pytest.mark.parametrize("text,size", [
    ("Accuforce 3.5 260226 3.5 15 2029-01-31 26 RM3515HHW 1010 W TERUMO", ("accuforce", 3.5, 15.0)),  # K21, 20-10
    ("OC-RM2720HHW", ("accuforce", 2.75, 20.0)), ("C-RM2720HHW", ("accuforce", 2.75, 20.0)),          # K16, 20-10
    ("RR2010HHW", ("ryurei", 2.0, 10.0)), ("RS2733ASM", ("ultimaster nagomi", 2.75, 33.0)),
])
def test_terumo_code_without_its_prefix_is_read_only_with_the_full_suffix(text, size):
    assert [(h.product, h.diameter, h.length) for h in parse_refs(text)] == [size]


@pytest.mark.parametrize("text", [
    "RM3515", "RM3515 HH", "RM2720HSW", "RR2010", "RS2733",          # no full suffix: a bare "RM" + digits is not a code
    "RM 2028-09-30 HHW", "RM1515HHW",                                  # a date after RM; 1.5 mm is not an Accuforce size
])
def test_terumo_code_without_prefix_and_suffix_is_not_read(text):
    assert parse_refs(text) == []


def test_ambiguous_terumo_digit_keeps_only_sizes_that_exist():
    from src.box_ref import parse_ref_hints
    (h,) = parse_ref_hints("DC-RM2812HHW")                # 2.8 -> 2.5 or 2.0, both made in 12 mm
    assert sorted(h.alts) == [2.0, 2.5]
    (h,) = parse_ref_hints("DC-RM2825HHW")                # 25 mm: only 2.5 x 25 exists, not 2.0 x 25
    assert h.alts == (2.5,)


@pytest.mark.parametrize("text,size", [
    ("2026-90-0 DC RM2T20NHWN 2026-900", ("accuforce", 2.75, 20.0)),   # K16 live webcam 23-10: the 7 read as T, suffix damaged
    ("DC-RM2T20HHW", ("accuforce", 2.75, 20.0)), ("DC-RR2T20HHW", ("ryurei", 2.75, 20.0)),
])
def test_t_read_for_a_seven_in_a_terumo_code(text, size):
    hits = [(h.product, h.diameter, h.length) for h in parse_refs(text)]
    assert hits == [size]


def test_t_is_not_a_digit_outside_terumo_codes():
    assert parse_refs("BCPRT4N150300040") == [] and parse_refs("2020-T-30 RM2T20") == []


def test_six_read_for_a_five_supports_but_never_makes_a_code():
    from src.box_ref import parse_ref_hints
    assert parse_refs("DC-RM2612HHW") == []                     # 2.6 is no Terumo size: not a code
    (h,) = parse_ref_hints("DC-RM2612HHW")                      # K07 live 23-10: 2.5 x 12 read with a 6
    assert h.alts == (2.5,)
    assert parse_ref_hints("DC-RM2613HHW") == []                # no 2.5 x 13 in the Accuforce table: nothing to support

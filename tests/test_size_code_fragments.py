"""Code/date OCR fragments must neither supply sizes nor hide real conflicts."""
import pytest

from src.box_ocr import OcrToken
from src.box_verifier import parse_identity, _size_token_text


def identity(*texts):
    return parse_identity([OcrToken(t, .99) for t in texts], ["essential pro"], .9,
                          (1., 10.), (6., 60.))


@pytest.mark.parametrize("fragment", ["20/0L92", "20/OCT26", "24/LOT92", "40/A3",
                                      "20/0192", "20/2192", "24/024"])
def test_alphanumeric_slash_fragment_cannot_supply_a_length(fragment):
    result = identity("essential pro", "3 mm", fragment)
    assert result.diameter_ok and result.diameter == 3
    assert not result.length_ok and result.length is None
    assert fragment in result.text


def test_real_size_survives_nearby_code_fragment_without_inventing_missing_numbers():
    result = identity("essential pro", "3 mm", "40 mm", "20/0L92")
    assert result.diameter_ok and result.length_ok and result.length == 40
    result = identity("essential pro", "20/0L92")
    assert not result.diameter_ok and not result.length_ok


@pytest.mark.parametrize("size", ["3/24", "3/24mm", "3/24 mm", "3/24rnm", "3/24rn", "3/24nm"])
def test_real_slash_size_is_retained(size):
    assert _size_token_text(size) == size
    result = identity("essential pro", size)
    assert result.length_ok and result.length == 24


def test_actual_conflicting_length_is_still_ambiguous():
    result = identity("essential pro", "3 mm", "40 mm", "20 mm", "20/0L92")
    assert not result.length_ok and result.length is None
    assert any("20" in note and "40" in note for note in result.notes)


def test_original_terumo_size_and_dates_keep_their_existing_meaning():
    assert _size_token_text("2.75-20") == "2.75-20"
    result = identity("essential pro", "3 mm", "40 mm", "2029/06/14")
    assert result.length_ok and result.length == 40
    assert _size_token_text("3/200") == "3/200"


def test_masking_a_code_fragment_never_supplies_a_second_source_or_hides_real_conflict():
    from src.box_ref import parse_refs
    from src.box_verifier import Evidence, MATCH, decide_multi, parse_request
    request = parse_request("essential pro 3 x 40")
    refs = parse_refs("BCDPR14N150300040")
    incomplete = Evidence(text=identity("essential pro", "3 mm", "20/0192"), refs=refs)
    assert decide_multi(incomplete, request)[0] != MATCH
    complete = Evidence(text=identity("essential pro", "3 mm", "40 mm", "20/0192"), refs=refs)
    assert decide_multi(complete, request)[0] == MATCH
    contradictory = Evidence(text=identity("essential pro", "3 mm", "20 mm", "20/0192"), refs=refs)
    assert decide_multi(contradictory, request)[0] != MATCH

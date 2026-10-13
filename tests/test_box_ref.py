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

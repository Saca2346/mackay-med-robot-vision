"""pytest tests/test_label_band.py -q  (Terumo label band read for the REF line; selects pixels only)"""
import numpy as np
import pytest

from src.box_evidence import gather_full_box, label_band
from src.box_ocr import OcrResult, OcrToken
from src.box_verifier import Evidence

NAMES = ["angiolite", "accuforce", "ryurei", "ultimaster nagomi", "permanent sled bag"]


def strip():
    s = np.full((40, 600, 3), 30, np.uint8)       # dark brand block | white label | dark brand block
    s[:, 150:450] = 235
    return s


class Reader:
    def __init__(self, words):
        self.words, self.widths = words, []

    def read_once(self, img):
        self.widths.append(img.shape[1])
        return OcrResult(tokens=[OcrToken(w, 0.99) for w in self.words])


def cfg(band=True):
    return {"catalog": NAMES, "ocr": {"min_confidence": 0.8}, "verify": {"full_box_label_band": band}}


def test_label_band_is_the_white_run_between_the_dark_blocks():
    band = label_band(strip())
    assert band is not None and 300 <= band.shape[1] <= 370
    assert band.mean() > 150
    assert label_band(np.full((40, 600, 3), 235, np.uint8)) is None            # no clear label
    assert label_band(np.zeros((600, 40, 3), np.uint8)) is None                # not a horizontal strip
    short = np.full((40, 600, 3), 30, np.uint8)
    short[:, 280:300] = 235
    assert label_band(short) is None                                           # a bright speck is not a label


def test_band_read_finds_ref_and_product_with_fewer_reads_than_the_whole_spine():
    full, banded = Reader(["Accuforce", "DC-RM2512HHW"]), Reader(["Accuforce", "DC-RM2512HHW"])
    gather_full_box(Evidence(), strip(), full, cfg(False), product="accuforce")
    ev = Evidence()
    n, _ = gather_full_box(ev, strip(), banded, cfg(True), product="accuforce")
    assert [(r.product, r.diameter, r.length) for r in ev.refs] == [("accuforce", 2.5, 12.0)]
    assert n == len(banded.widths) < len(full.widths)


def test_band_without_product_name_falls_back_to_the_whole_spine():
    reader = Reader(["DC-RM2512HHW"])                                          # the name sits on a dark block
    ev = Evidence()
    gather_full_box(ev, strip(), reader, cfg(True), product="accuforce")
    whole = Reader(["DC-RM2512HHW"])
    gather_full_box(Evidence(), strip(), whole, cfg(False), product="accuforce")
    assert len(reader.widths) > len(whole.widths)                              # band read + the old whole-spine read
    assert ev.refs


@pytest.mark.parametrize("product,band", [("angiolite", True), ("accuforce", False), (None, True)])
def test_other_products_and_the_default_keep_the_whole_spine_read(product, band):
    a, b = Reader(["Accuforce"]), Reader(["Accuforce"])
    gather_full_box(Evidence(), strip(), a, cfg(band), product=product)
    gather_full_box(Evidence(), strip(), b, cfg(False))
    assert a.widths == b.widths


def logo_and_label():
    logo = np.full((180, 54, 3), 50, np.uint8)
    logo[100:110, 20:30] = 240                                                  # ~2% bright: the TERUMO logo block
    label = np.full((380, 58, 3), 95, np.uint8)
    label[:140] = 235                                                           # ~37% bright: white label with digits
    return logo, label


def test_dark_logo_crop_is_dropped_only_when_a_bright_label_crop_exists():
    from src.box_evidence import bright_label_crops
    logo, label = logo_and_label()
    kept = bright_label_crops([logo, label])
    assert len(kept) == 1 and kept[0] is label
    assert bright_label_crops([label, logo])[0] is label
    dark = [logo, logo.copy()]
    assert bright_label_crops(dark) is dark                                    # a dark exposure keeps every crop
    assert bright_label_crops([logo]) == [logo]                                # a single crop is never dropped
    assert len(bright_label_crops([label, label.copy()])) == 2


class OrientReader(Reader):
    def read_orient(self, img, rot):
        return self.read_once(img)


@pytest.mark.parametrize("flag,product,expected_calls", [(True, "accuforce", 1), (False, "accuforce", 2),
                                                          (True, "angiolite", 2)])
def test_gather_skips_the_logo_crop_for_terumo_requests_only_when_enabled(flag, product, expected_calls):
    from src.box_evidence import gather
    from src.box_verifier import Request
    logo, label = logo_and_label()
    cfg_ = {"catalog": NAMES, "ocr": {"min_confidence": 0.8}, "matching": {"diameter_range_mm": [1.0, 10.0],
            "length_range_mm": [6.0, 200.0]}, "verify": {"skip_dark_label_crops": flag, "decode_barcodes": False}}
    reader = OrientReader(["Accuforce", "2.5", "12", "mm"])
    gather([logo, label], None, reader, cfg_, Request(product, 2.5, 12))
    assert len(reader.widths) == expected_calls

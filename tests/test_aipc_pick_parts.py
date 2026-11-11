"""aipc/tasks/pick_parts.py: the product / brand / detail / barcode crops of a PICKed box come from the tokens the OCR
read (10 Nov: the product name and its maker are two crops)."""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "aipc" / "tasks"))
import pick_parts as pp  # noqa: E402

from src.box_ocr import OcrToken  # noqa: E402
from src.box_verifier import Request  # noqa: E402


def tok(text, x0, y0, x1, y1, conf=0.99):
    return OcrToken(text, conf, np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]))


def plain_box():
    return np.full((400, 100, 3), 200, np.uint8), [[10, 10], [90, 10], [90, 390], [10, 390]]


def test_product_brand_and_detail_are_cut_from_the_tokens_read_and_the_ref_joins_from_another_input():
    label = np.full((60, 600, 3), 255, np.uint8)
    tile = np.full((40, 300, 3), 255, np.uint8)
    seen = [(label, [tok("Accuforce", 10, 10, 110, 40), tok("3.5", 150, 10, 190, 40), tok("15", 200, 10, 230, 40),
                     tok("TERUMO", 500, 10, 590, 40)]),
            (tile, [tok("DC-RM3515HHW", 5, 5, 200, 30)])]
    frame, poly = plain_box()
    p = pp.parts(frame, poly, seen, Request("accuforce", 3.5, 15.0))
    assert p["product_text"] == "Accuforce"                   # the maker's name printed far apart is not merged in
    assert p["product"].shape[1] < 200
    assert p["brand_text"] == "TERUMO" and p["brand"].shape[1] < 120   # the maker: its own crop
    assert p["detail_text"] == "3.5 15 | DC-RM3515HHW"         # size and REF: two blocks of print
    assert len(p["detail_blocks"]) == 2
    assert p["barcode"] is None and p["barcode_px"] is None    # a plain grey box has no bars


def test_a_size_number_printed_elsewhere_is_looked_for_in_the_other_inputs():
    a = np.full((60, 400, 3), 255, np.uint8)
    b = np.full((60, 400, 3), 255, np.uint8)
    seen = [(a, [tok("30", 10, 10, 50, 40), tok("REF5023243", 60, 10, 200, 40)]),
            (b, [tok("SeQuent", 10, 10, 100, 40), tok("2.75", 110, 10, 160, 40)])]
    frame, poly = plain_box()
    p = pp.parts(frame, poly, seen, Request("sequent please neo", 2.75, 30.0))
    assert p["detail_text"] == "30 REF5023243 | 2.75"
    assert pp.stack(p["detail_blocks"]).shape[0] > 60


def test_product_name_words_and_the_maker_next_to_them_are_two_crops():
    img = np.full((50, 700, 3), 255, np.uint8)
    seen = [(img, [tok("B|BRAUN", 5, 10, 90, 40), tok("SeQuent", 100, 10, 200, 40), tok("Please", 210, 10, 290, 40),
                   tok("NEO", 300, 10, 350, 40), tok("2.75", 500, 10, 560, 40)])]
    frame, poly = plain_box()
    p = pp.parts(frame, poly, seen, Request("sequent please neo", 2.75, 30.0))
    assert p["product_text"] == "SeQuent Please NEO" and p["product"].shape[1] < 270
    assert p["brand_text"] == "B|BRAUN" and p["brand"].shape[1] < 110
    assert p["detail_text"] == "2.75"                          # only what was read; 30 was not read here


def test_size_tokens_exclude_ref_codes_dates_and_other_numbers():
    req = Request("chroma", 2.75, 33.0)
    assert pp.is_size("2.75", req) and pp.is_size("33mm", req) and pp.is_size("2.75x33mm", req)
    assert not pp.is_size("BCR2733", req)                      # a REF code is detail via is_ref, not a size token
    assert not pp.is_size("2027-03-31", req)
    assert not pp.is_size("3.0", req) and not pp.is_size("LOT 33AB12", req)
    assert pp.is_ref("BCR2733", req) and not pp.is_ref("BCR2730", req)


def test_low_confidence_or_other_product_tokens_give_no_crop():
    img = np.full((50, 400, 3), 255, np.uint8)
    seen = [(img, [tok("Accuforce", 5, 5, 100, 40, conf=0.3), tok("Ryurei", 120, 5, 200, 40)])]
    frame, poly = plain_box()
    p = pp.parts(frame, poly, seen, Request("accuforce", 3.5, 15.0))
    assert p["product"] is None and p["product_text"] == "" and p["brand"] is None
    assert p["detail_blocks"] == [] and p["detail_text"] == ""


def test_save_writes_the_crops_and_their_texts(tmp_path):
    img = np.full((60, 400, 3), 255, np.uint8)
    seen = [(img, [tok("CHROMA", 10, 10, 120, 40), tok("2.75", 200, 10, 250, 40), tok("33", 260, 10, 290, 40)])]
    frame, poly = plain_box()
    info = pp.save(str(tmp_path / "c"), pp.parts(frame, poly, seen, Request("chroma", 2.75, 33.0)))
    assert info["files"] == {"product": "product.png", "detail": "detail.png", "spine": "spine.png"}
    assert (info["product_name"], info["brand_name"]) == ("Chroma", "Biosensors")
    assert info["brand_from"] == "not read on this face"        # no maker read, no reader to read the box again
    assert info["barcode"] == "not found" and info["detail_text"] == "2.75 33"
    assert (tmp_path / "c" / "crops.json").exists()


def _gtin(body13):
    s = sum(int(c) * (3 if i % 2 == 0 else 1) for i, c in enumerate(reversed(body13)))
    return body13 + str((10 - s % 10) % 10)


def test_gtin_text_joins_the_detail_and_a_block_the_others_already_show_is_dropped():
    a = np.full((60, 500, 3), 255, np.uint8)
    b = np.full((60, 500, 3), 255, np.uint8)
    c = np.full((60, 500, 3), 255, np.uint8)
    seen = [(a, [tok("5023243", 10, 10, 120, 40)]),                                   # the REF alone, read best
            (b, [tok("2.7530", 10, 10, 90, 40), tok("5023243", 100, 10, 200, 40, conf=0.9)]),
            (c, [tok("(01)" + _gtin("0404696300000"), 10, 10, 300, 40)])]
    frame, poly = plain_box()
    p = pp.parts(frame, poly, seen, Request("sequent please neo", 2.75, 30.0))
    assert p["detail_text"] == "2.7530 5023243 | (01)" + _gtin("0404696300000")
    assert [r[1] for r in p["read"]] == ["5023243", "2.7530", "5023243", "(01)" + _gtin("0404696300000")]


def test_text_running_down_a_tall_input_gives_a_tight_crop_turned_across_and_a_read_across_is_preferred():
    tall = np.full((600, 60, 3), 255, np.uint8)
    seen = [(tall, [tok("essential pro", 15, 20, 45, 220), tok("BCDPR14N150300040", 15, 240, 45, 560),
                    tok("x", 15, 0, 45, 15, conf=0.6)])]
    frame, poly = plain_box()
    p = pp.parts(frame, poly, seen, Request("essential pro", 3.0, 40.0))
    assert p["product_text"] == "essential pro"
    assert p["product"].shape[1] > 3 * p["product"].shape[0]    # turned to run across, not the whole tall spine
    assert p["product"].shape[1] < 260
    wide = np.full((60, 600, 3), 255, np.uint8)
    seen.append((wide, [tok("essential pro", 20, 15, 220, 45)]))
    p = pp.parts(frame, poly, seen, Request("essential pro", 3.0, 40.0))
    assert p["product"].shape == (48, 218, 3)                    # the same name read across wins: cut as it is


class _Reader:
    """RapidOCR's classifier turns upside-down text before reading: the upright crop reads the same with it on and off,
    the turned crop reads its digits with the classifier on but "ww" for "mm" without it."""

    def __init__(self, upright_img):
        self.up = upright_img

    def _read(self, img, cls):
        from types import SimpleNamespace
        text = "2,5 29 mm" if np.array_equal(img, self.up) or cls else "2,5 29 ww"
        return SimpleNamespace(tokens=[OcrToken(w, 0.99) for w in text.split()])

    def read_as_is(self, img):
        return self._read(img, False)

    def read_once(self, img):
        return self._read(img, True)


def test_upright_turns_a_crop_that_reads_differently_without_the_classifier():
    up = np.zeros((20, 60, 3), np.uint8)
    up[:, :10] = 255                                           # not symmetric under a 180 degree turn
    down = np.ascontiguousarray(up[::-1, ::-1])
    img, turned = pp.upright(down, _Reader(up))
    assert turned and np.array_equal(img, up)
    img, turned = pp.upright(up, _Reader(up))
    assert not turned


def test_upright_follows_the_direction_classifier_asked_on_both_turns():
    from types import SimpleNamespace
    up = np.zeros((40, 200, 3), np.uint8)
    up[:, :20] = 255                                           # upright = the white stripe on the left
    down = np.ascontiguousarray(up[::-1, ::-1])

    class Cls:
        def __call__(self, lines):
            return SimpleNamespace(cls_res=[("0", 0.9) if c[:, :5].mean() > 128 else ("180", 0.8) for c in lines])

    whole = np.float32([[0, 0], [199, 0], [199, 39], [0, 39]])
    reader = SimpleNamespace(_engine=SimpleNamespace(text_cls=Cls()),
                             read_as_is=lambda x: SimpleNamespace(tokens=[OcrToken("12", 0.99)]),   # a tie: no rule 1
                             read_once=lambda x: SimpleNamespace(tokens=[OcrToken("12", 0.99, whole)]))
    out, turned = pp.upright(down, reader, "12")
    assert turned and np.array_equal(out, up)
    assert not pp.upright(up, reader, "12")[1]                 # stable: the upright crop stays


def test_upright_first_follows_the_tokens_the_run_read():
    from types import SimpleNamespace
    up = np.zeros((20, 60, 3), np.uint8)
    up[:, :10] = 255
    down = np.ascontiguousarray(up[::-1, ::-1])
    reads = lambda x: SimpleNamespace(tokens=[OcrToken(w, 0.99) for w in (["12", "DC-RM2512HHW"] if np.array_equal(x, up)  # noqa: E731
                                                                            else ["12", "MHHZISZWH-OO"])])
    reader = SimpleNamespace(read_as_is=reads, read_once=reads)
    img, turned = pp.upright(down, reader, "12 DC-RM2512HHW")
    assert turned and np.array_equal(img, up)
    assert not pp.upright(up, reader, "12 DC-RM2512HHW")[1]
    joined = [OcrToken("2.75 33", 0.99)]
    split = [OcrToken("33", 0.99), OcrToken("2.75", 0.99)]
    assert pp._matches(joined, "2.75 33") == pp._matches(split, "2.75 33") == 2   # a tie: left to the classifier
    assert pp._matches([OcrToken("61", 0.9)], "19") == 0 and pp._matches([OcrToken("19", 0.9)], "19") == 1


def test_names_are_the_printed_names_and_agree_with_the_laptop_brands():
    from src.box_verifier import BRANDS, PRODUCTS, brand_of
    assert set(pp.NAMES) == set(PRODUCTS)
    for key, (product, brand) in pp.NAMES.items():
        assert pp.letters(product) == pp.letters(key)
        assert pp.letters(brand) == pp.letters(brand_of(key)), key
    assert pp.names("essential pro") == ("Essential Pro", "iVascular")
    assert pp.names("sequent please neo") == ("SeQuent Please NEO", "B. Braun")
    assert set(BRANDS)


class _SpineReader:
    """read_once: the maker's name across the image when the spine was turned one way, nothing the other way."""

    def __init__(self):
        self.calls = 0

    def read_once(self, img):
        self.calls += 1
        toks = [tok("iVascular", 20, 10, 160, 40)] if self.calls == 1 else []
        return type("R", (), {"tokens": toks})()


def test_a_brand_the_outline_did_not_read_is_read_on_the_box_after_the_pick(tmp_path):
    img = np.full((60, 400, 3), 255, np.uint8)
    seen = [(img, [tok("Angiolite", 10, 10, 120, 40), tok("2.5", 200, 10, 240, 40), tok("29", 250, 10, 290, 40)])]
    frame, poly = plain_box()
    p = pp.parts(frame, poly, seen, Request("angiolite", 2.5, 29.0))
    assert p["brand"] is None
    crop, text = pp.brand_from_spine(p["spine"], _SpineReader(), "angiolite")
    assert text == "iVascular" and crop is not None and crop.shape[1] < 200
    assert pp.brand_from_spine(p["spine"], None, "angiolite") == (None, "")


class _TurnReader:
    """read_as_is: a near reading of the maker on both turns, the upright one (a white left half here) surer."""

    def read_as_is(self, img):
        upright = img[:, : img.shape[1] // 2].mean() > img[:, img.shape[1] // 2:].mean()
        t = tok("TERUMC", 5, 5, 100, 30)
        t.conf = 0.92 if upright else 0.70
        if not upright:
            t.text = "TERUMD"
        return type("R", (), {"tokens": [t]})()


def test_the_brand_crop_is_shown_on_the_turn_that_reads_the_maker_surer():
    img = np.zeros((40, 200, 3), np.uint8)
    img[:, 100:] = 255                                         # upside down: the white half is on the right
    out, turned = pp.upright_maker(img, _TurnReader(), "accuforce", "TERUMO")
    assert turned and out[:, :100].mean() > out[:, 100:].mean()
    out, turned = pp.upright_maker(out, _TurnReader(), "accuforce", "TERUMO")
    assert not turned

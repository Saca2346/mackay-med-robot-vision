"""PICK crop geometry: corners must remain invertible at every box rotation."""
from types import SimpleNamespace

import cv2
import numpy as np

from src.box_crops import pick_crops, rectify, to_frame


def test_diamond_label_rectifies_without_duplicate_corners():
    # Sum/difference corner sorting ties at 45 degrees and can select one vertex twice.
    p = np.float32([[80, 40], [120, 80], [80, 120], [40, 80]])
    frame = np.zeros((160, 160, 3), np.uint8)
    cv2.fillConvexPoly(frame, p.astype(np.int32), (40, 80, 160))
    crop, transform = rectify(frame, p)
    h, w = crop.shape[:2]
    back = to_frame([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], transform)
    assert np.allclose(crop[h // 2, w // 2], [40, 80, 160])
    assert all(np.min(np.linalg.norm(back - vertex, axis=1)) < 1e-4 for vertex in p)


def test_decoded_barcode_on_diamond_box_has_finite_camera_coordinates():
    p = np.float32([[80, 40], [120, 80], [80, 120], [40, 80]])
    frame = np.full((160, 160, 3), 160, np.uint8)
    group = SimpleNamespace(box=SimpleNamespace(poly=p), labels=[])

    def decode(img, hard):
        h, w = img.shape[:2]
        return [SimpleNamespace(poly=np.float32([[w * .25, h * .25], [w * .75, h * .25],
                                                 [w * .75, h * .75], [w * .25, h * .75]]),
                                raw="decoded GS1", gtin="")]

    crops = pick_crops(frame, group, decode)
    assert crops["barcode_decoded"] and crops["barcode"].size
    assert np.isfinite(crops["barcode_px"]).all()
    assert np.allclose(np.mean(crops["barcode_px"], axis=0), [80, 80], atol=1.0)


# ---- owner review of 23-10: crops shown in the evidence panel
from types import SimpleNamespace

from src.box_crops import barcode_with_text, best_date_crop, numeral_crop, pick_number_tiles, upright_date


def test_numeral_crop_keeps_the_unit_below_the_numeral():
    img = np.zeros((200, 100, 3), np.uint8)
    poly = np.float32([[30, 40], [70, 40], [70, 80], [30, 80]])               # numeral 40 px tall
    tight, wide = numeral_crop(img, poly, unit=False), numeral_crop(img, poly)
    assert tight.shape[0] == 44 and wide.shape[0] >= 40 + 0.7 * 40 and wide.shape[1] > tight.shape[1] - 4
    edge = numeral_crop(img, np.float32([[0, 190], [10, 190], [10, 199], [0, 199]]))   # never outside the image
    assert edge.size and edge.shape[0] <= 12 and edge.shape[1] <= 12


def test_number_tiles_one_per_size_value_no_date_fragment_no_duplicate():
    n = lambda text, conf: {"text": text, "confidence": conf, "image": text}
    got = pick_number_tiles([n("25", 0.9), n("2,5", 0.95), n("2,5", 0.99), n("15", 1.0), n("202", 1.0), n("20", 0.9)], 2.5, 15.0)
    assert [(g["text"], g["confidence"]) for g in got] == [("2,5", 0.99), ("15", 1.0)]
    assert pick_number_tiles([n("2,5", 1.0)], 4.0, 19.0) == []                   # a wrong numeral is never shown for the size
    assert [g["text"] for g in pick_number_tiles([n("2,5", 0.9), n("2.5", 0.8), n("15", 0.7)])] == ["2,5", "15"]


def test_date_crop_needs_a_complete_confident_date():
    band = np.full((100, 300, 3), 200, np.uint8)
    tok = lambda text, conf, poly=((10, 10), (100, 10), (100, 30), (10, 30)): SimpleNamespace(text=text, conf=conf, poly=np.float32(poly))
    img, text = best_date_crop([{"image": band, "tokens": [tok("202", 1.0), tok("2029-03-12", 0.7), tok("2029-03-12", 0.99)]}])
    assert text == "2029-03-12" and img is not None and img.shape[0] > 20
    assert best_date_crop([{"image": band, "tokens": [tok("202", 1.0), tok("2029-03-12", 0.7)]}]) == (None, None)


def test_upright_date_turns_an_upside_down_crop_only():
    crop = np.zeros((20, 60, 3), np.uint8)
    crop[:10] = 255                                                              # top half white: tells the two turns apart
    reads_turned = SimpleNamespace(read_as_is=lambda img: SimpleNamespace(
        tokens=[SimpleNamespace(text="2029-03-12", conf=1.0)] if img[0, 0, 0] == 0 else []))
    assert upright_date(crop, reads_turned)[0, 0, 0] == 0                        # only the turned crop reads: use it
    reads_as_is = SimpleNamespace(read_as_is=lambda img: SimpleNamespace(
        tokens=[SimpleNamespace(text="2029-03-12", conf=1.0)] if img[0, 0, 0] == 255 else []))
    assert upright_date(crop, reads_as_is)[0, 0, 0] == 255                       # already upright: unchanged
    assert upright_date(crop, None) is crop


def test_located_barcode_needs_a_real_bar_pattern():
    spine = np.full((400, 64, 3), 230, np.uint8)
    assert barcode_with_text(spine)[0] is None                                   # blank spine: nothing to show
    for y in range(150, 250, 6):                                                 # bars across the spine
        spine[y:y + 3, 8:56] = 20
    crop, score = barcode_with_text(spine)
    assert crop is not None and score >= 0.3 and crop.shape[1] == 64 and crop.shape[0] > 100

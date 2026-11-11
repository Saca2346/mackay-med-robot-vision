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

from src.box_crops import barcode_only, best_date_crop, numeral_crop, pick_number_tiles, upright_date


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


def synthetic_spine(seed=0):
    """Upright spine, 64 px across: GS1-like bars (x 6-40) and a REF / LOT print column beside them (x 46-60)."""
    rng = np.random.default_rng(seed)
    spine = np.full((400, 64, 3), 235, np.uint8)
    y = 150
    while y < 260:                                                              # bars of 1-3 px, gaps of 1-3 px
        w = int(rng.integers(1, 4))
        spine[y:y + w, 6:40] = 25
        y += w + int(rng.integers(1, 4))
    for y in range(150, 260, 9):                                                # rotated text: short strokes
        spine[y:y + 5, 46 + int(rng.integers(0, 4)):52 + int(rng.integers(0, 6))] = 40
    return spine


def test_barcode_only_keeps_the_bars_and_leaves_the_ref_print_out():
    spine = np.full((400, 64, 3), 230, np.uint8)
    assert barcode_only(spine) == (None, None)                                  # blank spine: nothing to show
    spine = synthetic_spine()
    crop, corners = barcode_only(spine)
    assert crop is not None and crop.shape[1] > crop.shape[0]                   # straightened: bars vertical
    xs, ys = corners[:, 0], corners[:, 1]
    assert xs.max() <= 44 and xs.min() >= 2                                     # the print column at x 46-60 stays out
    assert ys.min() >= 130 and ys.max() <= 280                                  # the bars and their quiet zone only


# ---- owner review of 29-10: upside-down size labels and logos shown as barcodes in the panel
from src.box_crops import barcode_like, bar_count, readable


def test_readable_chooses_the_turn_with_the_ocr_classifier_off():
    crop = np.zeros((120, 30, 3), np.uint8)
    crop[:, :15] = 255                                                           # left half white: tells the two turns apart
    hit = SimpleNamespace(tokens=[SimpleNamespace(text="Accuforce", conf=0.99)])
    miss = SimpleNamespace(tokens=[])
    # With the classifier on, both turns read (the upside-down one through the classifier); off, only the upright one.
    reader = SimpleNamespace(read_once=lambda img: hit,
                             read_as_is=lambda img: hit if img[0, 0, 0] == 0 else miss)
    got = readable(crop, reader)
    assert got.shape == (30, 120, 3) and got[0, 0, 0] == 0                       # the counter-clockwise turn
    assert np.array_equal(readable(crop), cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE))


def test_logo_or_long_label_is_not_shown_as_a_barcode():
    spine = np.full((400, 64, 3), 230, np.uint8)
    block = np.float32([[8, 150], [56, 150], [56, 250], [8, 250]])
    spine[170:200, 8:56] = 20                                                    # two thick stripes: a logo or a glare edge
    spine[215:235, 8:56] = 20
    assert bar_count(spine, block) < 8 and not barcode_like(spine, block, 0.9)
    for y in range(150, 250, 6):                                                 # thin bars: a barcode
        spine[y:y + 3, 8:56] = 20
    assert bar_count(spine, block) >= 8 and barcode_like(spine, block, 0.9)
    assert not barcode_like(spine, block, 0.2)                                   # low bar-pattern score
    long = np.float32([[8, 0], [56, 0], [56, 399], [8, 399]])
    assert not barcode_like(spine, long, 0.9)                                    # 6 spine widths: a whole label


def test_pick_crops_gives_no_barcode_position_for_a_logo():
    frame = np.full((400, 64, 3), 230, np.uint8)
    frame[170:200, 8:56] = 20
    frame[215:235, 8:56] = 20
    group = SimpleNamespace(box=SimpleNamespace(poly=np.float32([[0, 0], [63, 0], [63, 399], [0, 399]])), labels=[])
    crops = pick_crops(frame, group)
    assert crops["barcode"] is None and crops["barcode_px"] is None


def test_find_barcode_merges_pieces_and_rejects_square_or_few_bar_outlines(monkeypatch):
    import src.box_crops as bc
    spine = synthetic_spine(1)
    img = cv2.rotate(spine, cv2.ROTATE_90_CLOCKWISE)                            # horizontal spine, as the reader has it
    # rotated clockwise: the bars (upright y 150-260, x 6-40) lie at x 139-249, y 6-40
    pieces = np.float32([[[136, 6], [196, 6], [196, 40], [136, 40]],             # one barcode in two pieces (low resolution)
                         [[194, 6], [252, 6], [252, 40], [194, 40]],
                         [[330, 10], [352, 10], [352, 32], [330, 32]]])          # a square outline on the print: rejected
    fake = SimpleNamespace(detect=lambda im: (True, pieces * (im.shape[0] / img.shape[0])))
    monkeypatch.setattr(bc, "_cv_detector", lambda: fake)
    corners, bars = bc.find_barcode(img)
    assert corners is not None and bars >= 8
    assert corners[:, 0].min() < 150 and corners[:, 0].max() > 240            # the two pieces became one outline
    blank = np.full_like(img, 230)
    monkeypatch.setattr(bc, "_cv_detector", lambda: SimpleNamespace(detect=lambda im: (True, pieces[:1] * (im.shape[0] / img.shape[0]))))
    assert bc.find_barcode(blank) == (None, 0)                                  # an outline with no bars in it: nothing


def test_pick_crops_barcode_is_straightened_bars_with_frame_corners():
    spine = synthetic_spine(2)
    group = SimpleNamespace(box=SimpleNamespace(poly=np.float32([[0, 0], [63, 0], [63, 399], [0, 399]])), labels=[])
    crops = pick_crops(spine, group)
    assert crops["barcode"] is not None and crops["barcode"].shape[1] > crops["barcode"].shape[0]
    px = np.asarray(crops["barcode_px"])
    assert px.shape == (4, 2) and px[:, 0].max() <= 44                          # aimed at the bars, not at the print


def test_panel_shows_a_thin_barcode_strip_taller_without_changing_its_bars():
    from src.box_crops import panel
    strip = np.full((10, 150, 3), 230, np.uint8)
    strip[:, ::6] = 20                                                          # bars every 6 px
    out = panel({"barcode": strip, "spine": np.full((200, 30, 3), 200, np.uint8)}, {"identity": "VERIFIED"}, height=300)
    assert out.shape[0] == 300
    tall = cv2.resize(strip, (150, 50), interpolation=cv2.INTER_LINEAR)
    assert np.array_equal(tall[0], tall[-1])                                    # every row the same bars: nothing invented

"""pytest tests/test_box_evidence_flip.py -q  (kotak yang ditaruh terbalik 180 derajat di rak)"""
import cv2
import numpy as np

from src.box_evidence import gather
from src.box_ocr import OcrResult, OcrToken
from src.box_verifier import MATCH, Request, decide_multi, support

NAMES = ["angiolite", "xperience pro", "essential pro", "accuforce", "ultimaster nagomi"]
CFG = {"catalog": NAMES, "ocr": {"min_confidence": 0.9},
       "matching": {"diameter_range_mm": [1.0, 10.0], "length_range_mm": [6, 60]}}
REQ = Request("angiolite", 2.5, 29.0)
CROP = np.zeros((200, 40, 3), np.uint8)                 # label ukuran tegak (punggung kotak berdiri)
ALONG, ALONG_180 = cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_90_CLOCKWISE
ACROSS, ACROSS_180 = None, cv2.ROTATE_180


class OrientReader:
    """Hasil OCR per arah putar (seperti crop asli di rekaman tripod 30-09)."""

    def __init__(self, by_rot):
        self.by_rot, self.calls = by_rot, []

    def read_orient(self, crop, rot):
        self.calls.append(rot)
        return OcrResult(tokens=[OcrToken(w, c) for w, c in self.by_rot.get(rot, [])])


# rekaman tripod frame 900 / 1500: arah biasa membaca "2,5" dan "29" yang terbalik sebagai "25" / "62 55"
UPSIDE = {ALONG: [("SCCDSR14150250029", 0.97), ("2511631", 0.99), ("29", 0.58), ("25", 0.76), ("nnite e", 0.57)],
          ACROSS: [("2", 0.83), ("22-", 0.54), ("29", 0.96), ("SCCDSR14150250029", 0.94), ("angiolite", 0.97)],
          ALONG_180: [("angiolite", 1.00), ("2.5", 0.99), ("12511631", 0.93), ("SCCDSR14150250029", 0.96)],
          ACROSS_180: [("angiolite", 0.95), ("SCCDSR14150250029", 0.97), ("2.5", 0.99), ("29", 1.00)]}


def test_upside_down_box_is_read_in_the_flipped_direction():
    ev, _ = gather([CROP], None, OrientReader(UPSIDE), CFG, REQ)
    assert ev.text.key()[1:] == (2.5, 29.0) and ev.text.product == "angiolite"
    assert decide_multi(ev, REQ)[0] == MATCH                              # teks + REF


def test_upside_down_misread_number_is_dropped_not_a_contradiction():
    # frame 1500: arah biasa "62 55", arah terbalik hanya "2.5" (+ "(R7)" sampah): tidak lengkap, TIDAK membantah
    reads = {ALONG: [("SCCDSR14150250029", 0.98), ("2028-07-31", 0.99), ("109152", 0.77), ("62", 0.88), ("55", 0.78),
                     ("nglie", 0.64)],
             ACROSS: [("1352", 0.63), ("angiolite", 0.89)],
             ALONG_180: [("angiolite", 0.97), ("2.5", 0.93), ("29", 1.0), ("2571631", 0.87), ("(R7)", 0.62),
                         ("SCCDSA14150250029", 0.98), ("2028-07-31", 0.98)],
             ACROSS_180: [("angiolite", 0.93), ("2.529", 0.91)]}
    ev, _ = gather([CROP], None, OrientReader(reads), CFG, REQ)
    atoms, contra = support(ev, REQ)
    assert not contra and ev.text.length != 55.0 and {"ref", "tp"} <= atoms


def test_upright_box_is_not_changed_by_flipped_garbage():
    reads = {ALONG: [("angiolite", 0.99), ("2.5", 0.99), ("mm", 0.99)],     # panjang belum terbaca
             ACROSS: [("2", 0.95)]}
    base, _ = gather([CROP], None, OrientReader(reads), CFG, REQ)
    garbage = {**reads, ALONG_180: [("52", 0.55), ("allo", 0.50)], ACROSS_180: [("62", 0.60)]}
    ev, _ = gather([CROP], None, OrientReader(garbage), CFG, REQ)
    assert ev.text.key() == base.text.key() and ev.text.diameter == 2.5 and ev.text.product == "angiolite"


def test_no_flipped_reading_for_other_products_or_unreadable_crops():
    other = OrientReader({ALONG: [("accuforce", 0.99), ("2.75", 0.99)], ACROSS: [("20", 0.99)]})
    gather([CROP], None, other, CFG, REQ)
    assert ALONG_180 not in other.calls and ACROSS_180 not in other.calls   # bukan produk yang diminta: hemat OCR
    empty = OrientReader({})
    gather([CROP], None, empty, CFG, REQ)
    assert empty.calls == [ALONG, ACROSS]                                    # tidak terbaca sama sekali
    scan = OrientReader({ALONG: [("accuforce", 0.99), ("2.75", 0.99)]})
    gather([CROP], None, scan, CFG, None)
    assert ALONG_180 in scan.calls                                           # pemindaian rak: semua dicoba


def test_lone_upside_down_number_is_rechecked_not_left_as_contradiction():
    # replay tripod 60,7 s: hanya "25" terbaca (= "2,5" terbalik) -> dulu panjang 25 mm, membantah 29
    reads = {ALONG: [("25", 0.95)], ALONG_180: [("2.5", 0.97), ("29", 0.98)]}
    ev, _ = gather([CROP], None, OrientReader(reads), CFG, REQ)
    assert not support(ev, REQ)[1] and ev.text.length != 25.0


# uji lapangan 01-10 (kardus): diameter bulat "4" selalu terbaca arah punggung, panjang "19" hanya arah
# melintang -> petunjuk satu-arah biasa (parse_identity) tidak pernah menyala karena dua-duanya tidak pernah
# bersebelahan dalam satu bacaan; sebelum perbaikan ini skor macet di REF saja (CANDIDATE), tidak pernah AMBIL
REQ_419 = Request("angiolite", 4.0, 19.0)
SPLIT_DIGIT = {ALONG: [("angiolite", 0.98), ("4", 0.94), ("SCCDSR14150400019", 0.90), ("noe", 0.60)],
               ACROSS: [("2020-00-21", 0.87), ("19", 0.95), ("SCCDSR14150400019", 0.92), ("nnole", 0.55)]}


def test_diameter_hint_combines_digit_and_length_read_in_different_orientations():
    ev, _ = gather([CROP], None, OrientReader(SPLIT_DIGIT), CFG, REQ_419)
    assert ev.text.diameter_hint == 4.0 and not ev.text.diameter_ok    # hanya petunjuk, bukan bacaan pasti
    assert decide_multi(ev, REQ_419)[0] == MATCH                       # teks (petunjuk + panjang) + REF


def test_diameter_hint_does_not_combine_when_two_different_digits_are_candidates():
    # "4" arah punggung DAN "6" arah melintang sekaligus: dua kandidat = ambigu, tidak boleh menebak salah satu
    reads = {ALONG: [("angiolite", 0.98), ("4", 0.94), ("SCCDSR14150400019", 0.90)],
             ACROSS: [("6", 0.93), ("19", 0.95), ("SCCDSR14150400019", 0.92)]}
    ev, _ = gather([CROP], None, OrientReader(reads), CFG, REQ_419)
    assert ev.text.diameter_hint is None
    assert decide_multi(ev, REQ_419)[0] != MATCH                       # REF saja: tidak cukup


def test_diameter_hint_does_not_combine_without_a_confident_length():
    # "19" terbaca tapi confidence di bawah ambang (belum length_ok): petunjuk lintas arah tidak dipaksakan
    reads = {ALONG: [("angiolite", 0.98), ("4", 0.94), ("SCCDSR14150400019", 0.90)],
             ACROSS: [("19", 0.5), ("SCCDSR14150400019", 0.92)]}
    ev, _ = gather([CROP], None, OrientReader(reads), CFG, REQ_419)
    assert ev.text.diameter_hint is None

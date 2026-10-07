"""pytest tests/test_box_cascade.py -q (kaskade dua model: pilih kotak kandidat, potong+geser ulang koordinat)"""
import numpy as np

from src.box_cascade import _crop_region, candidate_indices, refine_candidates
from src.box_detector import BoxGroup, Det
from src.box_verifier import CANDIDATE, READING


class FakeState:
    def __init__(self, stable="", hint=0.0):
        self._stable = stable
        self.hint = hint

    def stable(self, need):
        return self._stable


class FakeTracker:
    def __init__(self, states):
        self.states = states

    def state(self, tid):
        return self.states[tid]


class FakeSession:
    def __init__(self, pick=None):
        self.pick = pick


def box_group(x1, y1, x2, y2):
    poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    return BoxGroup(box=Det(cls="box", conf=0.9, poly=poly))


def test_candidate_indices_picks_pick_candidate_and_high_hint_only():
    groups = [box_group(0, 0, 10, 10), box_group(20, 0, 30, 10), box_group(40, 0, 50, 10), box_group(60, 0, 70, 10)]
    ids = [1, 2, 3, 4]
    tracker = FakeTracker({
        1: FakeState(stable=READING, hint=0.1),    # biasa: tidak dipilih
        2: FakeState(stable=CANDIDATE, hint=0.1),  # status CANDIDATE: dipilih
        3: FakeState(stable=READING, hint=0.9),    # mirip kotak inventaris yang dicari (hint tinggi): dipilih
        4: FakeState(stable=READING, hint=0.1),    # sedang diambil (session.pick): dipilih
    })
    session = FakeSession(pick=4)
    assert candidate_indices(groups, ids, tracker, session, need=2, hint_min=0.8) == [1, 2, 3]


def test_candidate_indices_skips_untracked_ids():
    groups = [box_group(0, 0, 10, 10)]
    assert candidate_indices(groups, [-1], FakeTracker({}), None, need=2, hint_min=0.8) == []


def test_crop_region_adds_padding_and_clamps_to_frame():
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    poly = np.array([[40, 10], [48, 10], [48, 20], [40, 20]], dtype=np.float32)  # dekat tepi kanan
    crop, x0, y0 = _crop_region(frame, poly, pad_frac=0.5)
    assert x0 < 40 and y0 < 10              # konteks ditambahkan ke kiri/atas
    assert x0 + crop.shape[1] <= 50         # tidak melewati tepi kanan frame
    assert y0 + crop.shape[0] <= 50


def test_refine_candidates_only_calls_big_model_on_candidate_boxes():
    groups = [box_group(0, 0, 10, 10), box_group(20, 0, 30, 10)]
    ids = [1, 2]
    tracker = FakeTracker({1: FakeState(stable=READING, hint=0.0), 2: FakeState(stable=CANDIDATE, hint=0.0)})
    frame = np.zeros((40, 80, 3), dtype=np.uint8)
    calls = []

    class FakeDetBig:
        def detect(self, crop):  # TODO(human) di box_cascade.refine_group belum diisi -> selalu [] (tidak dipakai)
            calls.append(crop.shape)
            return [], 0.0

    out = refine_candidates(frame, groups, ids, tracker, FakeDetBig(), session=None, need=2, hint_min=0.8)
    assert len(calls) == 1          # hanya kotak CANDIDATE (indeks 1) yang dipanggil ulang dengan model besar
    assert out[0] is groups[0]      # kotak non-kandidat tidak disentuh sama sekali

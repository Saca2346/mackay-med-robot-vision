"""pytest tests/test_webcam_apply.py -q  (jalur apply_reading di scripts/test_webcam_box.py dengan bukti gabungan)"""
import importlib.util

from src.box_ref import parse_refs
from src.box_verifier import CANDIDATE, MATCH, Evidence, Request, TrackState, parse_identity, support
from src.box_ocr import OcrToken

REQ = Request("accuforce", 2.75, 20)


def load():
    spec = importlib.util.spec_from_file_location("test_webcam_box", "scripts/test_webcam_box.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_apply_reading_combines_and_writes_reason():
    m = load()
    st = TrackState()
    ev1 = Evidence(text=parse_identity([OcrToken(t, 0.99) for t in ("Accuforce", "2.75", "20", "mm")], ["accuforce"]))
    ev1.support = support(ev1, REQ)
    m.apply_reading(st, ev1, CANDIDATE, "baru teks cocok", ("k",), 100.0, 1)
    ev2 = Evidence(refs=parse_refs("DC-RM2720HHW"))
    ev2.support = support(ev2, REQ)
    m.apply_reading(st, ev2, CANDIDATE, "baru REF cocok", ("k",), 100.0, 1)
    assert st.stable(2) == MATCH
    assert st.last_reason.startswith("combined 2 readings") and "text" in st.last_reason and "REF" in st.last_reason


def test_reading_log_keeps_raw_ocr_text_for_misread_analysis():
    import io
    m = load()
    m.READ_LOG = log = io.StringIO()
    try:
        ev = Evidence(text=parse_identity([OcrToken(t, 0.99) for t in ("Angiolite", "5", "mm")], ["angiolite"]),
                      refs=parse_refs("SCCDSR14150400010"), spine='REF SCCDSR14150400010 "LOT"')
        m.apply_reading(TrackState(), ev, CANDIDATE, "", ("k",), 100.0, 1)
    finally:
        m.READ_LOG = None
    row = log.getvalue().strip()
    assert row.endswith('"Angiolite 5 mm || SCCDSR14150400010 || REF SCCDSR14150400010 \'LOT\'"')
    assert "angiolite:4x10" in row


def test_process_logs_why_box_numbers_are_doubted_and_skips_wrong_motion():
    import io
    import numpy as np
    from src.box_detector import Det
    from src.box_verifier import SimpleTracker
    twb = load()

    def spine(x):
        return np.float32([[x, 100], [x + 20, 100], [x + 20, 400], [x, 400]])

    class FakeDet:
        xs = [100 + 30 * i for i in range(8)]

        def detect(self, frame):
            return [Det("box", 0.9, spine(x)) for x in self.xs], 1.0

    class FakeMotion:
        fails = 0
        m = np.float32([[1, 0, 0], [0, 1, 0]])

        def update(self, frame):
            return self.m.copy(), float(abs(self.m[0, 2]))

        def reject(self):
            self.fails += 1

    cfg = {"grouping": {}, "ocr": {"max_motion_px": 6}}
    frame = np.zeros((720, 1280, 3), np.uint8)
    det, tr, me, log = FakeDet(), SimpleTracker(), FakeMotion(), io.StringIO()
    twb.READ_LOG = log
    try:
        twb.process(frame, det, None, tr, None, cfg, 0, motion_est=me)
        for tid in list(tr.tracks):
            tr.state(tid).history.append((MATCH, "k"))
        me.m = np.float32([[1, 0, 15], [0, 1, 0]])       # tangan di depan kamera diam: flow bilang geser 15 px
        *_, ids, _, _ = twb.process(frame, det, None, tr, None, cfg, 1, motion_est=me)
        assert ids == [-1] * 8 and me.fails == 1           # frame dilewati, nomor kotak TIDAK diragukan
        assert not any(tr.state(t).stale for t in tr.tracks)
        me.m = np.float32([[1, 0, 0], [0, 1, 0]])
        det.xs = [x + 9 for x in det.xs]                   # semua kotak tergeser 9 px (IoU < 0,5)
        twb.process(frame, det, None, tr, None, cfg, 2, motion_est=me)
        rows = [r for r in log.getvalue().splitlines() if ",stale," in r]
        assert len(rows) == 8 and "weak pair" in rows[0] and "8 boxes" in rows[0]
    finally:
        twb.READ_LOG = None


def test_rack_area_drops_objects_outside_the_rack():
    # uji 30-09: kotak kertas dan kotak GPU di samping rak ikut terdeteksi -> "24 kotak di rak" padahal 20
    import numpy as np
    import pytest
    from src.box_detector import Det
    from src.box_verifier import SimpleTracker
    twb = load()
    assert twb.load_roi("none", True) is None and twb.load_roi("0.05,0,0.6,0.75", False) == (0.05, 0.0, 0.6, 0.75)
    with pytest.raises(SystemExit):
        twb.load_roi("0.6,0,0.1,1", False)
    assert twb.load_roi(None, False) is None                                  # rekaman lama: area tersimpan tidak dipakai

    def spine(x, y=100):
        return np.float32([[x, y], [x + 40, y], [x + 40, y + 600], [x, y + 600]])

    class FakeDet:
        def detect(self, frame):
            return [Det("box", 0.9, spine(100 + 60 * k)) for k in range(5)] + [Det("box", 0.9, spine(1800, 400))], 1.0

    frame = np.zeros((1080, 1920, 3), np.uint8)
    cfg = {"grouping": {}, "ocr": {"max_motion_px": 6}}
    *_, ids, _, _ = twb.process(frame, FakeDet(), None, SimpleTracker(), None, cfg, 0)
    assert len(ids) == 6
    *_, ids, _, _ = twb.process(frame, FakeDet(), None, SimpleTracker(), None, cfg, 0, roi=(0.0, 0.0, 0.6, 0.75))
    assert len(ids) == 5                                                      # benda di luar area rak diabaikan


def _group(x, width, length=400):
    from types import SimpleNamespace
    poly = [(x, 50), (x + width, 50), (x + width, 50 + length), (x, 50 + length)]
    return SimpleNamespace(box=SimpleNamespace(poly=poly))


def test_drop_wide_removes_monitor_but_keeps_cut_box():
    m = load()
    spines = [_group(400 + 40 * i, 35) for i in range(6)]
    cut = _group(1240, 35, length=90)                       # box cut by the frame edge: short, same width
    monitor = _group(0, 330, length=240)                    # D455 08-10: monitor behind the rack
    kept = m.drop_wide(spines + [cut, monitor])
    assert monitor not in kept and cut in kept and len(kept) == 7


def test_printed_expiry_steers_fefo_only_after_two_reads():
    import datetime as dt
    m = load()
    st = TrackState()
    day = dt.date(2029, 3, 10)
    m.apply_reading(st, Evidence(dates=(day,)), CANDIDATE, "", ("k",), 1.0, 1)
    assert st.expiry is None                                # one read is not enough
    m.apply_reading(st, Evidence(dates=(day,)), CANDIDATE, "", ("k",), 1.0, 1)
    assert st.expiry == day and st.expiry_src == "text"
    m.apply_reading(st, Evidence(dates=(dt.date(2029, 3, 12),)), CANDIDATE, "", ("k",), 1.0, 1)
    assert st.expiry is None                                # K03 vs K17 conflict -> no date


def test_roi_cut_warns_when_area_slices_a_box():
    m = load()
    boxes = [_group(400 + 40 * i, 35) for i in range(5)]   # x 400..595, width 35
    w, h = 1280, 800
    assert m.roi_cut(boxes, None, w, h) == 0
    assert m.roi_cut(boxes, (0.25, 0, 1, 1), w, h) == 0       # margin left of the first box
    assert m.roi_cut(boxes, (412 / w, 0, 1, 1), w, h) == 1     # D455 08-10: edge through K15

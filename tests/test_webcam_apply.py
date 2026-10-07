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
    assert st.last_reason.startswith("gabungan 2 bacaan") and "teks" in st.last_reason and "REF" in st.last_reason


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
        assert len(rows) == 8 and "pasangan lemah" in rows[0] and "8 kotak" in rows[0]
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

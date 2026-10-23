"""pytest tests/test_box_detector_input.py -q  (input size of .pt models: square by default, fixed [h, w] on request)"""
import ultralytics

from src.box_detector import BoxDetector


class _FakeYOLO:
    calls = []

    def __init__(self, weights, task=None):
        pass

    def predict(self, img, **kw):
        _FakeYOLO.calls.append(kw)
        return []


def _make(monkeypatch, **kw):
    _FakeYOLO.calls = []
    monkeypatch.setattr(ultralytics, "YOLO", _FakeYOLO)
    return BoxDetector("m.pt", "obb", [544, 960], "cpu", **kw)


def test_pt_model_gets_a_square_input_by_default(monkeypatch):
    det = _make(monkeypatch)
    assert det.imgsz == 960 and det.rect is True
    assert _FakeYOLO.calls[-1]["rect"] is True


def test_pt_model_keeps_the_fixed_laptop_input_on_request(monkeypatch):
    # 17-10: only the full 544 x 960 letterbox gives a .pt model the same boxes as the laptop's OpenVINO export
    det = _make(monkeypatch, fixed_input=True)
    assert det.imgsz == [544, 960] and det.rect is False
    assert _FakeYOLO.calls[-1]["imgsz"] == [544, 960] and _FakeYOLO.calls[-1]["rect"] is False

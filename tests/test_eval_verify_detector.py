"""A three-class model must remain usable for the identity safety evaluation."""
import sys
from types import SimpleNamespace

import numpy as np

from scripts.eval_verify_detector import predict


def test_predict_ignores_expiry_and_keeps_identity_thresholds(monkeypatch, tmp_path):
    class Tensor:
        def __init__(self, data):
            self.data = np.asarray(data)

        def cpu(self):
            return self

        def numpy(self):
            return self.data

    obb = SimpleNamespace(
        xyxyxyxyn=Tensor([[[.1, .1], [.8, .1], [.8, .2], [.1, .2]]] * 5),
        cls=Tensor([0, 1, 2, 0, 1]),
        conf=Tensor([.8, .8, .99, .34, .44]),
    )

    class Model:
        def __init__(self, weights):
            pass

        def predict(self, *args, **kwargs):
            return [SimpleNamespace(obb=obb)]

    monkeypatch.setitem(sys.modules, 'ultralytics', SimpleNamespace(YOLO=Model))
    args = SimpleNamespace(conf_box=.35, conf_sl=.45, imgsz=1280, nms_iou=.5, device='cpu')
    predict({'photo': 'three-class.pt'}, tmp_path, args, tmp_path)
    rows = (tmp_path / 'photo.txt').read_text().splitlines()
    assert [row.split()[0] for row in rows] == ['0', '1']
    assert all(len(row.split()) == 9 for row in rows)

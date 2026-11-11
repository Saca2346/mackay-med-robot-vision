"""Calibration evidence must preserve the source frame and show both carton ends."""
import hashlib
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

from scripts import kalibrasi_kamera as calibration
from src.box_preview import fit_preview


def test_calibration_header_is_outside_image_and_unverified_outlines_are_blue():
    frame = np.full((240, 320, 3), 91, np.uint8)
    poly = np.float32([[100, 2], [145, 2], [145, 238], [100, 238]])
    result = calibration.draw(frame, [poly], [], 'Position check')
    offset = result.shape[0] - frame.shape[0]
    np.testing.assert_array_equal(result[offset:, 160:], frame[:, 160:])
    np.testing.assert_array_equal(result[offset:offset+20, :90], frame[:20, :90])
    assert np.any(result[offset+2:offset+10, 110:130, 0] > result[offset+2:offset+10, 110:130, 2])
    assert np.any(result[offset+230:offset+239, 110:130, 0] > result[offset+230:offset+239, 110:130, 2])
    assert np.all(frame == 91)


def test_preview_keeps_all_four_image_corners_without_stretching():
    frame = np.full((1080, 1920, 3), 90, np.uint8)
    frame[:30, :30] = (255, 0, 0)
    frame[:30, -30:] = (0, 255, 0)
    frame[-30:, :30] = (0, 0, 255)
    frame[-30:, -30:] = (255, 255, 255)
    preview = fit_preview(frame)
    assert preview.shape == (720, 1280, 3)
    for y, x in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
        np.testing.assert_array_equal(preview[y, x], frame[y, x])


@pytest.mark.parametrize('single', [False, True])
def test_calibration_saves_raw_frame_geometry_and_actual_camera_provenance(tmp_path, monkeypatch, single):
    frame = np.random.default_rng(4).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    ps = [np.float32([[x, 50], [x+45, 50], [x+45, 400], [x, 400]]) for x in (160, 220, 280)]
    detections = [SimpleNamespace(cls='box', poly=p) for p in ps]
    model = {'weights': 'unused.pt', 'task': 'obb', 'imgsz': [544, 960], 'device': 'cpu',
             'conf': {'box': .35}, 'iou': .5, 'fixed_input': True}
    settings = {'index': 1, 'backend': 'dshow', 'width': 1920, 'height': 1080, 'fps': 20}
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump({'model': model, 'camera': settings, 'classes': {'box': 'box'}}))
    seen = {}
    def detector(*args, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(detect=lambda f: (detections, 1.0))
    monkeypatch.setattr(calibration, 'BoxDetector', detector)
    monkeypatch.setattr(calibration, 'open_camera', lambda *a: (object(), 1, (640, 480)))
    monkeypatch.setattr(calibration, 'Grabber', lambda cam: SimpleNamespace(stop=lambda: None))
    monkeypatch.setattr(calibration, 'adjust', lambda *a: True)
    monkeypatch.setattr(calibration, 'countdown', lambda *a: True)
    monkeypatch.setattr(calibration, 'collect', lambda *a: [frame] * 4)
    argv = ['kalibrasi_kamera.py', '--sesi', 'offline_fixture', '--config', str(config),
            '--out', str(tmp_path), '--no-window']
    if single:
        argv += ['--min-boxes', '1', '--roi', '0.23,0.05,0.35,0.9']
    monkeypatch.setattr('sys.argv', argv)
    assert calibration.main() == 0
    doc = json.loads((tmp_path / 'offline_fixture_kalibrasi.json').read_text())
    raw = tmp_path / doc['images']['raw']
    np.testing.assert_array_equal(cv2.imread(str(raw)), frame)
    assert doc['images']['raw_sha256'] == hashlib.sha256(raw.read_bytes()).hexdigest()
    assert doc['capture']['actual_size'] == [640, 480]
    assert doc['capture']['requested'] == settings
    assert doc['capture']['opened_index'] == 1
    assert any(c['check'] == 'capture resolution' and c['status'] == 'WARN' for c in doc['checks'])
    assert len(doc['geometry']['detector']) == (1 if single else 3)
    assert doc['screening']['min_boxes'] == (1 if single else 3)
    assert doc['screening']['roi'] == ([.23,.05,.35,.9] if single else None)
    assert doc['config']['sha256'] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert seen['fixed_input'] is True
    assert cv2.imread(str(tmp_path / doc['images']['annotated'])).shape[0] > 480


def test_existing_calibration_is_preserved_before_opening_camera(tmp_path, monkeypatch):
    previous = tmp_path / 'old_kalibrasi.json'
    previous.write_text('previous evidence')
    monkeypatch.setattr('sys.argv', ['kalibrasi_kamera.py', '--sesi', 'old', '--out', str(tmp_path)])
    monkeypatch.setattr(calibration, 'open_camera', lambda *a: pytest.fail('camera must not open'))
    with pytest.raises(SystemExit) as caught:
        calibration.main()
    assert caught.value.code == 2
    assert previous.read_text() == 'previous evidence'

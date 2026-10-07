"""pytest tests/test_box_camera.py -q  (tanpa kamera: VideoCapture diganti kamera palsu)"""
import numpy as np

import src.box_camera as bc


class FakeCap:
    def __init__(self, script):
        self.script = script          # daftar hasil read(): True = frame, False = gagal

    def isOpened(self):
        return True

    def read(self):
        ok = self.script.pop(0) if self.script else True
        return (True, np.zeros((1080, 1920, 3), np.uint8)) if ok else (False, None)

    def release(self):
        pass


def test_reconnects_after_msmf_failure(monkeypatch):
    opened = []
    scripts = [[True, False], [True]]           # kamera pertama gagal di bacaan kedua, sambungan ulang sehat

    def fake_open(index, w, h, fps, backend, fourcc):
        opened.append(backend)
        return FakeCap(scripts.pop(0) if scripts else [])

    monkeypatch.setattr(bc, "_open", fake_open)
    monkeypatch.setattr(bc.time, "sleep", lambda s: None)
    cam = bc.ReconnectingCamera(0, {"backend": "msmf"})
    assert cam.read()[0] and cam.read()[0]      # bacaan kedua gagal lalu tersambung ulang otomatis
    assert cam.reconnects == 1 and opened == ["msmf", "msmf"]


def test_falls_back_to_dshow_then_gives_up(monkeypatch):
    opened = []

    def fake_open(index, w, h, fps, backend, fourcc):
        opened.append(backend)
        return FakeCap([False] * 10)

    monkeypatch.setattr(bc, "_open", fake_open)
    monkeypatch.setattr(bc.time, "sleep", lambda s: None)
    cam = bc.ReconnectingCamera(0, {"backend": "msmf"}, retries=4)
    ok, frame = cam.read()
    assert not ok and frame is None
    assert opened == ["msmf", "msmf", "msmf", "dshow", "dshow"]


def test_recorder_keeps_real_frame_times(tmp_path):
    from src.box_stream import VideoRecorder, load_times
    rec = VideoRecorder(tmp_path / "uji.mp4", fps=30)
    for i in range(12):
        rec.write(np.full((120, 160, 3), i * 20, np.uint8), t=100.0 + i * 0.1)   # kamera nyata: 10 fps, bukan 30
    rec.close()
    times = load_times(tmp_path / "uji.mp4")
    assert rec.n == 12 and len(times) == 12 and abs(times[-1] - 1.1) < 1e-6
    assert abs(rec.real_fps - 10.0) < 0.01
    import cv2
    cap = cv2.VideoCapture(str(tmp_path / "uji.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 12


def test_sharpness_higher_for_sharp_image():
    rng = np.random.default_rng(0)
    sharp = (rng.random((480, 640, 3)) * 255).astype(np.uint8)
    import cv2
    blurred = cv2.GaussianBlur(sharp, (0, 0), 3)
    assert bc.sharpness(sharp) > 10 * bc.sharpness(blurred)

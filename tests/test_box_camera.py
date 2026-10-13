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


def fake_realsense(monkeypatch, script, supported=None):
    """pyrealsense2 palsu: script = daftar hasil wait_for_frames (True = frame warna, False = RuntimeError).
    supported = {(lebar, tinggi, fps)} yang ditawarkan kamera; start() di luar itu -> "Couldn't resolve requests"."""
    import sys
    import types

    log = {"starts": 0, "stops": 0, "streams": [], "serial": None}

    class Color:
        def __bool__(self):
            return True

        def get_data(self):
            return np.zeros((1080, 1920, 3), np.uint8)

    class Frames:
        def get_color_frame(self):
            return Color()

    class Config:
        def enable_device(self, serial):
            log["serial"] = serial

        def enable_stream(self, *a):
            log["streams"].append(a)
            self.last = a

    class Pipeline:
        def start(self, cfg):
            _, w, h, _, fps = cfg.last
            if supported is not None and (w, h, fps) not in supported:
                raise RuntimeError("Couldn't resolve requests")
            log["starts"] += 1

        def stop(self):
            log["stops"] += 1

        def wait_for_frames(self, timeout=5000):
            if script and not script.pop(0):
                raise RuntimeError("Frame didn't arrive within 5000")
            return Frames()

    class Profile:
        def __init__(self, w, h, fps, kind="color"):
            self.w, self.h, self._fps, self.kind = w, h, fps, kind

        def stream_type(self):
            return self.kind

        def as_video_stream_profile(self):
            return types.SimpleNamespace(width=lambda: self.w, height=lambda: self.h)

        def fps(self):
            return self._fps

    class Sensor:
        def get_stream_profiles(self):
            return [Profile(w, h, f) for w, h, f in sorted(supported or [])] + [Profile(640, 480, 30, "depth")]

    class Device:
        sensors = [Sensor()]

        def get_info(self, key):
            return "SERIAL1"

    rs = types.SimpleNamespace(
        config=Config, pipeline=Pipeline, context=lambda: types.SimpleNamespace(devices=[Device()]),
        stream=types.SimpleNamespace(color="color"), format=types.SimpleNamespace(bgr8="bgr8"),
        camera_info=types.SimpleNamespace(serial_number="serial"))
    monkeypatch.setitem(sys.modules, "pyrealsense2", rs)
    return log


def test_realsense_opens_color_stream_only_and_reads_bgr(monkeypatch):
    log = fake_realsense(monkeypatch, [])
    cam, idx, size = bc.open_camera({"backend": "realsense", "width": 1920, "height": 1080, "fps": 30})
    assert idx == "realsense" and size == (1920, 1080)
    assert log["streams"] == [("color", 1920, 1080, "bgr8", 30)]       # hanya warna, tanpa depth / inframerah
    ok, frame = cam.read()
    assert ok and frame.shape == (1080, 1920, 3)
    cam.release()
    assert not cam.isOpened() and log["stops"] == 1


def test_realsense_falls_back_to_nearest_supported_color_profile(monkeypatch, capsys):
    # model tanpa warna 1080p (mis. D455 maks 1280x800) atau terpasang di USB 2: "Couldn't resolve requests"
    log = fake_realsense(monkeypatch, [], supported={(1280, 800, 30), (1280, 800, 15), (848, 480, 60), (640, 480, 30)})
    cam, idx, size = bc.open_camera({"backend": "realsense", "width": 1920, "height": 1080, "fps": 30})
    assert cam is not None and log["starts"] == 1
    assert log["streams"][-1] == ("color", 1280, 800, "bgr8", 30)    # piksel terbanyak yang tidak melebihi permintaan
    assert "has no colour 1920x1080" in capsys.readouterr().out


def test_realsense_d455_on_usb2_prefers_15fps_over_8fps(monkeypatch):
    # profil warna D455 yang sebenarnya terbaca di USB 2.1 (hasil uji 08-10)
    usb2 = {(424, 240, f) for f in (5, 15, 30, 60)} | {(640, 480, f) for f in (5, 15, 30)} \
        | {(1280, 720, f) for f in (5, 10, 15)} | {(1280, 800, 8)}
    log = fake_realsense(monkeypatch, [], supported=usb2)
    cam, idx, size = bc.open_camera({"backend": "realsense", "width": 1920, "height": 1080, "fps": 30})
    assert cam is not None and log["streams"][-1] == ("color", 1280, 720, "bgr8", 15)


def test_realsense_no_usable_profile_lists_what_the_camera_offers(monkeypatch):
    fake_realsense(monkeypatch, [], supported={(2560, 1440, 30)})   # semua di atas permintaan: tidak ada yang cocok
    try:
        bc.RealSenseCamera({"width": 1920, "height": 1080, "fps": 30})
    except RuntimeError as e:
        assert "2560" in str(e) and "USB 3" in str(e)
    else:
        raise AssertionError("harus gagal dengan daftar profil yang ditawarkan kamera")


def test_realsense_serial_selects_device(monkeypatch):
    log = fake_realsense(monkeypatch, [])
    bc.RealSenseCamera({"serial": "123456789"})
    assert log["serial"] == "123456789"


def test_realsense_reconnects_after_dropped_frame(monkeypatch):
    log = fake_realsense(monkeypatch, [True, False])                  # bacaan kedua gagal (kabel USB goyang)
    monkeypatch.setattr(bc.time, "sleep", lambda s: None)
    cam = bc.RealSenseCamera({})
    assert cam.read()[0] and cam.read()[0]
    assert cam.reconnects == 1 and log["starts"] == 2


def test_realsense_missing_sdk_gives_clear_error(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "pyrealsense2", None)            # import -> ImportError
    try:
        bc.RealSenseCamera({})
    except RuntimeError as e:
        assert "pip install pyrealsense2" in str(e)
    else:
        raise AssertionError("harus gagal dengan pesan yang jelas")


def test_open_camera_realsense_failure_returns_none_not_traceback(monkeypatch, capsys):
    import sys
    monkeypatch.setitem(sys.modules, "pyrealsense2", None)
    assert bc.open_camera({"backend": "realsense"}) == (None, "realsense", None)
    assert "cannot open the RealSense" in capsys.readouterr().out


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

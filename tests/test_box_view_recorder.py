"""Video evidence must survive async encoding, caller mutation and write errors."""
import threading

import cv2
import numpy as np
import pytest

from src.box_view_recorder import ViewRecorder


def test_real_recording_flushes_every_frame_and_keeps_measured_times(tmp_path):
    path = tmp_path / "view.mp4"
    recorder = ViewRecorder(path, 6)
    for i in range(8):
        recorder.write(np.full((80, 120, 3), i * 25, np.uint8), i * .21)
    recorder.close()
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    assert len(frames) == 8
    assert frames[-1].mean() > frames[0].mean() + 150
    times = path.with_name("view.mp4.times.csv").read_text().splitlines()
    assert len(times) == 9 and times[-1] == "7,1.470000"


def test_encoder_receives_frozen_pixels(tmp_path, monkeypatch):
    release = threading.Event()
    entered = threading.Event()
    captured = []
    def encode(self, frame, seconds):
        entered.set()
        assert release.wait(5)
        captured.append(frame)
    monkeypatch.setattr(ViewRecorder, "_write", encode)
    recorder = ViewRecorder(tmp_path / "view.mp4", 6)
    frame = np.full((10, 10, 3), 7, np.uint8)
    try:
        recorder.write(frame, 0)
        assert entered.wait(5)
        frame[:] = 99
    finally:
        release.set()
        recorder.close()
    assert (captured[0] == 7).all()


def test_failed_video_open_fails_session(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(cv2, "VideoWriter", lambda *a: SimpleNamespace(isOpened=lambda: False, release=lambda: None))
    recorder = ViewRecorder(tmp_path / "view.mp4", 6)
    recorder.write(np.zeros((10, 10, 3), np.uint8), 0)
    with pytest.raises(OSError, match="annotated video"):
        recorder.close()


def test_only_video_is_resized_while_original_input_stays_full_resolution(tmp_path):
    path = tmp_path / "small.mp4"
    recorder = ViewRecorder(path, 6, max_width=120)
    frame = np.full((160, 240, 3), 180, np.uint8)
    recorder.write(frame, 0)
    recorder.close()
    cap = cv2.VideoCapture(str(path))
    ok, output = cap.read()
    cap.release()
    assert ok and output.shape == (80, 120, 3)
    assert frame.shape == (160, 240, 3) and (frame == 180).all()

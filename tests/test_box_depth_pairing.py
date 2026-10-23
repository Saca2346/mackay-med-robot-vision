"""A slow detector or PICK panel must not pair old colour with newer depth."""
import json
import queue

import numpy as np

from src.box_camera import RealSenseCamera
from src.box_detector import BoxGroup, Det
from src.box_stream import LatestFrame


def test_realsense_does_not_substitute_latest_depth_for_an_unknown_frame():
    cam = RealSenseCamera.__new__(RealSenseCamera)
    frame = np.zeros((10, 10, 3), np.uint8)
    depth = np.full((10, 10), 0.6, np.float32)
    cam._depth_ring = [(frame, depth)]
    assert cam.depth_for(frame) is depth
    assert cam.depth_for(frame.copy()) is None
    cam._depth_ring = [(frame.copy(), depth + 0.2)]
    assert cam.depth_for(frame) is None


def test_realsense_release_discards_previous_connection_depth():
    cam = RealSenseCamera.__new__(RealSenseCamera)
    frame = np.zeros((10, 10, 3), np.uint8)
    cam._depth_ring = [(frame, np.ones((10, 10), np.float32))]
    cam.pipeline, cam.align, cam.intrinsics = None, object(), {"fx": 100.0}
    cam.release()
    assert cam.depth_for(frame) is None
    assert cam.intrinsics is None and cam.align is None


class PairedCamera:
    """Only the current depth is available, as after eviction from the camera ring."""
    def __init__(self):
        self.items = queue.Queue()
        self.frame = self.depth = self.intrinsics = None

    def read(self):
        item = self.items.get(timeout=5)
        if item is None:
            return False, None
        self.frame, self.depth, self.intrinsics = item
        return True, self.frame

    def depth_for(self, frame):
        return self.depth if frame is self.frame else None

    def release(self):
        self.frame = self.depth = self.intrinsics = None


def test_stream_pins_depth_and_intrinsics_through_later_frames_and_grasp_output(tmp_path):
    from scripts.test_webcam_box import write_grasp

    source = PairedCamera()
    frame = np.zeros((100, 80, 3), np.uint8)
    depth = np.full((100, 80), 0.6, np.float32)
    intr = {"fx": 100.0, "fy": 100.0, "cx": 40.0, "cy": 50.0, "width": 80, "height": 100}
    source.items.put((frame, depth, intr))
    live = LatestFrame(source)
    try:
        assert live.get(timeout=2) is frame
        intr["fx"] = 200.0  # A reconnect may replace or mutate the camera calibration.
        for _ in range(5):
            source.items.put((frame.copy(), depth + 0.3, intr))
        source.items.put(None)
        with live._cond:
            assert live._cond.wait_for(lambda: live.ended, timeout=2)
        assert source.depth_for(frame) is None
        assert live.depth_for(frame) is depth
        assert live.depth_for(frame.copy()) is None
        assert live.intrinsics_for(frame)["fx"] == 100.0
        assert live.intrinsics_for(frame.copy()) is None

        poly = np.array([[20, 10], [60, 10], [60, 90], [20, 90]], np.float32)
        groups = [BoxGroup(Det("box", 0.99, poly))]
        output = tmp_path / "grasp.jsonl"
        barcode = [[25, 30], [55, 30], [55, 50], [25, 50]]
        write_grasp(output, "accuforce 2.5 x 12", 7, [7], groups, frame, live,
                    {"barcode_px": barcode, "barcode_decoded": False})
        row = json.loads(output.read_text(encoding="utf-8"))
        assert row["xyz_m"] == [0.0, 0.0, 0.6]
        assert row["intrinsics"]["fx"] == 100.0
        assert row["spine_width_mm"] == 240.0
        assert row["no_grip_px"] == [barcode]
        assert row["barcode_decoded"] is False
        assert row["barcode_xyz_m"][2] == 0.6
    finally:
        source.items.put(None)
        live.stop()


def test_stream_missing_depth_is_not_replaced_by_a_later_valid_pair():
    source = PairedCamera()
    frame = np.zeros((10, 10, 3), np.uint8)
    source.items.put((frame, None, {"fx": 100.0}))
    live = LatestFrame(source)
    try:
        assert live.get(timeout=2) is frame
        source.items.put((frame.copy(), np.ones((10, 10)), {"fx": 200.0}))
        source.items.put(None)
        with live._cond:
            assert live._cond.wait_for(lambda: live.ended, timeout=2)
        assert live.depth_for(frame) is None
        assert live.intrinsics_for(frame)["fx"] == 100.0
    finally:
        source.items.put(None)
        live.stop()

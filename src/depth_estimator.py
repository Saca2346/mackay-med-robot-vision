"""
Stage 3 of the architecture (design doc sections 2, 6, 8): 3D Position Estimation.

Pure geometry: depth value at the detection's pixel + camera intrinsics -> 3D point,
exactly as the design doc specifies (no learned monocular-depth model).

Two backends:
  - RealSenseDepth: real depth via pyrealsense2 on the actual D435i/D415 (laptop only).
  - MockDepth: synthesizes a plausible depth in meters from bounding-box size, so the
    rest of the pipeline (grasp point math, 3D logging) can be developed/tested here
    without RGB-D hardware. Swap to "realsense" in config once the camera is connected.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Point3D:
    x: float
    y: float
    z: float  # meters, camera frame


class DepthBackend:
    def get_depth_frame(self):
        raise NotImplementedError

    def pixel_to_3d(self, u: int, v: int, depth_frame) -> Point3D | None:
        raise NotImplementedError

    def release(self):
        pass


class MockDepth(DepthBackend):
    """No hardware needed. Approximates distance from apparent object size:
    smaller box in frame -> assumed farther away. Good enough to validate the
    geometry/plumbing; replace with RealSenseDepth for real measurements."""

    def __init__(self, cfg: dict, assumed_object_width_m: float = 0.06):
        self.fx = cfg["fx"]
        self.fy = cfg["fy"]
        self.cx = cfg["cx"]
        self.cy = cfg["cy"]
        self.assumed_width_m = assumed_object_width_m

    def get_depth_frame(self):
        return None  # unused in mock mode; box size stands in for depth

    def estimate_from_box(self, box: tuple, frame_shape) -> Point3D:
        x1, y1, x2, y2 = box
        box_w_px = max(1.0, x2 - x1)
        cx_px, cy_px = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        # similar-triangles: Z = (real_width * focal_length) / pixel_width
        z = (self.assumed_width_m * self.fx) / box_w_px
        z = float(np.clip(z, 0.10, 1.20))  # clamp to a plausible shelf-reach range (10cm-1.2m)
        x = (cx_px - self.cx) * z / self.fx
        y = (cy_px - self.cy) * z / self.fy
        return Point3D(x=float(x), y=float(y), z=z)

    def pixel_to_3d(self, u, v, depth_frame):
        raise NotImplementedError("Use estimate_from_box() for MockDepth")


class RealSenseDepth(DepthBackend):
    """Real RGB-D depth via Intel RealSense (D435i / D415). Requires pyrealsense2
    and an actual camera — only usable on the Lenovo laptop with the sensor attached."""

    def __init__(self, cfg: dict):
        try:
            import pyrealsense2 as rs
        except ImportError as e:
            raise RuntimeError(
                "pyrealsense2 tidak terpasang atau tidak ada kamera RealSense. "
                "Set depth.source: mock di config untuk pengembangan tanpa hardware."
            ) from e
        self.rs = rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        profile = self.pipeline.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        self.align = rs.align(rs.stream.color)
        intr = (
            profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        self.fx, self.fy, self.cx, self.cy = intr.fx, intr.fy, intr.ppx, intr.ppy

    def get_depth_frame(self):
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        return depth_frame, color_frame

    def pixel_to_3d(self, u: int, v: int, depth_frame) -> Point3D | None:
        depth_m = depth_frame.get_distance(int(u), int(v))
        if depth_m <= 0:
            return None
        x = (u - self.cx) * depth_m / self.fx
        y = (v - self.cy) * depth_m / self.fy
        return Point3D(x=float(x), y=float(y), z=float(depth_m))

    def release(self):
        self.pipeline.stop()


def build_depth_backend(cfg: dict) -> DepthBackend:
    if cfg.get("source", "mock") == "realsense":
        return RealSenseDepth(cfg)
    return MockDepth(cfg)


def grasp_point_from_detection(box: tuple, contour=None) -> tuple:
    """Bounding-box-plus-contour grasp point (design doc section 8): prefer the
    contour centroid when available, else the bounding-box center."""
    x1, y1, x2, y2 = box
    if contour is not None and len(contour) > 0:
        M = __import__("cv2").moments(contour)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            return (cx, cy)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

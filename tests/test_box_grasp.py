"""pytest tests/test_box_grasp.py -q  (grasp data of the picked box, src/box_grasp.py)"""
import math

import numpy as np

from src.box_grasp import grasp_info

INTR = {"fx": 640.0, "fy": 640.0, "cx": 640.0, "cy": 400.0}
H, W = 800, 1280


def rect(x0, y0, w, h):
    return [[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]]


def test_flat_spine_facing_the_camera():
    # wall 0.6 m away; spine 40 px wide -> 40 * 0.6 / 640 = 37.5 mm; neighbours 10 px left, touching right
    depth = np.full((H, W), 0.6, np.float32)
    target = rect(620, 200, 40, 400)
    g = grasp_info(target, depth, INTR, neighbours=[rect(530, 200, 80, 400), rect(660, 220, 40, 380)],
                   no_grip=[rect(625, 300, 30, 100)])
    assert g["xyz_m"] == [0.0, 0.0, 0.6]
    assert abs(g["spine_width_mm"] - 37.5) < 0.5 and abs(g["spine_length_mm"] - 375) < 1
    assert abs(g["gap_left_mm"] - 10 * 0.6 / 640 * 1000) < 0.5 and g["gap_right_mm"] == 0.0
    assert g["normal"][2] < -0.99 and g["approach"][2] > 0.99          # face looks at the camera, approach forward
    assert g["depth_valid"] == 1.0 and len(g["no_grip_px"]) == 1 and abs(g["angle_deg"]) < 0.5


def test_tilted_face_gives_a_tilted_normal():
    # face turned 30 deg around the vertical axis: depth grows to the right
    xs = np.arange(W, dtype=np.float32)
    depth = np.tile(0.6 + (xs - 640) / 640 * 0.6 * math.tan(math.radians(30)), (H, 1)).astype(np.float32)
    g = grasp_info(rect(600, 200, 80, 400), depth, INTR)
    angle = math.degrees(math.atan2(-g["normal"][0], -g["normal"][2]))
    assert abs(abs(angle) - 30) < 2


def test_no_depth_keeps_pixels_and_leaves_metric_fields_empty():
    g = grasp_info(rect(620, 200, 40, 400))
    assert g["center_px"] == [640.0, 400.0] and g["spine_width_px"] == 40.0
    assert g["xyz_m"] is None and g["spine_width_mm"] is None and g["depth_valid"] == 0.0


def test_glare_hole_is_reported_not_guessed():
    depth = np.zeros((H, W), np.float32)                                 # no depth at all on the spine (glare)
    g = grasp_info(rect(620, 200, 40, 400), depth, INTR)
    assert g["xyz_m"] is None and g["depth_valid"] == 0.0


def test_unequal_focal_lengths_are_used_for_each_spine_axis():
    intr = {**INTR, "fx": 600.0, "fy": 1200.0}
    depth = np.full((H, W), 0.6, np.float32)
    g = grasp_info(rect(610, 280, 60, 240), depth, intr)
    assert g["spine_width_mm"] == 60.0
    assert g["spine_length_mm"] == 120.0


def test_rotated_spine_and_gaps_use_both_focal_lengths():
    intr = {**INTR, "fx": 600.0, "fy": 1200.0}
    depth = np.full((H, W), 0.6, np.float32)
    theta = math.radians(60)
    u = np.array([math.cos(theta), math.sin(theta)])
    v = np.array([u[1], -u[0]])
    c = np.array([640.0, 400.0])
    p = np.array([c - u * 120 - v * 30, c - u * 120 + v * 30,
                  c + u * 120 + v * 30, c + u * 120 - v * 30])
    g = grasp_info(p, depth, intr, neighbours=[p - 90 * v, p + 90 * v])
    long_scale = 600 * np.linalg.norm(u / [intr["fx"], intr["fy"]])
    short_scale = 600 * np.linalg.norm(v / [intr["fx"], intr["fy"]])
    assert g["spine_length_mm"] == round(240 * long_scale, 1)
    assert g["spine_width_mm"] == round(60 * short_scale, 1)
    assert g["gap_left_mm"] == g["gap_right_mm"] == round(30 * short_scale, 1)


# ------------------------------------------------------------------ GraspGen-X hand-off (point clouds, not boxes)
from src.box_grasp import object_point_cloud, save_pointcloud_npz, scene_point_cloud  # noqa: E402


def test_object_cloud_is_the_depth_inside_the_box_in_metres_of_the_camera_frame():
    depth = np.full((H, W), 0.9, np.float32)                       # shelf 0.9 m away
    depth[200:600, 620:660] = 0.6                                  # the spine 0.6 m away
    pts = object_point_cloud(rect(620, 200, 40, 400), depth, INTR)
    assert pts.dtype == np.float32 and pts.shape == (2048, 3)
    assert np.allclose(pts[:, 2], 0.6, atol=1e-4)                  # shelf behind the spine is not in the cloud
    assert pts[:, 0].min() > (620 - 640) * 0.6 / 640 and pts[:, 0].max() < (660 - 640) * 0.6 / 640   # x = (u - cx) z / fx
    assert pts[:, 1].min() > (200 - 400) * 0.6 / 640 and pts[:, 1].max() < (600 - 400) * 0.6 / 640
    assert np.array_equal(pts, object_point_cloud(rect(620, 200, 40, 400), depth, INTR))   # fixed seed


def test_object_cloud_keeps_all_points_when_there_are_fewer_than_requested_and_never_duplicates():
    depth = np.full((H, W), 0.6, np.float32)
    pts = object_point_cloud(rect(620, 200, 20, 30), depth, INTR, n_points=2048)
    assert 100 <= len(pts) < 2048 and len(np.unique(pts, axis=0)) == len(pts)


def test_object_cloud_refuses_glare_missing_depth_and_a_tiny_patch():
    assert object_point_cloud(rect(620, 200, 40, 400), None, INTR) is None
    assert object_point_cloud(rect(620, 200, 40, 400), np.zeros((H, W), np.float32), INTR) is None   # glare hole
    assert object_point_cloud(rect(620, 200, 6, 10), np.full((H, W), 0.6, np.float32), INTR) is None


def test_scene_cloud_labels_target_neighbours_and_background():
    depth = np.full((H, W), 0.9, np.float32)
    pts, labels = scene_point_cloud(depth, INTR, rect(620, 200, 40, 400), [rect(560, 200, 40, 400)], stride=4)
    assert len(pts) == len(labels) and set(np.unique(labels)) == {0, 1, 2}
    on_target = pts[labels == 1]
    assert on_target[:, 0].min() >= (620 - 640) * 0.9 / 640 - 1e-4 and on_target[:, 0].max() <= (664 - 640) * 0.9 / 640 + 1e-4
    assert scene_point_cloud(None, INTR, rect(0, 0, 10, 10)) is None


def test_npz_holds_the_arrays_graspgen_x_needs_and_the_row_points_at_it(tmp_path):
    depth = np.full((H, W), 0.6, np.float32)
    info = save_pointcloud_npz(tmp_path / "grasp_7.npz", rect(620, 200, 40, 400), depth, INTR,
                               neighbours=[rect(560, 200, 40, 400)], frame_size=[W, H])
    assert info == {"file": "grasp_7.npz", "n_object_points": 2048, "frame": "camera_color_m"}
    z = np.load(tmp_path / "grasp_7.npz")
    assert set(z.files) == {"object_points", "scene_points", "scene_labels", "intrinsics", "obb_px", "frame_size"}
    assert z["object_points"].shape == (2048, 3) and z["scene_points"].shape[1] == 3
    assert list(z["intrinsics"]) == [640.0, 640.0, 640.0, 400.0] and list(z["frame_size"]) == [W, H]
    assert save_pointcloud_npz(tmp_path / "none.npz", rect(620, 200, 40, 400), None, INTR) is None
    assert not (tmp_path / "none.npz").exists()

"""pytest tests/test_box_geometry.py -q"""
import math

import numpy as np
import pytest

from src.box_geometry import (aabb_poly, axis_deviation_deg, parse_label_line, point_in_poly, poly_area,
                              poly_iou, rotated_rect_poly, warp_upright)


def test_iou_identical_and_disjoint():
    a = rotated_rect_poly(100, 100, 80, 20, 30)
    assert poly_iou(a, a) == pytest.approx(1.0, abs=1e-4)
    b = rotated_rect_poly(400, 400, 80, 20, 30)
    assert poly_iou(a, b) == 0.0


def test_iou_half_overlap():
    a = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    b = a + np.array([5, 0], dtype=np.float32)
    # irisan 5x10 = 50, gabungan 100 + 100 - 50 = 150 -> 1/3
    assert poly_iou(a, b) == pytest.approx(1 / 3, abs=1e-4)


@pytest.mark.parametrize("theta,expected", [(10, 0.2796), (5, 0.4333)])
def test_straight_box_on_tilted_package(theta, expected):
    """Contoh di panduan: kemasan 600 x 40 px miring theta derajat, dilabel kotak lurus."""
    true_box = rotated_rect_poly(500, 500, 600, 40, theta)
    straight = aabb_poly(true_box)
    L, W, t = 600, 40, math.radians(theta)
    w_aabb = L * math.cos(t) + W * math.sin(t)
    h_aabb = L * math.sin(t) + W * math.cos(t)
    assert poly_area(straight) == pytest.approx(w_aabb * h_aabb, rel=1e-3)
    assert poly_iou(true_box, straight) == pytest.approx(expected, abs=2e-3)


def test_axis_deviation():
    assert axis_deviation_deg(rotated_rect_poly(0, 0, 100, 20, 0)) == pytest.approx(0, abs=0.5)
    assert axis_deviation_deg(rotated_rect_poly(0, 0, 100, 20, 12)) == pytest.approx(12, abs=0.5)


def test_parse_label_lines():
    cls, p = parse_label_line("1 0.1 0.1 0.3 0.1 0.3 0.2 0.1 0.2", 1000, 500)
    assert cls == 1 and p[1].tolist() == pytest.approx([300, 50])
    cls, p = parse_label_line("0 0.5 0.5 0.2 0.4", 1000, 1000)
    assert cls == 0 and poly_area(p) == pytest.approx(200 * 400)
    assert parse_label_line("0 0.5 0.5", 10, 10) is None


def test_point_in_poly():
    p = rotated_rect_poly(50, 50, 60, 10, 45)
    assert point_in_poly((50, 50), p)
    assert not point_in_poly((0, 0), p)


def test_warp_upright_deskews_and_keeps_size():
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    crop = warp_upright(img, rotated_rect_poly(200, 200, 200, 40, 80), min_side=10, pad=0.0)
    h, w = crop.shape[:2]
    assert sorted((h, w)) == [pytest.approx(40, abs=2), pytest.approx(200, abs=2)]

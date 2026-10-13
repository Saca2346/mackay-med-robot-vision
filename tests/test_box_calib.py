"""pytest tests/test_box_calib.py -q  (camera-position calibration card, src/box_calib.py)"""
import math

import numpy as np

from src.box_calib import compare, judge, judge_move, measure, spines, verdict

W, H = 1280, 800


def spine(x, width=55, top=100, h=520, lean_deg=0.0):
    """Upright box polygon; lean_deg turns it around its centre (camera roll)."""
    p = np.array([[x, top], [x + width, top], [x + width, top + h], [x, top + h]], dtype=float)
    c, a = p.mean(axis=0), math.radians(lean_deg)
    r = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return (p - c) @ r.T + c


def shelf(n=12, x0=300, **kw):
    return [spine(x0 + 60 * i, **kw) for i in range(n)]


def card(polys=None, sharp=(700,), still=0.5):
    return measure(shelf() if polys is None else polys, (W, H), list(sharp), still)


def status(checks, name):
    return next(c["status"] for c in checks if c["check"] == name)


def test_good_shelf_passes_every_check():
    checks = judge(card())
    assert verdict(checks) == "PASS", checks


def test_box_width_limits_come_from_the_measured_runs():
    assert status(judge(card(shelf(width=24))), "box width") == "FAIL"     # 08-10 11:18: 24 px -> 7 % size reads
    assert status(judge(card(shelf(width=36))), "box width") == "WARN"     # 08-10 18:28: 36 px -> PICK for iVascular only
    assert status(judge(card(shelf(width=45))), "box width") == "PASS"


def test_monitor_and_post_are_not_counted_as_spines():
    polys = shelf(10) + [np.array([[0, 0], [330, 0], [330, 240], [0, 240]], dtype=float)]   # monitor: 6x wider
    assert len(spines(polys)) == 10 and card(polys)["boxes"] == 10


def test_blur_and_motion_fail():
    assert status(judge(card(sharp=(120,))), "sharpness") == "FAIL"
    assert status(judge(card(still=9.0)), "camera still") == "FAIL"


def test_camera_roll_is_measured_with_its_sign():
    c = card(shelf(lean_deg=5.0))
    assert 4.0 <= abs(c["tilt_deg"]) <= 6.0
    assert status(judge(c), "camera roll") == "WARN"
    assert status(judge(card(shelf(lean_deg=12.0))), "camera roll") == "FAIL"


def test_lying_box_does_not_count_as_camera_roll():
    polys = shelf(10) + [spine(900, width=45, h=520, lean_deg=90.0)]
    c = card(polys)
    assert c["lying"] == 1 and abs(c["tilt_deg"]) < 1.0


def test_shelf_at_the_image_edge_warns():
    c = card(shelf(12, x0=10))                 # first box 10 px from the left edge: 0.8 %
    assert c["margin_left"] < 0.03 and status(judge(c), "shelf inside the frame") == "WARN"
    assert status(judge(card()), "shelf inside the frame") == "PASS"


def test_nothing_detected_fails_without_crashing():
    c = card([])
    assert c["boxes"] == 0 and verdict(judge(c)) == "FAIL"


def test_comparison_sees_a_moved_camera_and_a_changed_distance():
    ref = card()
    moved = card(shelf(12, x0=330))                                     # whole shelf 30 px to the right: within one
    cmp = compare(moved, ref)                                           # spine spacing, where nearest-box matching fails
    assert cmp["shift_px"] == 30.0 and abs(cmp["scale_change"]) < 0.01
    assert judge_move(cmp, same_position=True)["status"] == "WARN"
    assert judge_move(cmp, same_position=False)["status"] == "INFO"
    closer = compare(card(shelf(width=70)), ref)
    assert closer["scale_change"] > 0.2
    other = compare(card(shelf(5)), ref)                                # another part of the shelf: no shift to report
    assert other["shift_px"] is None and judge_move(other, same_position=True)["status"] == "WARN"
    assert judge_move(compare(card(), ref), same_position=True)["status"] == "PASS"

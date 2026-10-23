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


def test_near_top_or_bottom_warns_even_with_safe_horizontal_margins():
    for polys in (shelf(8, top=2), shelf(8, top=278)):
        checks = judge(card(polys))
        assert status(checks, "shelf inside the frame") == "PASS"
        assert status(checks, "top and bottom inside frame") == "WARN"
        assert verdict(checks) == "WARN"


def test_legacy_card_cannot_certify_unmeasured_vertical_margins():
    c = card()
    del c['margin_top'], c['margin_bottom']
    assert status(judge(c), "top and bottom inside frame") == "WARN"


def test_other_box_population_or_resolution_cannot_supply_distance_proxy():
    ref = card()
    for c in (card(shelf(5)), card()):
        if c['boxes'] == ref['boxes']:
            c['frame'] = [1920, 1080]
        comparison = compare(c, ref)
        assert comparison['shift_px'] is None
        assert comparison['scale_change'] is None
        assert judge_move(comparison, True)['status'] == 'WARN'


def test_missing_reference_width_does_not_certify_scale_or_crash():
    ref = card()
    ref['width_px'] = 0
    comparison = compare(card(), ref)
    assert comparison['scale_change'] is None
    assert judge_move(comparison, True)['status'] == 'WARN'


def test_nothing_detected_fails_without_crashing():
    c = card([])
    assert c["boxes"] == 0 and verdict(judge(c)) == "FAIL"


def test_explicit_single_carton_screening_keeps_other_quality_checks():
    one = card([spine(500)])
    assert verdict(judge(one)) == 'FAIL'
    assert verdict(judge(one, min_boxes=1)) == 'PASS'
    assert status(judge(card([spine(500)], sharp=(120,)), min_boxes=1), 'sharpness') == 'FAIL'
    assert status(judge(card([]), min_boxes=1), 'boxes found') == 'FAIL'


def test_region_selection_preserves_original_coordinates_and_rejects_invalid_regions():
    import pytest
    from src.box_calib import parse_region, region_polygons
    target, background = spine(500), spine(900)
    selected = region_polygons([target, background], (W,H), parse_region('0.35,0.05,0.5,0.9'))
    assert len(selected) == 1
    np.testing.assert_array_equal(selected[0], target)
    assert parse_region('none') is None
    for bad in ('0,0,0,1', '0.5,0,0.4,1', '-0.1,0,1,1', '0,0,1,2', '0,0,nan,1', '0,0,1', 'bad'):
        with pytest.raises(ValueError):
            parse_region(bad)
    with pytest.raises(ValueError):
        judge(card(), min_boxes=0)


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


def _scene(seed=0, w=1280, h=800):
    """Textured grey 'shelf' picture (stand-in for a camera frame)."""
    import cv2
    rng = np.random.default_rng(seed)
    img = cv2.resize(rng.integers(0, 255, (h // 20, w // 20)).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return cv2.GaussianBlur(img, (0, 0), 2)


def _small(img):
    import cv2
    return cv2.resize(img, (320, 200), interpolation=cv2.INTER_AREA)


def test_brightness_change_and_noise_are_not_camera_motion():
    # 14 Oct: the old mean-grey check gave 56 on a camera on a tripod (auto-exposure / lamp flicker)
    from src.box_calib import STILL_MAX, camera_motion
    base = _scene().astype(np.float32)
    rng = np.random.default_rng(1)
    frames = [np.clip(base * g + o + rng.normal(0, 4, base.shape), 0, 255).astype(np.uint8)
              for g, o in ((1.0, 0), (1.25, 20), (0.8, -15), (1.1, 35))]
    shift, moving = camera_motion([_small(f) for f in frames], 1280)
    assert shift < 1.0 and moving < 5.0
    c = card(still=shift)
    c["moving_pct"] = moving
    assert status(judge(c), "camera still") == "PASS"


def test_a_shifted_camera_fails_and_a_hand_only_warns():
    from src.box_calib import STILL_MAX, camera_motion
    base = _scene()
    moved = np.roll(base, 12, axis=1)                                   # camera turned by 12 px
    shift, _ = camera_motion([_small(base), _small(moved)], 1280)
    assert abs(shift - 12) < 3 and status(judge(card(still=shift)), "camera still") == "FAIL"
    hand = base.copy()
    hand[300:700, 500:800] = 255 - hand[300:700, 500:800]               # something in front of a part of the shelf
    shift, moving = camera_motion([_small(base), _small(hand)], 1280)
    c = measure(shelf(), (W, H), [700], shift, moving)
    assert shift < STILL_MAX and moving >= 5.0
    assert status(judge(c), "camera still") == "PASS" and status(judge(c), "nothing moving in view") == "WARN"

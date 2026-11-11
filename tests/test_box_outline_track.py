"""Static jitter is suppressed, while actual target motion/loss releases the hold."""
import cv2
import numpy as np

from src.box_outline_track import OutlineTracker


def scene():
    frame = np.full((400, 500, 3), 30, np.uint8)
    poly = np.float32([[180, 50], [260, 50], [260, 350], [180, 350]])
    frame[50:351, 180:261] = 210
    rng = np.random.default_rng(19)
    for x, y in rng.integers([188, 60], [252, 340], (100, 2)):
        cv2.circle(frame, (int(x), int(y)), 2, (20, 60, 80), -1)
    return frame, poly


def settle(tracker, frame, poly):
    return [tracker.update(frame, poly, (1, 0)) for _ in range(4)][-1]


def test_static_pixels_hold_exactly_despite_detector_jitter_and_corner_order():
    frame, poly = scene()
    tracker = OutlineTracker()
    anchor = settle(tracker, frame, poly)
    for i in range(20):
        noise = np.random.default_rng(i).uniform(-2, 2, (4, 2)).astype(np.float32)
        observed = np.roll(poly + noise, i % 4, axis=0)
        observed = observed[::-1] if i % 2 else observed
        output = tracker.update(frame, observed, (1, 0))
        np.testing.assert_array_equal(output, anchor)
        assert tracker.mode == "held"


def test_real_motion_follows_pixels_without_detector_shape_jitter():
    frame, poly = scene()
    tracker = OutlineTracker()
    settle(tracker, frame, poly)
    matrix = np.float32([[1, 0, 8], [0, 1, 4]])
    moved = cv2.warpAffine(frame, matrix, (500, 400), borderValue=(30, 30, 30))
    truth = poly + [8, 4]
    output = tracker.update(moved, truth + [[1, 1], [-1, 0], [1, -1], [0, 1]], (1, 0))
    np.testing.assert_allclose(output, truth, atol=.6)
    assert tracker.mode == "following motion"


def test_subthreshold_motion_accumulates_against_anchor():
    frame, poly = scene()
    tracker = OutlineTracker(deadband_px=1.5)
    settle(tracker, frame, poly)
    for shift in (.5, 1., 2.):
        moved = cv2.warpAffine(frame, np.float32([[1, 0, shift], [0, 1, 0]]), (500, 400))
        output = tracker.update(moved, poly + [shift, 0], (1, 0))
    np.testing.assert_allclose(output, poly + [2, 0], atol=.6)


def test_occlusion_low_texture_loss_and_new_epoch_do_not_keep_old_outline():
    frame, poly = scene()
    tracker = OutlineTracker()
    settle(tracker, frame, poly)
    blank = np.full_like(frame, 128)
    changed = poly + [5, 0]
    np.testing.assert_array_equal(tracker.update(blank, changed, (1, 0)), changed)
    assert tracker.mode == "reacquiring"
    assert tracker.update(frame, None, None) is None
    assert tracker.anchor is None
    settle(tracker, frame, poly)
    np.testing.assert_array_equal(tracker.update(frame, changed, (1, 1)), changed)
    assert tracker.mode == "acquiring"


def test_jump_to_sibling_does_not_reuse_previous_outline():
    frame, poly = scene()
    tracker = OutlineTracker()
    settle(tracker, frame, poly)
    sibling = poly + [90, 0]
    np.testing.assert_array_equal(tracker.update(frame, sibling, (1, 0)), sibling)
    assert tracker.mode == "reacquiring"


def test_obscured_end_releases_hold_even_when_remaining_text_still_tracks():
    frame, poly = scene()
    tracker = OutlineTracker()
    settle(tracker, frame, poly)
    obscured = frame.copy()
    obscured[280:360, 175:266] = (15, 15, 15)
    visible = poly.copy()
    visible[2:, 1] = 280
    output = tracker.update(obscured, visible, (1, 0))
    np.testing.assert_array_equal(output, visible)
    assert tracker.mode == "reacquiring"


def test_display_refinement_jump_does_not_replace_pixel_supported_detector_track():
    frame, poly = scene()
    tracker = OutlineTracker()
    anchor = settle(tracker, frame, poly)
    misleading = poly.copy()
    misleading[:, 0] += 35  # a parallel printed edge chosen by display refinement
    output = tracker.update(frame, misleading, (1, 0), observed_poly=poly)
    np.testing.assert_array_equal(output, anchor)
    assert tracker.mode == "held"
    # A detector observation of a different box must still release the hold.
    sibling = poly + [90, 0]
    np.testing.assert_array_equal(tracker.update(frame, sibling, (1, 0), observed_poly=sibling), sibling)
    assert tracker.mode == "reacquiring"


def test_refinement_runs_on_acquisition_and_loss_but_not_on_a_supported_hold():
    frame, poly = scene()
    tracker = OutlineTracker()
    calls = []
    def refine():
        calls.append(True)
        return poly
    for _ in range(20):
        output = tracker.update(frame, poly, (1, 0), observed_poly=poly, refine=refine)
    assert len(calls) == 3
    np.testing.assert_array_equal(output, poly)
    tracker.update(np.full_like(frame, 128), poly, (1, 0), observed_poly=poly, refine=refine)
    assert len(calls) == 4
    assert tracker.mode == "reacquiring"

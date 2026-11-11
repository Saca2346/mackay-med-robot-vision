"""Display-only outline refinement must follow visible carton edges safely."""
import cv2
import numpy as np
import pytest

from src.box_geometry import poly_iou
from src.box_outline import refine_box_outline


def _scene(angle=0):
    frame = np.full((420, 420, 3), 35, dtype=np.uint8)
    truth = np.float32([[180, 60], [240, 60], [240, 360], [180, 360]])
    detector = np.float32([[183, 85], [237, 85], [237, 325], [183, 325]])
    cv2.rectangle(frame, (180, 60), (240, 360), (190, 220, 235), -1)
    # Coloured bottom branding creates a strong internal horizontal seam. It
    # must not be mistaken for the end, since both carton sides continue.
    cv2.rectangle(frame, (180, 325), (240, 360), (170, 60, 110), -1)
    for y in range(120, 290, 30):
        cv2.putText(frame, '20', (193, y), cv2.FONT_HERSHEY_SIMPLEX, .4, (45, 45, 45), 1)
    if angle:
        m = cv2.getRotationMatrix2D((210, 210), angle, 1.)
        frame = cv2.warpAffine(frame, m, (420, 420), borderValue=(35, 35, 35))
        truth = cv2.transform(truth[None], m)[0]
        detector = cv2.transform(detector[None], m)[0]
    return frame, detector, truth


@pytest.mark.parametrize('angle', [0, 12, 45, -27, 90])
def test_reaches_visible_top_bottom_and_sides_past_printed_seam(angle):
    frame, detector, truth = _scene(angle)
    original = detector.copy()
    pixels = frame.copy()
    refined = refine_box_outline(frame, detector)
    assert poly_iou(refined, truth) > .94
    assert poly_iou(refined, truth) > poly_iou(detector, truth) + .12
    np.testing.assert_array_equal(detector, original)
    np.testing.assert_array_equal(frame, pixels)


def test_neighbour_blocks_new_overlap():
    frame, detector, truth = _scene()
    # A detected box occupies the proposed extension: do not bridge into it.
    neighbour = np.float32([[178, 328], [243, 328], [243, 390], [178, 390]])
    refined = refine_box_outline(frame, detector, [neighbour])
    np.testing.assert_array_equal(refined, detector)


def test_adjacent_separate_box_does_not_absorb_neighbour():
    frame, detector, truth = _scene()
    neighbour = np.float32([[246, 60], [300, 60], [300, 360], [246, 360]])
    cv2.fillConvexPoly(frame, neighbour.astype('int32'), (245, 245, 245))
    refined = refine_box_outline(frame, detector, [neighbour])
    assert poly_iou(refined, truth) > .94
    assert cv2.intersectConvexConvex(refined, neighbour)[0] == 0


def test_blank_and_texture_leave_detector_unchanged():
    _, detector, _ = _scene()
    blank = np.full((420, 420, 3), 128, dtype=np.uint8)
    texture = np.random.default_rng(42).integers(0, 256, (420, 420, 3), dtype=np.uint8)
    for frame in (blank, texture):
        np.testing.assert_array_equal(refine_box_outline(frame, detector), detector)


def test_hidden_end_is_not_extended_behind_occlusion():
    frame, detector, _ = _scene()
    # Occluding the bottom removes the observed termination. The outline must
    # not invent the original box extent behind the occluder.
    frame[310:, :] = 35
    refined = refine_box_outline(frame, detector)
    assert refined[:, 1].max() <= detector[:, 1].max() + 1


def test_frame_clipped_detector_is_kept_without_guessing():
    frame, detector, _ = _scene()
    detector[:, 1] -= 85
    np.testing.assert_array_equal(refine_box_outline(frame, detector), detector)


def test_return_is_independent_copy_on_fallback():
    frame, detector, _ = _scene()
    result = refine_box_outline(np.zeros_like(frame), detector)
    result[0] = -10
    assert np.all(detector >= 0)


def test_visible_bottom_refines_even_when_top_is_not_resolved():
    frame, detector, truth = _scene()
    # The sides continue above the search ROI: top cannot be resolved here.
    cv2.rectangle(frame, (180, 0), (240, 60), (190, 220, 235), -1)
    result = refine_box_outline(frame, detector)
    np.testing.assert_allclose(result[:2], detector[:2], atol=.01)
    assert abs(result[:, 1].max() - truth[:, 1].max()) < 2


@pytest.mark.parametrize('angle', [0, 12, 90])
def test_supported_ends_contract_an_outline_that_overhangs_both_ends(angle):
    frame, _, truth = _scene()
    detector = np.float32([[180,35],[240,35],[240,390],[180,390]])
    if angle:
        matrix = cv2.getRotationMatrix2D((210,210),angle,1.)
        frame = cv2.warpAffine(frame,matrix,(420,420),borderValue=(35,35,35))
        detector = cv2.transform(detector[None],matrix)[0]
        truth = cv2.transform(truth[None],matrix)[0]
    result = refine_box_outline(frame,detector)
    assert poly_iou(result,truth) > .94
    assert poly_iou(result,truth) > poly_iou(detector,truth)+.10


@pytest.mark.parametrize('angle', [0, 12, 90])
def test_inward_stroke_stays_inside_carton_without_moving_source_geometry(angle):
    from src.box_outline import inset_display_outline
    from src.box_target_view import draw_target
    frame, _, truth = _scene(angle)
    original = truth.copy()
    display = inset_display_outline(truth)
    vis = draw_target(frame, display, 'target', 'CHECKING', verified=False)
    background = draw_target(frame, None, 'target', 'CHECKING', verified=False)
    mask = np.zeros(frame.shape[:2],np.uint8)
    cv2.fillConvexPoly(mask,np.rint(truth).astype(np.int32),1)
    # Compare the same native viewport/header so only outline ink is measured.
    changed = np.any(vis != background,axis=2)
    assert changed.any() and not changed[mask==0].any()
    np.testing.assert_array_equal(truth,original)


def test_inset_is_independent_of_corner_direction_and_does_not_change_invalid_input():
    from src.box_outline import inset_display_outline
    _, _, truth = _scene()
    np.testing.assert_allclose(inset_display_outline(truth[::-1])[::-1],inset_display_outline(truth))
    invalid = truth.copy();invalid[0]=np.nan
    np.testing.assert_equal(inset_display_outline(invalid),invalid)


def test_same_colour_neighbour_keeps_uncertain_side_but_refines_visible_ends():
    frame, detector, truth = _scene()
    # A same-colour sibling hides the selected carton's left side.
    cv2.rectangle(frame, (120, 60), (185, 360), (190, 220, 235), -1)
    cv2.rectangle(frame, (120, 325), (185, 360), (170, 60, 110), -1)
    neighbour = np.float32([[120, 60], [180, 60], [180, 360], [120, 360]])
    result = refine_box_outline(frame, detector, [neighbour])
    assert abs(result[:, 1].min() - 60) < 2
    assert abs(result[:, 1].max() - 360) < 2
    assert result[:, 0].min() >= detector[:, 0].min() - .01
    assert cv2.intersectConvexConvex(result, neighbour)[0] == 0


def test_blocked_side_correction_can_still_refine_ends_without_crossing_neighbour():
    frame, detector, _ = _scene()
    neighbour = np.float32([[120, 60], [182, 60], [182, 360], [120, 360]])
    result = refine_box_outline(frame, detector, [neighbour])
    assert abs(result[:, 1].min() - 60) < 2
    assert abs(result[:, 1].max() - 360) < 2
    assert result[:, 0].min() >= 182
    assert cv2.intersectConvexConvex(result, neighbour)[0] == 0


@pytest.mark.parametrize('upside_down', [False, True])
def test_out_of_frame_end_proposal_does_not_cancel_opposite_visible_end(upside_down):
    frame, _, truth = _scene()
    # The carton continues out of the top of the frame. Zero padding around
    # the rotated ROI can resemble an end; it must not cancel the real bottom.
    cv2.rectangle(frame, (180, 0), (240, 60), (190, 220, 235), -1)
    detector = np.float32([[183, 6], [237, 6], [237, 325], [183, 325]])
    if upside_down:
        frame = frame[::-1].copy()
        detector[:, 1] = frame.shape[0] - 1 - detector[:, 1]
        truth[:, 1] = frame.shape[0] - 1 - truth[:, 1]
    result = refine_box_outline(frame, detector)
    if not upside_down:
        np.testing.assert_allclose(result[:2], detector[:2], atol=.01)
        assert abs(result[:, 1].max() - truth[:, 1].max()) < 2
    else:
        assert abs(result[:, 1].min() - truth[:, 1].min()) < 2
        assert abs(result[:, 1].max() - detector[:, 1].max()) < .01

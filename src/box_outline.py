"""Conservative image-supported outlines for the selected box's display only.

The detector polygon remains the geometry used by tracking, OCR, verification and
grasp output. This helper only follows visible straight carton edges near that
polygon. Uncertain ends retain the detector boundary; neighbours block expansion.
"""
from __future__ import annotations

import cv2
import numpy as np


def inset_display_outline(poly, pixels=3.):
    """Inset stroke centres so a thin antialiased line stays inside the boundary.

    Display only: keep the source polygon and its edge/tracking measurements intact.
    Each edge moves inward by a fixed pixel distance, independent of carton length.
    """
    original = np.asarray(poly, np.float32).reshape(4, 2).copy()
    if not np.isfinite(original).all() or not cv2.isContourConvex(original):
        return original
    edges = np.roll(original, -1, axis=0) - original
    lengths = np.linalg.norm(edges, axis=1)
    if lengths.min() < 2 or pixels <= 0:
        return original
    normals = np.column_stack((-edges[:, 1], edges[:, 0])) / lengths[:, None]
    centre = original.mean(axis=0)
    normals *= np.where(np.sum(normals * (centre-original), axis=1) >= 0, 1., -1.)[:, None]
    amount = min(float(pixels), float(lengths.min()) * .10)
    offsets = np.sum(normals*original, axis=1) + amount
    try:
        result = np.array([np.linalg.solve(normals[[i-1, i]], offsets[[i-1, i]]) for i in range(4)], np.float32)
    except np.linalg.LinAlgError:
        return original
    return result if np.isfinite(result).all() and cv2.isContourConvex(result) else original


def _line_support(gradient, xs, ys, tolerance=True):
    """Sample a one-pixel-tolerant line, suppressing isolated printed details."""
    xi = np.rint(xs).astype(np.int32)
    yi = np.rint(ys).astype(np.int32)
    h, w = gradient.shape
    xi = np.clip(xi, 1, w - 2)
    yi = np.clip(yi, 1, h - 2)
    if not tolerance:
        return gradient[yi, xi]
    return np.maximum.reduce([gradient[yi, xi - 1], gradient[yi, xi], gradient[yi, xi + 1]])


def _side(gradient, x, y0, y1, width, outward):
    # The detector already supplies orientation. Search small offsets and small
    # changes in lean; a label edge cannot replace most of a long carton edge.
    ys = np.linspace(y0 + .05 * (y1 - y0), y1 - .05 * (y1 - y0), 160)
    mid = (y0 + y1) / 2
    outside = min(20., max(5., .28 * width))
    inside = min(4., .08 * width)
    offsets = np.arange(-inside, outside + .5) * outward
    slopes = np.linspace(-.018, .018, 13)
    xx = x + offsets[:, None, None] + slopes[None, :, None] * (ys - mid)
    vals = _line_support(gradient, xx, np.broadcast_to(ys, xx.shape), tolerance=False)
    cover = np.mean(vals >= 9., axis=-1)
    quarters = np.min(np.mean((vals >= 9.).reshape(*vals.shape[:-1], 4, 40), axis=-1), axis=-1)
    score = np.mean(np.minimum(vals, 60.), axis=-1) + cover * 12 - np.abs(offsets[:, None]) * .10 - np.abs(slopes[None, :]) * 20
    score[(cover < .58) | (quarters < .40)] = -1
    best = np.unravel_index(np.argmax(score), score.shape)
    if score[best] < 0:
        return None
    # Dense texture with many equally supported locations gives no clear edge.
    if cover[best] < np.median(cover) + .12:
        return None
    return float(x + offsets[best[0]]), float(slopes[best[1]]), float(mid)


def _at(line, ys):
    return line[0] + line[1] * (ys - line[2])


def _end(gx, gy, left, right, y, height, outward, pad, supported_sides=None):
    # Prefer the farthest supported termination, not a printed colour seam. Both
    # long sides must continue to it, then cease outside the proposed carton.
    offsets = np.arange(-min(40., .10 * height), pad - 3.) * outward
    candidates = sorted(y + offsets, reverse=outward > 0)
    across = np.linspace(.12, .88, 40)
    outside_len = min(16., max(7., height * .025))
    sides = (left, right) if supported_sides is None else supported_sides
    strengths = []
    for yy in candidates:
        xl, xr = _at(left, yy), _at(right, yy)
        xs = xl + (xr - xl) * across
        vals = _line_support(gy.T, np.full(xs.shape, yy), xs, tolerance=False)
        strengths.append(float(np.mean(np.minimum(vals, 60.))))
    for index, yy in enumerate(candidates):
        if strengths[index] < max(strengths[max(0, index - 3):index + 4]):
            continue
        xl, xr = _at(left, yy), _at(right, yy)
        xs = xl + (xr - xl) * across
        vals = _line_support(gy.T, np.full(xs.shape, yy), xs, tolerance=False)
        if np.mean(vals >= 9.) < .70:
            continue
        # A shelf rail beyond the carton can have a strong horizontal edge.
        # Require both carton sides immediately inside the end as well as over
        # the longer bridge; distant support alone can jump across a blank gap.
        inside = yy - outward * np.linspace(3., outside_len, 16)
        if any(np.mean(_line_support(gx, _at(side, inside), inside) >= 7.) < .55
               for side in sides):
            continue
        distance = outward * (yy - y)
        if distance > 3:
            bridge = np.linspace(y, yy - 2 * outward, max(8, int(distance)))
            if any(np.mean(_line_support(gx, _at(side, bridge), bridge) >= 7.) < .55
                   for side in sides):
                continue
        beyond = yy + outward * np.linspace(4., outside_len, 16)
        if any(np.mean(_line_support(gx, _at(side, beyond), beyond) >= 7.) > .45
               for side in sides):
            continue
        return float(yy)
    return None


def refine_box_outline(frame: np.ndarray, poly, neighbour_polys=()) -> np.ndarray:
    """Return a display polygon, or an unchanged copy when edges are uncertain.

    Only one bounded, rotated ROI is processed (long dimension <= 900 pixels).
    At least one long side and each moved end must be supported. An ambiguous
    opposite side retains the detector's line. A new outline cannot leave the
    frame, increase overlap with neighbours, or move far from the detector.
    ``neighbour_polys`` must exclude the selected target itself. No input mutates.
    """
    original = np.asarray(poly, dtype=np.float32).reshape(4, 2).copy()
    if frame is None or frame.ndim not in (2, 3) or not np.isfinite(original).all():
        return original
    ih, iw = frame.shape[:2]
    if ih < 8 or iw < 8 or np.any(original < 1) or np.any(original[:, 0] >= iw - 1) or np.any(original[:, 1] >= ih - 1):
        return original
    rect = cv2.minAreaRect(original)
    box = cv2.boxPoints(rect)
    edges = np.roll(box, -1, axis=0) - box
    lengths = np.linalg.norm(edges, axis=1)
    k = int(np.argmax(lengths))
    vertical = edges[k] / lengths[k]
    if vertical[1] < 0:
        vertical = -vertical
    horizontal = np.array([vertical[1], -vertical[0]], dtype=np.float32)
    height, width = float(lengths[k]), float(lengths[(k + 1) % 4])
    if width < 12 or height < 40 or height / width < 1.8:
        return original
    pad_x = min(24., max(8., width * .32))
    pad_y = min(100., max(12., height * .20))
    scale = min(1., 900. / (height + 2 * pad_y), 300. / (width + 2 * pad_x))
    centre = np.mean(original, axis=0)
    origin = centre - horizontal * (width / 2 + pad_x) - vertical * (height / 2 + pad_y)
    transform = np.array([[horizontal[0], horizontal[1], -np.dot(horizontal, origin)],
                          [vertical[0], vertical[1], -np.dot(vertical, origin)]], dtype=np.float32) * scale
    size = (int(np.ceil((width + 2 * pad_x) * scale)), int(np.ceil((height + 2 * pad_y) * scale)))
    roi = cv2.warpAffine(frame, transform, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    smooth = cv2.GaussianBlur(roi, (3, 3), .6)
    gx = np.abs(cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3, scale=.125))
    gy = np.abs(cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3, scale=.125))
    if gx.ndim == 3:
        gx, gy = gx.max(axis=2), gy.max(axis=2)
    # Horizontal print strokes do not support the carton's long side, and vice versa.
    side_gradient = np.where(gx >= gy * 1.2, gx, 0)
    end_gradient = np.where(gy >= gx * 1.2, gy, 0)
    x0, y0 = pad_x * scale, pad_y * scale
    x1, y1 = x0 + width * scale, y0 + height * scale
    left = _side(side_gradient, x0, y0, y1, width * scale, -1)
    right = _side(side_gradient, x1, y0, y1, width * scale, 1)
    if left is None and right is None:
        return original
    # Adjacent cartons of the same design can hide their shared side. Keep that
    # side fixed; the observed outer side plus the broad end edge can still
    # establish the visible top/bottom without guessing the missing lateral edge.
    supported_sides = tuple(s for s in (left, right) if s is not None)
    left = left if left is not None else (x0, 0., (y0 + y1) / 2)
    right = right if right is not None else (x1, 0., (y0 + y1) / 2)
    top = _end(side_gradient, end_gradient, left, right, y0, height * scale, -1, pad_y * scale, supported_sides)
    bottom = _end(side_gradient, end_gradient, left, right, y1, height * scale, 1, pad_y * scale, supported_sides)
    if top is None and bottom is None:
        return original
    # Preserve the detector's corners at an uncertain end, while still allowing
    # the opposite visible termination to be corrected independently.
    local = cv2.transform(original[None], transform)[0]
    by_y = local[np.argsort(local[:, 1])]
    old_top = by_y[:2][np.argsort(by_y[:2, 0])]
    old_bottom = by_y[2:][np.argsort(by_y[2:, 0])]
    inverse = cv2.invertAffineTransform(transform)

    def visible_end(end, l, r):
        # A padded ROI can contain a false termination just outside the actual
        # camera image. Reject that end independently so it cannot cancel an
        # observable correction at the opposite end of the carton.
        if end is None:
            return None
        points = np.float32([[_at(l, end), end], [_at(r, end), end]])
        points = cv2.transform(points[None], inverse)[0]
        if np.any(points < 1) or np.any(points[:, 0] >= iw - 1) or np.any(points[:, 1] >= ih - 1):
            return None
        return end

    top, bottom = visible_end(top, left, right), visible_end(bottom, left, right)
    if top is None and bottom is None:
        return original
    top_known, bottom_known = top is not None, bottom is not None
    top = y0 if top is None else top
    bottom = y1 if bottom is None else bottom
    proposed = np.float32([[_at(left, top), top], [_at(right, top), top],
                           [_at(right, bottom), bottom], [_at(left, bottom), bottom]])
    if not top_known:
        proposed[:2] = old_top
    if not bottom_known:
        proposed[2:] = old_bottom[::-1]
    def candidates():
        yield proposed
        # A neighbouring spine may block a lateral correction while the top/bottom
        # remain observable. Try those ends again on the original side lines.
        original_left, original_right = (x0, 0., (y0 + y1) / 2), (x1, 0., (y0 + y1) / 2)
        end_top = _end(side_gradient, end_gradient, original_left, original_right, y0, height * scale,
                       -1, pad_y * scale, supported_sides)
        end_bottom = _end(side_gradient, end_gradient, original_left, original_right, y1, height * scale,
                          1, pad_y * scale, supported_sides)
        end_top = visible_end(end_top, original_left, original_right)
        end_bottom = visible_end(end_bottom, original_left, original_right)
        ends_only = np.float32([[x0, y0 if end_top is None else end_top],
                                [x1, y0 if end_top is None else end_top],
                                [x1, y1 if end_bottom is None else end_bottom],
                                [x0, y1 if end_bottom is None else end_bottom]])
        if end_top is None:
            ends_only[:2] = old_top
        if end_bottom is None:
            ends_only[2:] = old_bottom[::-1]
        yield ends_only
    area = abs(cv2.contourArea(original))
    for candidate in candidates():
        candidate = cv2.transform(candidate[None], inverse)[0]
        if np.any(candidate < 1) or np.any(candidate[:, 0] >= iw - 1) or np.any(candidate[:, 1] >= ih - 1):
            continue
        proposed_area = abs(cv2.contourArea(candidate))
        # The detector can overhang both physical ends. Supported inward end
        # corrections must be allowed to remove that excess; the former 95%
        # area floor rejected legitimate shortening by more than five percent.
        if area <= 0 or proposed_area < .70 * area or proposed_area > 1.70 * area:
            continue
        blocked = False
        for neighbour in neighbour_polys:
            other = np.asarray(neighbour, dtype=np.float32).reshape(4, 2)
            if not np.isfinite(other).all():
                return original
            old_overlap = cv2.intersectConvexConvex(cv2.convexHull(original), cv2.convexHull(other))[0]
            new_overlap = cv2.intersectConvexConvex(cv2.convexHull(candidate), cv2.convexHull(other))[0]
            if new_overlap > old_overlap + max(2., .002 * area):
                blocked = True
                break
        if not blocked:
            return candidate.astype(np.float32)
    return original

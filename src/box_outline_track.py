"""Motion-checked display outlines. Never supply geometry to OCR or verification."""
from __future__ import annotations

import cv2
import numpy as np

from src.box_geometry import poly_iou


def align_corners(poly, reference):
    q = np.asarray(poly, np.float32).reshape(4, 2)
    choices = [np.roll(order, n, axis=0) for order in (q, q[::-1]) for n in range(4)]
    return min(choices, key=lambda p: float(np.square(p - reference).sum())).copy()


class OutlineTracker:
    """Hold a consensus outline only while target pixels support the same position.

    Optical flow is measured directly from the anchor image, not accumulated over
    frames. Thus small real movement accumulates against the anchor and eventually
    exceeds the deadband. Missing/ambiguous motion always releases the anchor.
    """
    def __init__(self, deadband_px=1.5, settle_frames=3, image_width=960):
        self.deadband = float(deadband_px)
        self.settle_frames = max(1, int(settle_frames))
        self.image_width = image_width
        self.reset()

    def reset(self):
        self.key = self.gray = self.points = self.anchor = self.output = None
        self.samples = []
        self.mode = "untracked"

    def _anchor(self, gray, poly, scale):
        self.gray = gray.copy()
        self.anchor = poly.copy()
        mask = np.zeros(gray.shape, np.uint8)
        cv2.fillConvexPoly(mask, np.rint(poly * scale).astype(np.int32), 255)
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
        self.points = cv2.goodFeaturesToTrack(gray, 100, .015, 4, mask=mask)

    def _motion(self, gray, scale):
        if self.points is None or len(self.points) < 12:
            return None
        lk = dict(winSize=(17, 17), maxLevel=3)
        nxt, a, _ = cv2.calcOpticalFlowPyrLK(self.gray, gray, self.points, None, **lk)
        if nxt is None:
            return None
        back, b, _ = cv2.calcOpticalFlowPyrLK(gray, self.gray, nxt, None, **lk)
        if back is None:
            return None
        keep = (a.ravel() == 1) & (b.ravel() == 1) & (np.linalg.norm((back-self.points).reshape(-1, 2), axis=1) < .7)
        if keep.sum() < max(12, len(keep) * .65):
            return None
        p, q = self.points[keep].reshape(-1, 2), nxt[keep].reshape(-1, 2)
        matrix, inliers = cv2.estimateAffinePartial2D(p, q, method=cv2.RANSAC, ransacReprojThreshold=1.)
        if matrix is None or inliers is None or inliers.sum() < max(12, len(p)*.75):
            return None
        # Text on a small occluder cannot establish the whole target's motion.
        pp = p[inliers.ravel() == 1]
        extent = np.ptp(self.anchor * scale, axis=0)
        axis = int(np.argmax(extent))
        if np.ptp(pp[:, axis]) < extent[axis] * .4:
            return None
        magnification = np.linalg.norm(matrix[:, 0])
        if not .85 <= magnification <= 1.15:
            return None
        matrix[:, 2] /= scale
        return cv2.transform(self.anchor[None], matrix)[0].astype(np.float32)

    def update(self, frame, poly, key, neighbours=(), *, observed_poly=None, refine=None):
        if poly is None or key is None:
            self.reset()
            return None
        raw = np.asarray(poly, np.float32).reshape(4, 2).copy()
        if not np.isfinite(raw).all():
            self.reset()
            return raw
        scale = min(1., self.image_width / frame.shape[1])
        gray = cv2.cvtColor(cv2.resize(frame, None, fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
        if key != self.key or self.gray is None or self.gray.shape != gray.shape:
            if refine is not None:
                raw = np.asarray(refine(), np.float32).reshape(4, 2)
            self.reset()
            self.key, self.output = key, raw.copy()
            self.samples = [raw.copy()]
            self._anchor(gray, raw, scale)
            self.mode = "acquiring"
            return raw
        candidate = align_corners(raw, self.anchor)
        moved = self._motion(gray, scale)
        # A display-edge refinement can jump to a print seam. Association is
        # checked against the current detector observation when it is supplied;
        # the pixel-supported anchor should survive that display-only jump.
        observation = candidate if observed_poly is None else np.asarray(observed_poly, np.float32).reshape(4, 2)
        if moved is None or poly_iou(moved, observation) < .65:
            if refine is not None:
                candidate = align_corners(refine(), self.anchor)
            self.reset()
            self.key, self.output = key, candidate.copy()
            self.samples = [candidate.copy()]
            self._anchor(gray, candidate, scale)
            self.mode = "reacquiring"
            return candidate
        displacement = float(np.linalg.norm(moved-self.anchor, axis=1).max())
        if displacement > self.deadband:
            self.output = moved
            self.samples = [moved.copy()] * self.settle_frames
            self._anchor(gray, moved, scale)
            self.mode = "following motion"
        elif len(self.samples) < self.settle_frames:
            if refine is not None:
                candidate = align_corners(refine(), self.anchor)
            self.samples.append(candidate)
            self.output = np.median(np.stack(self.samples), axis=0).astype(np.float32)
            self._anchor(gray, self.output, scale)
            self.mode = "acquiring" if len(self.samples) < self.settle_frames else "held"
        else:
            self.mode = "held"
        area = abs(cv2.contourArea(candidate))
        for other in neighbours:
            other = cv2.convexHull(np.asarray(other, np.float32).reshape(4, 2))
            old = cv2.intersectConvexConvex(cv2.convexHull(candidate), other)[0]
            new = cv2.intersectConvexConvex(cv2.convexHull(self.output), other)[0]
            if new > old + max(2., area * .015):
                self.reset()
                self.mode = "neighbour guard"
                return candidate
        return self.output.copy()

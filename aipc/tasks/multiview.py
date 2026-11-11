"""Several detector views of one photo merged into one box list (7 Nov 2026), for tilted and missed boxes.

A view = (weights, input size, turn angle). The photo is turned by the angle (canvas enlarged), detected, and the
outlines are turned back. The FIRST view's outlines are kept as they are (so the reads and decisions of the boxes
it finds do not change); a later view only adds an outline that does not overlap a kept one of its class (IoU >=
merge_iou or its centre inside it). merge_iou 0.3. Same interface as src.box_detector.BoxDetector.detect: (list of Det, ms).

  --views "B:544x960:0+B:1280:0+B:1280:-25+B:1280:25"   B = the reader's --det weights, else a path to other weights

8 Nov: the returned list also carries .alt = the later views' outlines that were dropped as the same box as a kept one,
with all labels of the later views; aipc_tasks --alt-views 1 reads them as extra outlines of that box (rd_va vs rd_vb:
the 544 and the 1280 outlines of the same box read different size labels, each one green where the other was held).
"""
import time

import cv2
import numpy as np

from src.box_detector import BoxDetector, Det


def _iou(a, b):
    a, b = cv2.convexHull(a.astype(np.float32)), cv2.convexHull(b.astype(np.float32))
    inter, _ = cv2.intersectConvexConvex(a, b)
    union = cv2.contourArea(a) + cv2.contourArea(b) - inter
    return inter / union if union > 0 else 0.0


def turn(img, deg):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    c, s = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * s + w * c), int(h * c + w * s)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, m, (nw, nh), borderValue=(114, 114, 114)), m


class MultiDetector:
    def __init__(self, spec, default_weights, task, conf, iou, classes, merge_iou=0.3):
        self.views, self.merge_iou, cache = [], merge_iou, {}
        for v in spec.split("+"):
            w, size, deg = v.split(":")
            w = default_weights if w == "B" else w
            imgsz = [int(x) for x in size.split("x")] if "x" in size else int(size)
            key = (w, size)
            if key not in cache:
                cache[key] = BoxDetector(w, task, imgsz, "0", conf, iou, classes, fixed_input=isinstance(imgsz, list))
            self.views.append((cache[key], float(deg)))

    def detect(self, frame):
        t0 = time.perf_counter()
        kept, alt = Dets(), []
        for n, (d, deg) in enumerate(self.views):
            if deg:
                im, m = turn(frame, deg)
                inv = cv2.invertAffineTransform(m)
                dets = [Det(x.cls, x.conf, np.asarray(x.poly, np.float64) @ inv[:, :2].T + inv[:, 2])
                        for x in d.detect(im)[0]]
            else:
                dets = d.detect(frame)[0]
            if n == 0:
                kept.extend(dets)                    # the first view's outlines are kept exactly as they are
                continue
            alt += [x for x in dets if x.cls != "box"]
            # a later view only ADDS what the earlier views missed: an outline overlapping a kept one of its class
            # (IoU >= merge_iou, or its centre inside it) is the same box and is dropped
            new = []
            for x in sorted((x for x in dets if x.cls == "box"), key=lambda x: -x.conf):
                c = tuple(map(float, np.asarray(x.poly).mean(0)))
                if not any(k.cls == "box" and (_iou(k.poly, x.poly) >= self.merge_iou or
                                               cv2.pointPolygonTest(np.asarray(k.poly, np.float32), c, False) >= 0)
                           for k in kept):
                    kept.append(x)
                    new.append(x)
                else:
                    alt.append(x)                    # the same box seen by this view: an extra outline (--alt-views)
            # labels of this view only for the boxes it added: a box an earlier view found keeps its own labels
            for x in dets:
                if x.cls != "box":
                    c = tuple(map(float, np.asarray(x.poly).mean(0)))
                    if any(cv2.pointPolygonTest(np.asarray(b.poly, np.float32), c, False) >= 0 for b in new):
                        kept.append(x)
        kept.alt = alt
        return kept, (time.perf_counter() - t0) * 1000.0


class Dets(list):
    """The merged detections; .alt = the later views' outlines of boxes already kept, and their labels."""
    alt = ()


def attach_alts(groups, alt, min_label_overlap, match_iou=0.5):
    """g.alts = the other views' groups (box + its own labels) of the same box: IoU >= match_iou with g's outline and
    with no other kept box closer (so a neighbour's outline is never read as this box)."""
    from src.box_detector import group_labels
    for g in groups:
        g.alts = []
    if not alt:
        return groups
    agroups, _ = group_labels(list(alt), min_label_overlap)
    for a in agroups:
        ious = [_iou(np.asarray(g.box.poly), np.asarray(a.box.poly)) for g in groups]
        if ious:
            i = int(np.argmax(ious))
            if ious[i] >= match_iou:
                groups[i].alts.append(a)
    return groups

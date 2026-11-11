"""
Crops of the verified (PICK) box for the verification step, the arm and the report.

Response to the review, Figure 2: image -> spine segmentation -> barcode & OCR -> structured output; meeting 15-10:
"crop the barcode inside the box, show it in the output and give its position". Only the ONE PICK box is cut, so
this costs nothing for the other boxes.

  spine       the box's spine, straightened and turned upright (long side vertical)
  size_label  the size label(s) the detector found inside the box (what the OCR read)
  barcode     the barcode inside the spine, bars only (no REF / LOT line, no digits): DECODED (zxing: exact corners,
              GTIN / LOT / expiry) or, when it is too small to decode from this distance, only LOCATED (find_barcode:
              OpenCV's barcode outline, else the bar-pattern block trimmed to its bars), so the scanner on the arm / at
              the basket can be aimed at it. `barcode_decoded` says which; a located barcode is a position, never evidence.
Corners are given in the camera frame (px), like the rest of grasp.jsonl.
"""
from __future__ import annotations

import re
import threading

import cv2
import numpy as np

from src.box_geometry import to_poly


def rectify(frame: np.ndarray, poly, min_side: int = 64):
    """Spine straightened and upright -> (image, 3x3 transform frame -> image)."""
    p = to_poly(poly).astype(np.float32)
    # Walk every corner once. Independent min/max sum/difference sorting can reuse a corner at 45 degrees,
    # collapsing a valid diamond and mapping the barcode position to one vertex of the spine.
    delta = p - p.mean(axis=0)
    p = p[np.argsort(np.arctan2(delta[:, 1], delta[:, 0]))]
    p = np.roll(p, -int(np.argmin(p.sum(axis=1))), axis=0)
    tl, tr, br, bl = p
    w = float(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = float(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    if w > h:                                       # long side horizontal: turn upright (top = old right end)
        p = np.array([tr, br, bl, tl], np.float32)
        w, h = h, w
    s = max(1.0, min_side / max(w, 1.0))
    W, H = max(2, int(round(w * s))), max(2, int(round(h * s)))
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.float32)
    m = cv2.getPerspectiveTransform(p, dst)
    img = cv2.warpPerspective(frame, m, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return img, m


def to_frame(pts, m) -> np.ndarray:
    """Points in the rectified image -> camera frame px."""
    q = np.asarray(pts, np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(q, np.linalg.inv(m)).reshape(-1, 2)


def locate_barcode(spine: np.ndarray):
    """Barcode block in an upright spine by its bar pattern -> (corners (4, 2), score 0..1) or (None, 0)."""
    g = cv2.cvtColor(spine, cv2.COLOR_BGR2GRAY) if spine.ndim == 3 else spine
    H, W = g.shape[:2]
    if W < 12 or H < 24:
        return None, 0.0
    gx = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    best, best_s = None, 0.0
    for bars in (gy - gx, gx - gy):                 # bars across the spine (usual) or along it
        b = np.clip(bars, 0, None)
        b = cv2.blur(b, (max(3, W // 8), max(3, W // 8)))
        if b.max() <= 0:
            continue
        b8 = (255 * b / b.max()).astype(np.uint8)
        _, th = cv2.threshold(b8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        k = max(3, W // 6)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, k // 2),) * 2))
        n, lab, st, _ = cv2.connectedComponentsWithStats(th)
        for i in range(1, n):
            x, y, w, h, area = st[i]
            if w < 0.35 * W or h < 0.6 * W or h > 0.7 * H or area < 0.5 * w * h:
                continue
            region = bars[y:y + h, x:x + w]
            s = float(np.clip(region.mean() / (gx + gy)[y:y + h, x:x + w].mean().clip(1e-6), 0, 1))
            if s > best_s:
                best, best_s = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.float32), s
    return best, best_s


def bar_count(spine: np.ndarray, corners) -> int:
    """Median number of light/dark changes down the middle columns of a located block. A barcode crosses many thin
    bars; a logo, a glare edge or a big numeral only a few (live 29-10, 62 px spines: logos and glare 3-6, real
    barcodes 8-46)."""
    g = cv2.cvtColor(spine, cv2.COLOR_BGR2GRAY) if spine.ndim == 3 else spine
    x0, y0 = np.floor(np.asarray(corners).min(axis=0)).astype(int)
    x1, y1 = np.ceil(np.asarray(corners).max(axis=0)).astype(int)
    reg = g[max(0, y0):y1, max(0, x0):x1].astype(np.float32)
    if reg.shape[0] < 4 or reg.shape[1] < 2:
        return 0
    w = reg.shape[1]
    counts = []
    for col in reg[:, w // 4:max(w // 4 + 1, 3 * w // 4)].T:
        s = np.sign(col - col.mean())
        s = s[s != 0]
        counts.append(int((s[1:] != s[:-1]).sum()))
    return int(np.median(counts))


def barcode_like(spine: np.ndarray, corners, score: float, min_score: float = 0.3, min_bars: int = 8,
                 max_aspect: float = 5.5) -> bool:
    """Is a located block worth SHOWING as a barcode (display and aiming only, never evidence)? Bar pattern score,
    enough bars, and not a whole printed label (Agent 29-10: a block 7 spine widths long)."""
    if corners is None or score < min_score:
        return False
    c = np.asarray(corners)
    return np.ptp(c[:, 1]) <= max_aspect * max(spine.shape[1], 1) and bar_count(spine, c) >= min_bars


# ------------------------------------------------------------------ the barcode alone (owner review 29-10)
# "Only the barcode, without the REF": the bars of a 1-D barcode, cut along their own outline, with no REF / LOT /
# date line beside them and no digits under them. Display, aiming and the verification panel only, never evidence.
_DETECTORS = threading.local()


def _cv_detector():
    """OpenCV's 1-D barcode detector (gradient based, gives the oriented outline of the bars), one per thread."""
    if not hasattr(_DETECTORS, "d"):
        try:
            _DETECTORS.d = cv2.barcode.BarcodeDetector()
        except Exception:  # noqa: BLE001 - an OpenCV build without the barcode module: located path only
            _DETECTORS.d = None
    return _DETECTORS.d


def order_quad(q) -> np.ndarray:
    """4 corners -> tl, tr, br, bl with tl -> tr along the LONG side (along the barcode, across its bars)."""
    q = np.asarray(q, np.float32).reshape(4, 2)
    d = q - q.mean(axis=0)
    q = q[np.argsort(np.arctan2(d[:, 1], d[:, 0]))]
    if np.linalg.norm(q[1] - q[0]) < np.linalg.norm(q[2] - q[1]):
        q = np.roll(q, -1, axis=0)
    if q[0, 0] > q[1, 0]:                                     # read left to right
        q = np.roll(q, 2, axis=0)
    return q


def _quad_transform(quad, min_short: int = 0):
    """Ordered outline -> (3x3 transform img -> straightened crop, crop width, crop height)."""
    q = order_quad(quad)
    L = max(2, int(round(max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3])))))
    S = max(2, int(round(max(np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1])))))
    s = max(1.0, min_short / S)
    L, S = int(round(L * s)), int(round(S * s))
    return cv2.getPerspectiveTransform(q, np.float32([[0, 0], [L - 1, 0], [L - 1, S - 1], [0, S - 1]])), L, S


def warp_quad(img: np.ndarray, quad, min_short: int = 0) -> np.ndarray:
    """The inside of an outline, straightened: long side horizontal (bars vertical, as a barcode is shown)."""
    m, L, S = _quad_transform(quad, min_short)
    return cv2.warpPerspective(img, m, (L, S), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def bars_across(crop: np.ndarray) -> int:
    """Light/dark changes along a straightened barcode crop (bars vertical), median over its middle rows."""
    return bar_count(cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE),
                     [[0, 0], [crop.shape[0], 0], [crop.shape[0], crop.shape[1]], [0, crop.shape[1]]])


def _corr(a, b) -> float:
    sa, sb = a.std(), b.std()
    return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb)) if sa > 1e-6 and sb > 1e-6 else 0.0


def trim_to_bars(crop: np.ndarray, min_row_corr: float = 0.5):
    """A straightened outline (bars vertical) cut to the bars -> (x0, y0, x1, y1) in crop px or None.

    Every row through a barcode repeats the same light/dark profile and every bar runs the full height; the printed
    lines around it do not (29-10 photos: the digits under the bars, the LOT / REF box and the BIOSENSORS logo at the
    ends of a CHROMA barcode were inside OpenCV's outline). Rows: kept from the middle out while they repeat the middle
    profile. Along the barcode: the longest run of bar edges whose columns are even top to bottom (see below)."""
    g = (cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop).astype(np.float32)
    S, L = g.shape
    if S < 4 or L < 12:
        return None
    ref = np.median(g[S // 4:max(S // 4 + 1, 3 * S // 4)], axis=0)
    rc = np.array([_corr(r, ref) for r in g])
    top = bottom = S // 2
    while top > 0 and max(rc[top - 1], rc[max(0, top - 2)]) >= min_row_corr:
        top -= 1
    while bottom < S - 1 and max(rc[bottom + 1], rc[min(S - 1, bottom + 2)]) >= min_row_corr:
        bottom += 1
    body = g[top:bottom + 1]
    if body.shape[0] < 3:
        return None
    p = body.mean(axis=0)
    edge = np.abs(np.diff(p, append=p[-1]))
    even = cv2.blur(body.std(axis=0).reshape(1, -1), (3, 1)).ravel()
    # a bar is ink from top to bottom; printed digits beside the bars (the expiry date next to an Accuforce barcode,
    # live 30-10) are ink in the middle only: such columns, and the edges next to them, are not bars
    n = body.shape[0]
    light = float(np.percentile(p, 90))
    ink = [np.clip(light - body[r0:r1].mean(axis=0), 0, None) for r0, r1 in ((0, max(1, n // 4)), (n // 4, max(n // 4 + 1, 3 * n // 4)),
                                                                          (3 * n // 4, n))]
    dark = ink[1] > 0.1 * (light - float(p.min()) + 1e-6)
    short = dark & ((ink[0] < 0.4 * ink[1]) | (ink[2] < 0.4 * ink[1]))
    short = cv2.dilate(short.astype(np.uint8).reshape(1, -1), np.ones((1, 5), np.uint8)).ravel().astype(bool)
    xs = np.where((edge > 0.3 * np.percentile(edge, 95)) & (even < 0.6 * (p.std() + 1e-6)) & ~short)[0]
    if len(xs) < 4:
        return None
    # one bar edge spans 1-3 px: neighbouring pixels are one edge; a bar run ends where the next edge is > 5 typical
    # edge spacings away (a space inside a barcode is <= 4 modules, the quiet zone >= 10; a logo or label border
    # beside the bars is one lone edge past the quiet zone)
    edges = np.split(xs, np.where(np.diff(xs) > 1)[0] + 1)
    centres = np.array([e.mean() for e in edges])
    if len(centres) < 4:
        return None
    gap = max(3.0, 5.0 * float(np.median(np.diff(centres))))
    runs, start = [], 0
    for i in range(1, len(edges) + 1):
        if i == len(edges) or centres[i] - centres[i - 1] > gap:
            runs.append((i - start, edges[start][0], edges[i - 1][-1]))
            start = i
    _, a, b = max(runs)
    pad = max(2, int(0.01 * L))
    return max(0, a - pad), top, min(L, b + 1 + pad), bottom + 1


def _deskew(img: np.ndarray, quad) -> np.ndarray:
    """Outline turned to the lean of its bars: an outline square to the spine leaves the bars of a box photographed at
    an angle slanted inside it (29-10 CHROMA photo: ~5 px over a 40 px bar), so no two rows repeat one profile. The
    shift between the profiles of the top and the bottom band gives the lean; the outline becomes a parallelogram
    along the bars (img px)."""
    m, L, S = _quad_transform(quad)
    if S < 8:
        return quad
    g = cv2.cvtColor(cv2.warpPerspective(img, m, (L, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE),
                     cv2.COLOR_BGR2GRAY).astype(np.float32) if img.ndim == 3 else \
        cv2.warpPerspective(img, m, (L, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    band = max(2, S // 5)
    yt, yb = S // 4, 3 * S // 4
    top = g[max(0, yt - band // 2):yt + band // 2 + 1].mean(axis=0)
    bot = g[max(0, yb - band // 2):yb + band // 2 + 1].mean(axis=0)
    reach = max(1, min(L // 4, S))
    shifts = range(-reach, reach + 1)
    scores = [_corr(top[max(0, -d):L - max(0, d)], bot[max(0, d):L - max(0, -d)]) for d in shifts]
    d = list(shifts)[int(np.argmax(scores))]
    if d == 0:
        return quad
    k = d / float(yb - yt)                                    # x shift per row along a bar
    mid = (yt + yb) / 2.0
    s0, s1 = k * (0 - mid), k * (S - 1 - mid)
    lean = np.float32([[s0, 0], [L - 1 + s0, 0], [L - 1 + s1, S - 1], [s1, S - 1]])
    return to_frame(lean, m).astype(np.float32)


def _grow(quad, shape, along: float = 0.5, across: float = 0.15) -> np.ndarray:
    """Outline lengthened by `along` of its length at each end and widened by `across`, inside the image: OpenCV's
    outline often covers only part of a blurred barcode (29-10 CHROMA photo: 20-60 % of it); trim_to_bars then finds
    where the bars really end (the print and the quiet zone beyond them are not bar columns)."""
    q = order_quad(quad)
    u, v = (q[1] - q[0]), (q[3] - q[0])
    g = np.float32([q[0] - along * u - across * v, q[1] + along * u - across * v,
                    q[2] + along * u + across * v, q[3] - along * u + across * v])
    h, w = shape[:2]
    g[:, 0], g[:, 1] = g[:, 0].clip(0, w - 1), g[:, 1].clip(0, h - 1)
    return g


def _bars_only(img: np.ndarray, quad, min_bars: int):
    """Outline -> outline of its bars alone (img px) and bars count, or (None, bars) when what is left is not a
    barcode. The outline itself is checked first; a lengthened one (_grow) replaces it only when it is a barcode too
    and longer (a lengthened outline of a clean barcode can take in print that breaks the check)."""
    best = _bars_in(img, _deskew(img, quad), min_bars)
    for _ in range(2):                                        # at most twice: 50 % then 50 % more at each end
        if best[0] is None:
            break
        longer = _bars_in(img, _deskew(img, _grow(best[0], img.shape)), min_bars)
        if longer[0] is None or np.linalg.norm(longer[0][1] - longer[0][0]) <= 1.02 * np.linalg.norm(best[0][1] - best[0][0]):
            break
        best = longer
    return best


def _bars_in(img: np.ndarray, quad, min_bars: int):
    """One outline cut to its bars and checked (>= min_bars bars, rows repeating one profile)."""
    m, L, S = _quad_transform(quad)
    crop = cv2.warpPerspective(img, m, (L, S), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    box = trim_to_bars(crop)
    if box is None:
        return None, 0
    x0, y0, x1, y1 = box
    bars = crop[y0:y1, x0:x1]
    g = (cv2.cvtColor(bars, cv2.COLOR_BGR2GRAY) if bars.ndim == 3 else bars).astype(np.float32)
    ref = np.median(g, axis=0)
    rows = float(np.median([_corr(r, ref) for r in g]))
    n = bars_across(bars)
    if n < min_bars or rows < 0.9 or x1 - x0 < 12 or y1 - y0 < max(4, 0.04 * (x1 - x0)):   # not a thin strip of colour
        return None, n
    corners = np.float32([[x0, y0], [x1 - 1, y0], [x1 - 1, y1 - 1], [x0, y1 - 1]])
    return to_frame(corners, m).astype(np.float32), n


def _merge(rects):
    """Pieces of one barcode (a low-resolution spine breaks it into 2-4 outlines) -> one outline per barcode."""
    rects = [list(r) for r in rects]
    merged = True
    while merged and len(rects) > 1:
        merged = False
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                si, sj = cv2.minAreaRect(np.float32(rects[i]))[1], cv2.minAreaRect(np.float32(rects[j]))[1]
                both = cv2.minAreaRect(np.float32(rects[i] + rects[j]))[1]
                # merged outline barely thicker than its pieces and no longer than both end to end: one line
                if min(both) <= 1.6 * max(min(si), min(sj)) and \
                        max(both) <= max(si) + max(sj) + 3.0 * max(1.0, min(min(si), min(sj))):
                    rects[i] = rects[i] + rects[j]
                    del rects[j]
                    merged = True
                    break
            if merged:
                break
    return [cv2.boxPoints(cv2.minAreaRect(np.float32(r))) for r in rects]


def find_barcode(img: np.ndarray, min_bars: int = 16, min_aspect: float = 1.8):
    """Outline of the bars of a 1-D barcode in a spine crop (any orientation) -> (corners (4, 2) tl, tr, br, bl in
    img px, bars count) or (None, 0). Candidates: OpenCV's barcode detector (spines under 100 px across enlarged for it,
    pieces on one line merged; long side >= min_aspect x short side), else the bar-pattern block of locate_barcode.
    Each is cut to its bars (trim_to_bars) and kept only with >= min_bars bars and rows that repeat one profile
    (29-10: real barcodes 25-196 bars; "MR", "mm", the "0" of a "10", "angiolite" and "TE" 9-13 bars or uneven rows)."""
    if img is None or img.size == 0 or min(img.shape[:2]) < 8:
        return None, 0
    cands = []
    det = _cv_detector()
    f = min(4.0, max(1.0, 100.0 / min(img.shape[:2])))
    big = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC) if f > 1 else img
    if det is not None:
        try:
            ok, pts = det.detect(big)
        except cv2.error:
            ok, pts = False, None
        if ok and pts is not None and len(pts):
            for q in _merge([list(map(tuple, p)) for p in np.asarray(pts, np.float32).reshape(-1, 4, 2)]):
                q = order_quad(q)
                short = np.linalg.norm(q[3] - q[0])
                if short >= 2 and np.linalg.norm(q[1] - q[0]) >= min_aspect * short:
                    cands.append(q)
    if not cands:
        upright, turned = (big, False) if big.shape[0] >= big.shape[1] else (cv2.rotate(big, cv2.ROTATE_90_CLOCKWISE), True)
        corners, score = locate_barcode(upright)
        if corners is not None and score >= 0.3:
            if turned:                                       # back to big px: upright (x, y) = (h - 1 - y, x) of big
                corners = np.float32([[c[1], big.shape[0] - 1 - c[0]] for c in corners])
            cands.append(order_quad(corners))
    best, best_key = (None, 0), 0.0
    for q in cands:
        corners, n = _bars_only(big, q, min_bars)
        if corners is not None:
            key = n * np.linalg.norm(corners[1] - corners[0])  # the longest outline with the most bars
            if key > best_key:
                best, best_key = (order_quad(corners / f), n), key
    return best


def barcode_only(spine: np.ndarray):
    """The barcode of a spine crop, bars only, straightened (bars vertical) -> (image, corners in spine px) or
    (None, None)."""
    corners, _ = find_barcode(spine)
    if corners is None:
        return None, None
    return warp_quad(spine, corners), corners


def numeral_crop(img, poly, unit=True):
    """Crop of one OCR token (a size numeral). With unit=True the box is widened to the print around it, mostly
    downwards, so the "mm" under the numeral stays in (owner review 23-10: "mm" was cut off). Display only."""
    p = np.asarray(poly, np.float32)
    (x0, y0), (x1, y1) = p.min(axis=0), p.max(axis=0)
    w, h = x1 - x0, y1 - y0
    px, top, bottom = (0.12 * w, 0.10 * h, 0.75 * h) if unit else (2, 2, 2)
    H, W = img.shape[:2]
    a, b = max(0, int(np.floor(x0 - px))), max(0, int(np.floor(y0 - top)))
    c, d = min(W, int(np.ceil(x1 + px))), min(H, int(np.ceil(y1 + bottom)))
    return img[b:d, a:c].copy()


def _number(text):
    m = re.fullmatch(r"\s*(\d{1,2}(?:[.,]\d{1,2})?)\s*(?:mm)?\s*", text or "", re.I)
    return float(m.group(1).replace(",", ".")) if m else None


def pick_number_tiles(numbers, diameter=None, length=None, limit=4):
    """The numeral crops to show: the best-read one for the diameter and for the length of the verified identity (one each, so
    no date fragment and no duplicate); without a verified size, the distinct short numerals. -> list of number dicts."""
    best = {}
    for n in numbers or []:
        v = _number(n.get("text"))
        if v is None:
            continue
        for want in (diameter, length):
            if want is not None and abs(v - float(want)) < 1e-6 and n["confidence"] >= best.get(want, (0,))[0]:
                best[want] = (n["confidence"], n)
    chosen = [best[w][1] for w in (diameter, length) if w is not None and w in best]
    if chosen or diameter is not None or length is not None:
        return chosen
    seen, out = set(), []
    for n in sorted(numbers or [], key=lambda n: -n["confidence"]):
        key = _number(n.get("text"))
        if key is not None and key not in seen:
            seen.add(key)
            out.append(n)
    return out[:limit]


_DATE = re.compile(r"(?:19|20)\d\d\s*[-./]\s*\d{1,2}\s*[-./]\s*\d{1,2}")


def best_date_crop(bands, min_conf=0.9):
    """The crop of the best-read COMPLETE printed date token among the bands the OCR read (upright, as it was read)
    -> (image, text) or (None, None). A fragment ("202") or a low-confidence read gives nothing."""
    best = None
    for band in bands or []:
        for t in band.get("tokens", []):
            if t.poly is None or t.conf < min_conf or not _DATE.fullmatch(t.text.strip()):
                continue
            if best is None or t.conf > best[0]:
                best = (t.conf, band["image"], t.poly, t.text.strip())
    if best is None:
        return None, None
    image = numeral_crop(best[1], best[2], unit=False)
    if not image.size:
        return None, None
    return cv2.copyMakeBorder(image, 4, 4, 4, 4, cv2.BORDER_REPLICATE), best[3]


def upright_date(image, reader):
    """A date crop turned so it reads upright: the OCR reads upside-down text through its built-in classifier, so the crop
    itself may be upside down (a Sled Bag spine, owner review 23-10). Both turns are read with the classifier off and the
    one that reads as a complete date with the higher confidence wins; the crop as it is wins a tie. Display only."""
    if image is None or reader is None:
        return image
    def score(img):
        toks = reader.read_as_is(img).tokens
        return max([t.conf for t in toks if _DATE.fullmatch(t.text.strip())] or [0.0])
    try:
        turned = cv2.rotate(image, cv2.ROTATE_180)
        return turned if score(turned) > score(image) else image
    except Exception:  # noqa: BLE001 - never lose the evidence panel over a display detail
        return image


def pick_crops(frame: np.ndarray, group, decode=None) -> dict:
    """The crops of one box. decode = src.box_barcode.decode_gs1 (or None: locate only)."""
    spine, m = rectify(frame, group.box.poly)
    out = {"spine": spine, "size_labels": [], "barcode": None, "barcode_px": None, "barcode_decoded": False,
           "barcode_text": "", "barcode_score": 0.0}
    for lab in group.labels:
        crop, _ = rectify(frame, lab.poly, min_side=48)
        out["size_labels"].append(crop)
    corners = None
    if decode is not None:
        big = cv2.resize(spine, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC) if spine.shape[1] < 120 else spine
        f = big.shape[1] / spine.shape[1]
        for code in decode(big, hard=True):
            if code.poly is not None:
                corners = np.asarray(code.poly, np.float32) / f
                out["barcode_decoded"], out["barcode_text"] = True, code.raw or code.gtin
                break
    if corners is None:
        corners, out["barcode_score"] = find_barcode(spine)     # None for a logo or a printed label: nothing to aim at
    if corners is not None:
        # the bars alone, along their own outline: no REF / LOT line and no digits beside them (owner review 29-10)
        out["barcode"] = warp_quad(spine, corners)
        out["barcode_px"] = to_frame(order_quad(corners), m).round(1).tolist()
    return out


def _fit(img, h, max_w=None):
    s = h / max(img.shape[0], 1)
    if max_w:
        s = min(s, max_w / max(img.shape[1], 1))
    size = (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s)))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)


def readable(img, reader=None):
    """A long crop turned so its text runs left to right: with an OCR reader the turn (90 / 270 degrees) that
    reads with the higher confidence, without one 90 degrees clockwise. The OCR's 180 degree classifier is off for
    this choice (read_as_is): with it on, the upside-down turn reads as well and can win (owner review 29-10:
    Accuforce, Ryurei, Conqueror and Nagomi size labels upside down in the panel). Display only."""
    if img is None or img.shape[0] <= img.shape[1] * 1.2:
        return img
    turns = [cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)]
    if reader is None:
        return turns[0]
    read = getattr(reader, "read_as_is", None) or reader.read_once
    try:
        conf = [sum(x.conf * len(x.text) for x in read(t).tokens) for t in turns]
    except Exception:  # noqa: BLE001 - never lose the evidence panel over a display detail
        return turns[0]
    return turns[int(np.argmax(conf))]


def panel(crops: dict, fields: dict, height: int = 300, reader=None) -> np.ndarray:
    """Figure-2 style strip (Response to the review): spine | barcode | size label | structured output."""
    bg, ink, grey = (245, 245, 242), (20, 20, 20), (90, 90, 90)
    body_h = height - 34

    def block(img, title, max_w):
        top = np.full((34, 10, 3), bg, np.uint8)
        body = _fit(img, body_h, max_w) if img is not None and img.size else np.full((60, 160, 3), 228, np.uint8)
        w = max(body.shape[1], 8 + 10 * len(title))
        top = np.full((34, w, 3), bg, np.uint8)
        cv2.putText(top, title, (4, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, grey, 1, cv2.LINE_AA)
        body = cv2.copyMakeBorder(body, 0, body_h - body.shape[0], 0, w - body.shape[1], cv2.BORDER_CONSTANT,
                                  value=bg)
        return np.vstack([top, body])

    parts = [block(crops.get("spine"), "spine", 60)]
    bc = crops.get("barcode")
    if bc is not None and bc.shape[0] > bc.shape[1]:
        bc = readable(bc, reader)                                # bars upright, its printed digits the right way up
    if bc is not None and bc.shape[1] > 3 * bc.shape[0]:
        # a bars-only crop of a far spine is a thin strip (13 x 152 px live); a 1-D code carries nothing along its
        # bars, so it is shown taller (bars stretched, never squeezed across) to stay visible in the panel
        bc = cv2.resize(bc, (bc.shape[1], bc.shape[1] // 3), interpolation=cv2.INTER_LINEAR)
    if not crops.get("barcode_disabled", False):
        parts.append(block(bc, "barcode, " + ("decoded" if crops.get("barcode_decoded") else
                                              "located (not decoded)" if bc is not None else "not found"), 380))
    labs = crops.get("size_labels") or []
    parts.append(block(readable(labs[0], reader) if labs else None, crops.get("size_title", "size label (OCR)"), 380))
    if crops.get("dates"):
        parts.append(block(crops["dates"][0], "expiry date (OCR)", 200))
    txt = np.full((height, 400, 3), bg, np.uint8)
    cv2.putText(txt, "structured output", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, grey, 1, cv2.LINE_AA)
    y = 56
    for k, v in fields.items():
        if v in (None, ""):
            continue
        cv2.putText(txt, f"{k}: {v}"[:40], (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, ink, 1, cv2.LINE_AA)
        y += 25
    parts.append(txt)
    parts = [cv2.copyMakeBorder(p, 0, 0, 0, 10, cv2.BORDER_CONSTANT, value=bg) for p in parts]
    out = np.hstack(parts)
    cv2.rectangle(out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1), (0, 200, 0), 3)
    return out


def overlay(vis: np.ndarray, pnl: np.ndarray, margin: int = 10) -> None:
    """Paste the panel at the bottom of the screen picture (scaled to fit)."""
    H, W = vis.shape[:2]
    s = min(1.0, (W - 2 * margin) / pnl.shape[1], 0.42 * H / pnl.shape[0])
    p = cv2.resize(pnl, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else pnl
    y0, x0 = H - p.shape[0] - margin, margin
    vis[y0:y0 + p.shape[0], x0:x0 + p.shape[1]] = p

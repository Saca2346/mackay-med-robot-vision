"""
Crops of the verified (PICK) box for the verification step, the arm and the report.

Response to the review, Figure 2: image -> spine segmentation -> barcode & OCR -> structured output; meeting 15-10:
"crop the barcode inside the box, show it in the output and give its position". Only the ONE PICK box is cut, so
this costs nothing for the other boxes.

  spine       the box's spine, straightened and turned upright (long side vertical)
  size_label  the size label(s) the detector found inside the box (what the OCR read)
  barcode     the barcode inside the spine, DECODED (zxing: exact corners, GTIN / LOT / expiry) or, when it is too
              small to decode from this distance, only LOCATED by its bar pattern (strong contrast across the bars,
              little along them, a compact block), so the scanner on the arm / at the basket can be aimed at it.
              `barcode_decoded` says which; a located barcode is a position, never evidence.
Corners are given in the camera frame (px), like the rest of grasp.jsonl.
"""
from __future__ import annotations

import re

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


def barcode_with_text(spine, min_score=0.3):
    """Located (not decoded) barcode of an upright spine with the print around it (REF text, dates) -> (image, score)
    or (None, score). Below min_score the block is more likely a logo or text (K16 and K04 of 23-10 scored 0.14 and
    0.10; the real barcodes of K07 and K21 0.39 and 0.44)."""
    corners, score = locate_barcode(spine)
    if corners is None or score < min_score:
        return None, score
    y0, y1 = int(corners[:, 1].min()), int(corners[:, 1].max())
    h = y1 - y0
    top, bottom = max(0, y0 - int(0.35 * h)), min(spine.shape[0], y1 + int(0.6 * h))
    return spine[top:bottom, :].copy(), score


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
        corners, out["barcode_score"] = locate_barcode(spine)
    if corners is not None:
        x0, y0 = np.floor(corners.min(axis=0)).astype(int)
        x1, y1 = np.ceil(corners.max(axis=0)).astype(int)
        pad = max(2, int(0.08 * spine.shape[1]))
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(spine.shape[1], x1 + pad), min(spine.shape[0], y1 + pad)
        out["barcode"] = spine[y0:y1, x0:x1].copy()
        out["barcode_px"] = to_frame([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], m).round(1).tolist()
    return out


def _fit(img, h, max_w=None):
    s = h / max(img.shape[0], 1)
    if max_w:
        s = min(s, max_w / max(img.shape[1], 1))
    size = (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s)))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)


def readable(img, reader=None):
    """A long crop turned so its text runs left to right: with an OCR reader the turn (90 / 270 degrees) that
    reads with the higher confidence, without one 90 degrees clockwise."""
    if img is None or img.shape[0] <= img.shape[1] * 1.2:
        return img
    turns = [cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)]
    if reader is None:
        return turns[0]
    conf = []
    for t in turns:
        toks = reader.read_once(t).tokens
        conf.append(sum(x.conf * len(x.text) for x in toks))
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
        bc = cv2.rotate(bc, cv2.ROTATE_90_CLOCKWISE)             # bars upright, as a barcode is usually shown
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

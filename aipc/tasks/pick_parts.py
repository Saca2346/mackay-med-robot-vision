"""PICK crops (8 Nov 2026, owner: "crop brand, detail and barcode from the box"): the brand, the detail and the barcode of
the PICKed box, cut after its MATCH as part of the PICK (aipc_tasks.py --pick-crops 1, inside the PICK time). Display,
the report and the arm only, never evidence: the decision is made before this runs and nothing here can change it.

  brand    the product name as the OCR read it on the outline that gave the MATCH (tokens naming the requested product,
           and the maker's name where it is printed next to it), cut from the very image the OCR read
  detail   the printed size (tokens holding the requested diameter / length), the REF code and the GTIN text read on
           that outline, cut from the images the OCR read; one line per block of print. A product without a size: the
           printed date read
  barcode  the bars alone (src/box_crops.find_barcode: no REF / LOT line, no digits) on the box straightened from the
           drawn outline at the photo's full resolution. LOCATED, not decoded: barcode decoding stays off as in the
           production config. Its corners in photo px, for the scanner on the arm

No OCR is run for brand and detail: they come from the reads already made (aipc_tasks records the OCR inputs and tokens
of each outline read). Turning a crop upright for display (upright) reads it once more, after the clock has stopped.
"""
from __future__ import annotations

import json
import os
import re
import sys

import cv2
import numpy as np

from src.box_barcode import find_gtins_in_text
from src.box_crops import find_barcode, order_quad, rectify, to_frame, warp_quad
from src.box_expiry import find_dates
from src.box_ref import parse_refs
from src.box_verifier import brand_of

_NUM = re.compile(r"(?<![\d.,])\d{1,2}(?:[.,]\d{1,2})?(?![\d.,]*\d)")
_JOINED = re.compile(r"(?<![\d.,])(\d)[.,](\d{1,2})(\d{2})(?![\d.,]*\d)")    # "3.533" = 3.5 and 33 (size_split.py)
MIN_CONF = 0.5


def letters(s):
    return re.sub(r"[^a-z]", "", (s or "").lower())


def is_name(text, product):
    """A token naming the requested product: its letters are a word of the name (>= 3 letters), or hold one / are held
    by one (>= 4 letters): "SeQuent", "Please", "NEO", "Accuforce", "CHROMA"."""
    t = letters(text)
    if len(t) < 3 or not product:
        return False
    return any(t == w or (len(w) >= 4 and w in t) or (len(t) >= 4 and t in w)
               for w in (letters(x) for x in product.split()) if w)


def is_maker(text, product):
    """A token with the maker's name of the requested product ("TERUMO", "B|BRAUN", "BIOSENSORS", "Boston")."""
    t, b = letters(text), letters(brand_of(product or ""))
    return len(t) >= 5 and bool(b) and (b in t or t in b)


def numbers(text):
    s = {float(n.replace(",", ".")) for n in _NUM.findall(text or "")}
    for a, b, c in _JOINED.findall(text or ""):
        s |= {float(f"{a}.{b}"), float(c)}
    return s


def is_size(text, req):
    """A token holding a requested size number and little else (no REF code, no date, at most 3 other letters: "mm",
    "x", "L")."""
    if req.diameter is None or parse_refs(text) or find_dates(text):
        return False
    rest = letters(re.sub(r"mm", "", text, flags=re.I))
    return len(rest) <= 3 and bool(numbers(text) & {req.diameter, req.length})


def is_ref(text, req):
    """A token whose REF code names the requested product and size (the REF the decision could use)."""
    return any(r.product == req.product and (req.diameter is None or (r.diameter, r.length) == (req.diameter, req.length))
               for r in parse_refs(text))


def _rect(t):
    p = np.asarray(t.poly, np.float32).reshape(-1, 2)
    return p.min(0), p.max(0)


def _thick(t):
    """The height of a token's text: the short side of its box (a token can run down a tall OCR input)."""
    lo, hi = _rect(t)
    return float(min(hi - lo))


def _vertical(t):
    """A token whose text runs along y in its OCR input (3+ characters in a box taller than wide)."""
    lo, hi = _rect(t)
    w, h = hi - lo
    return len(t.text.strip()) >= 3 and h > 1.2 * w


def cluster(tokens, start, reach=1.5):
    """start + the tokens of the list whose box lies within `reach` text heights of the growing union (one line or
    block of print, not the same word printed again at the other end of the box) -> (lo, hi, tokens)."""
    lo, hi = _rect(start)
    picked, rest = [start], [t for t in tokens if t is not start]
    grown = True
    while grown:
        grown = False
        hgt = float(np.median([_thick(t) for t in picked]))
        for t in list(rest):
            a, b = _rect(t)
            gap = np.maximum(np.maximum(lo - b, a - hi), 0)
            if gap.max() <= reach * max(hgt, 4.0):
                lo, hi = np.minimum(lo, a), np.maximum(hi, b)
                picked.append(t)
                rest.remove(t)
                grown = True
    return lo, hi, picked


def cut(img, lo, hi, picked, pad=0.3):
    """The block cut from the OCR input with a margin; text running down the input is turned to run across (which of
    the two ways is upright is chosen for display by upright())."""
    m = max(3, int(round(pad * float(np.median([_thick(t) for t in picked])))))
    x0, y0 = np.maximum(np.floor(lo).astype(int) - m, 0)
    x1, y1 = np.minimum(np.ceil(hi).astype(int) + m, [img.shape[1], img.shape[0]])
    out = img[y0:y1, x0:x1].copy()
    if sum(_vertical(t) for t in picked) * 2 > len(picked):
        out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)
    return out


def best_block(seen, want, extra=None):
    """The best-read block of tokens satisfying want(text) over all OCR inputs (score: confidence x characters; a
    token running down its input counts 0.9, so the same print read across is preferred), with the tokens satisfying
    extra(text) that lie next to it -> (crop, text, token texts) or (None, "", [])."""
    best = None
    weight = lambda t: t.conf * len(t.text) * (0.9 if _vertical(t) else 1.0)  # noqa: E731
    for img, toks in seen:
        toks = [t for t in toks or [] if t.poly is not None and t.conf >= MIN_CONF]
        hits = [t for t in toks if want(t.text)]
        if not hits:
            continue
        start = max(hits, key=weight)
        pool = hits + ([t for t in toks if extra(t.text) and t not in hits] if extra else [])
        lo, hi, picked = cluster(pool, start)
        score = sum(weight(t) for t in picked if want(t.text))
        if best is None or score > best[0]:
            down = sum(_vertical(t) for t in picked) * 2 > len(picked)
            key = (lambda t: (round(float(_rect(t)[0][0]) / 12), float(_rect(t)[0][1]))) if down else                 (lambda t: (round(float(_rect(t)[0][1]) / 12), float(_rect(t)[0][0])))
            order = sorted(picked, key=key)
            best = (score, cut(img, lo, hi, picked), " ".join(t.text for t in order), [t.text for t in order])
    return best[1:] if best else (None, "", [])


def stack(blocks, gap=6):
    """Crops one above the other, centred, on a light background."""
    w = max(b.shape[1] for b in blocks)
    pad = [cv2.copyMakeBorder(b, 0, gap, (w - b.shape[1]) // 2, w - b.shape[1] - (w - b.shape[1]) // 2,
                              cv2.BORDER_CONSTANT, value=(242, 242, 242)) for b in blocks]
    return np.vstack(pad)[:-gap]


def shows(text, req):
    """What a token shows of the detail: "d" / "l" (the requested diameter / length as a size print), "ref", "gtin" (a
    GTIN with a valid check digit in the printed text)."""
    out = set()
    if find_gtins_in_text(text):
        out.add("gtin")
    if is_size(text, req):
        nums = numbers(text)
        out |= {k for k, v in (("d", req.diameter), ("l", req.length)) if v in nums}
    if is_ref(text, req):
        out.add("ref")
    return out


def detail_blocks(seen, req):
    """The detail as blocks of print: the block that shows most of what is still missing (diameter, length, REF, GTIN
    text), then the next, at most four; a block whose items the others already show is dropped; the size first ->
    [(crop, text)]. A product without a size: the printed date read."""
    if req.diameter is None:
        img, text, _ = best_block(seen, lambda s: bool(find_dates(s)))
        return [(img, text)] if img is not None else []
    missing, out = {"d", "l", "ref", "gtin"}, []
    while missing and len(out) < 4:
        need = set(missing)
        img, text, words = best_block(seen, lambda s: bool(shows(s, req) & need), lambda s: bool(shows(s, req)))
        got = set().union(*(shows(w, req) for w in words)) if words else set()
        if img is None or not got & missing:
            break
        out.append((img, text, got))
        missing -= got
    for b in sorted(out, key=lambda b: len(b[2])):          # redundant blocks out, the one showing least first
        if len(out) > 1 and b[2] <= set().union(*(o[2] for o in out if o is not b)):
            out.remove(b)
    out.sort(key=lambda b: (not b[2] & {"d", "l"}, "gtin" in b[2]))   # the size first, then the REF, the GTIN last
    return [(img, text) for img, text, _ in out]


def text_parts(seen, req):
    """Brand and detail from the OCR record of the MATCH outline (no OCR here)."""
    brand, brand_text, _ = best_block(seen, lambda s: is_name(s, req.product), lambda s: is_maker(s, req.product))
    if brand is None:
        brand, brand_text, _ = best_block(seen, lambda s: is_maker(s, req.product))
    blocks = detail_blocks(seen, req)
    read = [[k, t.text, round(float(t.conf), 3)] for k, (_, toks) in enumerate(seen) for t in toks or []
            if t.conf >= MIN_CONF]
    return {"brand": brand, "brand_text": brand_text, "detail_blocks": [b for b, _ in blocks],
            "detail_texts": [t for _, t in blocks],
            "detail_text": " | ".join(t for _, t in blocks), "read": read}


def init_worker(code_dir):
    """The crop process (aipc_tasks.py): the repository's src/ importable."""
    sys.path.insert(0, code_dir)
    os.chdir(code_dir)


def spine_of(frame, poly):
    """The box straightened and upright at the photo's full resolution -> (image, 3x3 transform photo -> image)."""
    return rectify(frame, np.asarray(poly, np.float32))


def find_bars(spine):
    """The bars of a 1-D barcode in the straightened box (src/box_crops.find_barcode, as the laptop's PICK panel) ->
    (corners in spine px or None, bars). aipc_tasks.py runs this in its own process, beside the drawn outline."""
    corners, bars = find_barcode(spine)
    return (None if corners is None else np.asarray(corners, np.float32)), int(bars or 0)


def barcode_from(spine, m, corners, bars):
    """The located barcode, bars only and straightened, and its corners in photo px. Located, never decoded (decoding
    stays off as in the production config): a position for the scanner, not evidence."""
    if corners is None:
        return {"barcode": None, "barcode_px": None, "barcode_bars": int(bars), "spine": spine}
    return {"barcode": warp_quad(spine, corners), "barcode_px": to_frame(order_quad(corners), m).round(1).tolist(),
            "barcode_bars": int(bars), "spine": spine}


def barcode_part(frame, poly):
    """spine_of + find_bars + barcode_from in one call (= src/box_crops.pick_crops with decode=None)."""
    spine, m = spine_of(frame, poly)
    return barcode_from(spine, m, *find_bars(spine))


def parts(frame, poly, seen, req):
    """The crop task on the PICKed box -> dict of images and texts (see the module doc). aipc_tasks.py runs the two
    halves at the same time (the barcode in its own process, beside the drawn outline)."""
    return {**text_parts(seen, req), **barcode_part(frame, poly)}


def _alnum(s):
    return re.sub(r"[^0-9a-z]", "", (s or "").lower())


def _matches(tokens, expect):
    """How many tokens of `expect` (the text the run read there; letters and digits, 2+ characters) the reading holds,
    looked for in the whole reading joined (one reading can split or join tokens: "2.75 33" vs "2.75", "33")."""
    joined = "".join(_alnum(t.text) for t in tokens)
    return sum(1 for w in {_alnum(x) for x in (expect or "").split()} if len(w) >= 2 and w in joined)


def upright(img, reader, expect=""):
    """Display only: the crop or its 180 degree turn, decided in this order, the crop as it is winning a final tie:
    1. read with the OCR's own direction classifier off, the turn whose reading holds more of the tokens the reader read
       there in the run (`expect`): upside down, most print reads as other strings ("19" -> "61", "REF" -> garbage);
    2. RapidOCR's direction classifier on the lines of print found in the crop that hold the run's tokens (else on
       every line), weighted by their characters (blocks that read the same either way, "2.75 33");
    3. without that classifier: the turn that reads the same with the classifier on and off ("mm" -> "ww").
    Each signal alone failed on some 8 Nov crops: confidence, closeness of the whole string, the on / off rule, the
    classifier on GTIN and small digit lines."""
    if img is None or reader is None or min(img.shape[:2]) < 8:
        return img, False
    turned = cv2.rotate(img, cv2.ROTATE_180)
    as_is = {k: reader.read_as_is(x).tokens for k, x in (("img", img), ("turned", turned))}
    m = {k: _matches(v, expect) for k, v in as_is.items()}
    if m["turned"] != m["img"]:
        return (turned, True) if m["turned"] > m["img"] else (img, False)
    cls = getattr(getattr(reader, "_engine", None), "text_cls", None)
    if cls is not None:
        want = {_alnum(x) for x in (expect or "").split()}

        def upside(x):
            """The classifier's weighted vote that x is upside down (+) or upright (-)."""
            lines, weights = [], []
            toks = [t for t in reader.read_once(x).tokens if t.poly is not None]
            own = [t for t in toks if _alnum(t.text) in want]
            for t in own or toks:      # the lines of the run's own tokens vote, if any (a block can hold print set the
                q = np.asarray(t.poly, np.float32).reshape(-1, 2)   # other way: "40" above "BALLOON", Essential Pro)
                x0, y0 = np.maximum(np.floor(q.min(0)).astype(int) - 2, 0)
                x1, y1 = np.minimum(np.ceil(q.max(0)).astype(int) + 2, [x.shape[1], x.shape[0]])
                c = x[y0:y1, x0:x1]
                if c.size == 0 or min(c.shape[:2]) < 4:
                    continue
                if c.shape[0] >= 1.5 * c.shape[1]:              # a tall line is turned as RapidOCR turns it
                    c = np.ascontiguousarray(np.rot90(c))
                lines.append(c)
                weights.append(len(t.text.strip()))
            if not lines:
                lines, weights = [x], [1]
            return sum(w * float(sc) * (1 if lab == "180" else -1) for w, (lab, sc) in zip(weights, cls(lines).cls_res))
        # both turns are asked and compared, so the answer does not depend on which turn came in (a lone "4" was
        # "180" both ways, 8 Nov)
        return (turned, True) if upside(turned) < upside(img) else (img, False)
    from collections import Counter

    def differ(k, x):
        off = Counter(_alnum(t.text) for t in as_is[k])
        on = Counter(_alnum(t.text) for t in reader.read_once(x).tokens)
        return sum(((off - on) + (on - off)).values())
    d_img, d_turned = differ("img", img), differ("turned", turned)
    return (turned, True) if d_turned < d_img else (img, False)


def save(folder, p, reader=None):
    """After the clock: the crops as PNG files (brand / detail turned upright for display) and their texts -> dict for
    the run log."""
    os.makedirs(folder, exist_ok=True)
    info = {"brand_text": p["brand_text"], "detail_text": p["detail_text"],
            "barcode": "located (not decoded)" if p["barcode"] is not None else "not found",
            "barcode_px": p["barcode_px"], "barcode_bars": p["barcode_bars"], "files": {},
            "read": p.get("read", [])}               # every token (>= 0.5) the MATCH outline's OCR read: [input, text, conf]
    blocks = []
    texts = p.get("detail_texts") or [""] * len(p.get("detail_blocks") or [])
    for k, (b, t) in enumerate(zip(p.get("detail_blocks") or [], texts), 1):   # each block of print upright on its own
        img, turned = upright(b, reader, t)
        blocks.append(img)
        info["detail_turned"] = info.get("detail_turned", False) or turned
        cv2.imwrite(os.path.join(folder, f"detail_{k}.png"), img)
    imgs = {"brand": p.get("brand"), "detail": stack(blocks) if blocks else None, "barcode": p.get("barcode"),
            "spine": p.get("spine")}
    for k, img in imgs.items():
        if img is None or not img.size:
            continue
        if k == "brand":
            img, info["brand_turned"] = upright(img, reader, p["brand_text"])
        cv2.imwrite(os.path.join(folder, k + ".png"), img)
        info["files"][k] = k + ".png"
    with open(os.path.join(folder, "crops.json"), "w") as f:
        json.dump(info, f, indent=1)
    return info

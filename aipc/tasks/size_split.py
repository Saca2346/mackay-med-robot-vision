"""AI PC parser option (7 Nov 2026, `--split-sizes 1`): a printed size whose two numbers the OCR joined into one token.

On many labels the diameter and the length sit side by side ("3.5 | 33", "3.0 | 19"); the OCR line finder often returns
them as ONE token "3.533", "3.019", "2.519". The laptop parser (src/box_verifier.parse_identity) drops any number with
more than two decimals, so the printed size never counts as a source. This option splits such a token before the
laptop parser sees it, only where exactly one split is a valid size:
  d.X + YZ   X in {0, 5} (one-decimal diameter), YZ a length 6-99 without a leading 0      "3.533" -> "3.5 33"
  d.XY + Z.. XY in {00, 25, 50, 75} (two-decimal diameter), the rest a length 6-200          "2.7520" -> "2.75 20"
Both splits valid at once cannot happen (shown in the docstring test below). The OCR's own confidence is kept: the
digits were read; only the space between the two printed fields was lost. Nothing else changes: the text source still
needs product + diameter + length, and green still needs a second, independent source (REF, GTIN text).
src/ (the laptop) is not changed; the laptop keeps its parser until the owner decides.

>>> split_text("BCR-3533 3.533 CHROMA 2.519 2.7520 3.0 2028-04-20 4.00 H26040015")
'BCR-3533 3.5 33 CHROMA 2.5 19 2.75 20 3.0 2028-04-20 4.00 H26040015'
"""
import re

_TOK = re.compile(r"(?<![\w.,])(\d)[.,](\d+)(?![\w.,])")
QUARTER = {"00", "25", "50", "75"}


def _split(m):
    a, frac = m.group(1), m.group(2)
    if len(frac) <= 2:
        return m.group(0)
    out = []
    if frac[0] in "05" and len(frac) == 3 and frac[1] != "0" and 6 <= int(frac[1:]) <= 99:
        out.append(f"{a}.{frac[0]} {frac[1:]}")
    if frac[:2] in QUARTER and frac[2] != "0" and 6 <= int(frac[2:]) <= 200:
        out.append(f"{a}.{frac[:2]} {frac[2:]}")
    return out[0] if len(out) == 1 else m.group(0)


def split_text(text):
    return _TOK.sub(_split, text)


def install():
    """Wrap parse_identity wherever the laptop modules imported it (tokens re-made with the split text)."""
    import src.box_evidence as be
    import src.box_scan as bs
    import src.box_spine_read as bsr
    import src.box_verifier as bv
    from src.box_ocr import OcrToken
    orig = bv.parse_identity

    def parse_identity(tokens, *a, **k):
        new = []
        for t in tokens:
            s = split_text(t.text)
            # a joined size becomes its own tokens ("3.0", "19"), so the laptop's symbol rule sees "8" "3.0" as "⌀ 3.0"
            new += [t] if s == t.text else [OcrToken(w, t.conf) for w in s.split()]
        return orig(new, *a, **k)
    for mod in (bv, be, bs, bsr):
        if getattr(mod, "parse_identity", None) is orig:
            mod.parse_identity = parse_identity
    return parse_identity


def _split_token(t):
    """A joined size token ("3.019") as two tokens ("3.0", "19"), each with its share of the token's outline (by
    characters along x; the height, which the size-field rule uses, is unchanged)."""
    s = split_text(t.text)
    if s == t.text or t.poly is None:
        return [t]
    from dataclasses import replace
    import numpy as np
    words, n = s.split(), len(t.text)
    x0, x1 = float(t.poly[:, 0].min()), float(t.poly[:, 0].max())
    y0, y1 = float(t.poly[:, 1].min()), float(t.poly[:, 1].max())
    out, at = [], 0
    for w in words:
        a, b = x0 + (x1 - x0) * at / n, x0 + (x1 - x0) * (at + len(w)) / n
        out.append(replace(t, text=w, poly=np.array([[a, y0], [b, y0], [b, y1], [a, y1]], dtype=float)))
        at += len(w)
    return out


def install_spine_size(cfg):
    """`--spine-size 1` (7 Nov): when a box has no detected size label, the laptop reads its whole spine only for the
    product name, REF and GTIN text; the printed size on the spine is not used. This option parses that same spine read
    (the laptop parser, with the split above) and, when it gives ONE diameter and ONE length at full confidence and
    the same product as the text source, fills the missing diameter / length of the text source. A value the text
    source already read is never replaced; a disagreeing spine size makes the text source unsure (not a match). The
    REF and GTIN text found on the spine stay separate sources; green still needs two of them to agree."""
    import threading
    import src.box_evidence as be
    import src.box_visual_read as bvr
    from src.box_verifier import Identity
    from src.box_spine_read import column_tokens, size_field_tokens
    local = threading.local()
    orig_tokens, orig_full, orig_tiles = be._spine_tokens, be.gather_full_box, be._tiles

    def _tiles(img, *a, **k):
        local.h = img.shape[0]                  # height of the (enlarged) spine strip the tokens are measured in
        return orig_tiles(img, *a, **k)

    def _spine_tokens(img, reader, cfg_=None):
        toks, n = orig_tokens(img, reader, cfg_)
        h = getattr(local, "h", None)
        if h and toks and all(t.poly is not None for t in toks):
            # the laptop's own size-field rule (src/box_spine_read.size_field_tokens): only numerals at least 22% of
            # the strip height (the printed size), not LOT, GTIN, date or icon digits; words without digits for the name
            words = [t for t in toks if not any(ch.isdigit() for ch in t.text)]
            parts = [p for t in toks for p in _split_token(t)]
            local.toks = getattr(local, "toks", []) + words + column_tokens(size_field_tokens(parts, h), h)
        return toks, n

    def gather_full_box(ev, box_crop, reader, cfg_=None, product=None):
        local.toks = []
        out = orig_full(ev, box_crop, reader, cfg_, product)
        t = ev.text
        if t is None or not t.product_ok or (t.diameter_ok and t.length_ok) or not local.toks:
            return out
        sp = be.parse_identity(local.toks, cfg["catalog"], cfg["ocr"]["min_confidence"],
                               tuple(cfg["matching"]["diameter_range_mm"]), tuple(cfg["matching"]["length_range_mm"]))
        if not (sp.product_ok and sp.product == t.product and sp.diameter_ok and sp.length_ok):
            t.notes.append("spine size incomplete (AI PC): " + "; ".join(sp.notes)[:80])
            return out
        clash = (t.diameter_ok and abs(t.diameter - sp.diameter) > 1e-6) or (t.length_ok and abs(t.length - sp.length) > 1e-6)
        if clash:
            t.notes.append("spine size disagrees (AI PC)")
            return out
        t.diameter, t.diameter_ok, t.length, t.length_ok = sp.diameter, True, sp.length, True
        t.notes.append("size from the spine (AI PC)")
        return out
    be._spine_tokens = _spine_tokens
    be._tiles = _tiles
    be.gather_full_box = gather_full_box
    if getattr(bvr, "gather_full_box", None) is orig_full:
        bvr.gather_full_box = gather_full_box


if __name__ == "__main__":
    import doctest
    print(doctest.testmod())
    for s in ["3.533", "3.019", "2.519", "2.758", "3.520", "2.2512", "3.025", "4.020", "3.06", "2.75", "3.0E", "1.200", "3.506"]:
        print(s, "->", split_text(s))

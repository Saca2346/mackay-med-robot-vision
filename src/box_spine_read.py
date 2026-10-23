"""Read actual printed name/REF and size regions on supported spine layouts.

Positions select pixels only. Values, units and confidence always come from OCR;
request values never fill fields. Unreadable layouts fall back to the full reader.
"""
from __future__ import annotations

import re
import time
from dataclasses import replace

import numpy as np

from src.box_evidence import redecide
from src.box_verifier import Evidence, parse_identity


def column_tokens(tokens, height):
    """Keep a numeral with the unit printed below it, using actual token positions."""
    columns = []
    for token in sorted(tokens, key=lambda t: float(t.poly[:, 0].mean()) if t.poly is not None else -1):
        if token.poly is None:
            return tokens  # no fabricated geometry/order
        x = float(token.poly[:, 0].mean())
        column = next((c for c in columns if abs(c[0] - x) <= max(8, height * .4)), None)
        if column is None:
            columns.append([x, [token]])
        else:
            column[1].append(token)
    return [t for _, col in columns for t in sorted(col, key=lambda t: float(t.poly[:, 1].mean()))]


def size_field_tokens(tokens, height):
    """Large printed size numerals and their units; exclude tiny LOT/date fragments."""
    if any(t.poly is None for t in tokens):
        return tokens
    selected = [t for t in tokens if re.fullmatch(r"mm", t.text.strip(), re.I)
            or (re.fullmatch(r"\d{1,2}(?:[.,]\d{1,2})?(?:\s*mm)?", t.text.strip(), re.I)
                and float(t.poly[:, 1].max() - t.poly[:, 1].min()) >= height * .22)]
    # A date strip beside the size columns can contain fairly tall "20"/"25"
    # fragments. When two actual unit/number pairs define the printed layout,
    # retain those pairs instead of allowing nearby unpaired dates to become
    # competing length readings. No unit or requested value is synthesized.
    units = [t for t in selected if re.fullmatch(r"mm", t.text.strip(), re.I) and t.conf >= .9]
    paired, used, merged = [], set(), 0
    unit_ids = {id(t) for t in units}
    for token in selected:
        if id(token) in unit_ids or token.conf < .9:
            continue
        if re.fullmatch(r"\d{1,2}(?:[.,]\d{1,2})?\s*mm", token.text.strip(), re.I):
            paired.append(token)
            merged += 1
            continue
        center = token.poly.mean(0)
        width = float(np.ptp(token.poly[:, 0]))
        for i, unit in enumerate(units):
            uc = unit.poly.mean(0)
            uw = float(np.ptp(unit.poly[:, 0]))
            below = (abs(float(uc[0]-center[0])) <= max(8, (width+uw)*.45)
                     and -height*.08 <= float(uc[1]-center[1]) <= height*.55)
            inline = (0 <= float(unit.poly[:, 0].min()-token.poly[:, 0].max()) <= height*.25
                      and abs(float(uc[1]-center[1])) <= height*.16)
            if below or inline:
                paired.append(token)
                used.add(i)
                break
    if (len(paired) >= 2 and len(used) + merged >= 2
            and float(np.ptp([t.poly[:, 0].mean() for t in paired])) >= height * .4):
        accepted = {id(t) for t in paired} | {id(units[i]) for i in used}
        return [t for t in selected if id(t) in accepted]
    return selected


def read_label_regions(labels, box, reader, cfg, request, catalog=None):
    """Use detected label geometry for sizes; read a small name patch only if missing."""
    import cv2
    from src.box_barcode import decode_gs1, find_gtins_in_text
    from src.box_expiry import find_dates
    from src.box_ref import parse_refs, parse_ref_hints
    from src.box_verifier import _find_product
    if not labels:
        return None
    started = time.perf_counter()
    raw, numeric, bands = [], [], []
    for index, label in enumerate(labels):
        image = cv2.rotate(label, cv2.ROTATE_90_CLOCKWISE) if label.shape[0] > label.shape[1] else label
        found = reader.read_once(image).tokens
        selected = size_field_tokens(found, image.shape[0])
        raw.extend(found)
        numeric.extend(column_tokens(selected, image.shape[0]))
        bands.append({"name": f"size_{index+1}", "image": image, "tokens": found, "field_tokens": selected})
    text = " ".join(t.text for t in raw)
    refs = parse_refs(text)
    if not refs:
        return None
    ident = parse_identity(numeric, cfg["catalog"], cfg["ocr"]["min_confidence"],
                           tuple(cfg["matching"]["diameter_range_mm"]), tuple(cfg["matching"]["length_range_mm"]))
    product, ok, note = _find_product(raw, cfg["catalog"], cfg["ocr"]["min_confidence"])
    reads = len(labels)
    if not ok and box is not None:
        image = box[:, round(box.shape[1]*.16):round(box.shape[1]*.48)].copy()
        more = reader.read_once(image).tokens
        reads += 1
        bands.append({"name": "product_ref", "image": image, "tokens": more})
        raw.extend(more)
        text = " ".join(t.text for t in raw)
        product, ok, note = _find_product(raw, cfg["catalog"], cfg["ocr"]["min_confidence"])
    ident.product, ident.product_ok = product, ok
    ident.text = text
    if ok:
        ident.notes = [n for n in ident.notes if n != "product not read"]
    ev = Evidence(text=ident, refs=parse_refs(text), ref_hints=parse_ref_hints(text),
                  codes=(decode_gs1(box, hard=False) if box is not None
                         and cfg.get("verify", {}).get("decode_barcodes", True) else []) + find_gtins_in_text(text),
                  dates=tuple(find_dates(text)))
    ev.band_read = True
    decision, reason, key, ev.support = redecide(ev, request, cfg, catalog)
    return (ev, decision, reason, key, (time.perf_counter()-started)*1000, reads), bands


def read_spine_bands(box, reader, cfg, request, catalog=None):
    """Return evidence + raw band inputs, or None when this printed layout is not readable."""
    from src.box_barcode import decode_gs1, find_gtins_in_text
    import cv2
    from src.box_ref import parse_refs, parse_ref_hints
    from src.box_expiry import find_dates
    started = time.perf_counter()
    if box is None or box.shape[1] < box.shape[0] * 5:
        return None
    h, w = box.shape[:2]
    product = box[:, round(w * .16):round(w * .48)].copy()
    size = box[:, round(w * .48):round(w * .73)].copy()
    joint_tokens = None
    if cfg.get("visual", {}).get("joint_spine_read", False):
        # One text-detector invocation for two adjacent printed regions. Geometry
        # assigns observations to regions; no request value supplies any token.
        joint = box[:, round(w * .16):round(w * .73)].copy()
        split = round(w * .48) - round(w * .16)
        found = reader.read_once(joint).tokens
        if found and all(t.poly is not None for t in found):
            name_tokens = [t for t in found if float(t.poly[:, 0].max()) <= split]
            joint_tokens = [t for t in found if float(t.poly[:, 0].min()) >= split]
            joint_band = {"name": "size_joint", "image": joint, "tokens": found}
    if joint_tokens is None:
        name_tokens = reader.read_once(product).tokens
    refs = parse_refs(" ".join(t.text for t in name_tokens))
    from src.box_verifier import _find_product
    product_name, product_ok, _ = _find_product(name_tokens, cfg["catalog"], cfg["ocr"]["min_confidence"])
    if not refs and not (product_ok and product_name == request.product):
        return None
    if request.diameter is not None and any(r.product != request.product or r.diameter != request.diameter
                                            or r.length != request.length for r in refs):
        # A conflicting exact REF cannot authorize a match. Do not spend another
        # OCR pass on this sibling; repeated exclusions still use the normal tracker.
        text = " ".join(t.text for t in name_tokens)
        ident = parse_identity(name_tokens, cfg["catalog"], cfg["ocr"]["min_confidence"],
                               tuple(cfg["matching"]["diameter_range_mm"]), tuple(cfg["matching"]["length_range_mm"]))
        ev = Evidence(text=ident, refs=refs)
        ev.band_read = True
        decision, reason, key, ev.support = redecide(ev, request, cfg, catalog)
        return ((ev, decision, reason, key, (time.perf_counter()-started)*1000, 1),
                [joint_band] if joint_tokens is not None else
                [{"name": "product_ref", "image": product, "tokens": name_tokens}])
    size_tokens = joint_tokens if joint_tokens is not None else reader.read_once(size).tokens
    field_tokens = size_field_tokens(size_tokens, h)
    tokens = name_tokens + column_tokens(field_tokens, h)
    text = " ".join(t.text for t in tokens)
    ident = parse_identity(tokens, cfg["catalog"], cfg["ocr"]["min_confidence"],
                           tuple(cfg["matching"]["diameter_range_mm"]), tuple(cfg["matching"]["length_range_mm"]))
    retry_bands = []
    merged = any(re.fullmatch(r"\d{1,2}[.,]\d{3,}", t.text.strip()) and t.poly is not None
                 and np.isfinite(t.poly).all() for t in size_tokens)
    uncertain_size = [t for t in size_tokens if re.fullmatch(r"\d{1,2}(?:[.,]\d{1,2})?", t.text.strip())
                         and .75 <= t.conf < cfg["ocr"]["min_confidence"]
                         and t.poly is not None and np.isfinite(t.poly).all()
                         and float(np.ptp(t.poly[:, 1])) >= h * .22]
    if not cfg.get("visual", {}).get("size_token_retry", True):
        uncertain_size = []
    if (request.product == "angiolite" and request.diameter is not None
            and (merged or uncertain_size)
            and cfg.get("visual", {}).get("size_column_retry", True)):
        # OCR can merge the two printed columns into "2.529". Re-read actual
        # column pixels. An uncertain individual numeral instead gets one tight
        # pixel crop, avoiding two ineffective whole-column passes. Values and
        # acceptance thresholds never come from the request or REF.
        offset = split if joint_tokens is not None else 0
        patches = []
        if merged:
            for name, start, end in [("left", 0, size.shape[1] // 2),
                                     ("right", size.shape[1] // 2, size.shape[1])]:
                patches.append((name, size[:, start:end].copy(), [start + offset, 0]))
        else:
            source = joint if joint_tokens is not None else size
            for index, token in enumerate(uncertain_size[:2], 1):
                x1, y1 = np.floor(token.poly.min(0)).astype(int)
                x2, y2 = np.ceil(token.poly.max(0)).astype(int)
                x1, y1 = max(0, x1-3), max(0, y1-2)
                x2, y2 = min(source.shape[1], x2+4), min(h, y2+3)
                if x2 <= x1 or y2 <= y1:
                    continue
                patch = cv2.copyMakeBorder(source[y1:y2, x1:x2], 4, 4, 4, 4,
                                           cv2.BORDER_CONSTANT, value=(255, 255, 255))
                patches.append((f"token_{index}", patch, [x1-4, y1-4]))
        for name, patch, translation in patches:
            if not patch.size:
                continue
            found = reader.read_once(patch).tokens
            selected = size_field_tokens(found, patch.shape[0])
            field_tokens += [replace(t, poly=t.poly + translation)
                             if t.poly is not None else t for t in selected]
            retry_bands.append({"name": "size_retry_" + name, "image": patch,
                                "tokens": found, "field_tokens": selected})
        tokens = name_tokens + column_tokens(field_tokens, h)
        text = " ".join(t.text for t in tokens)
        ident = parse_identity(tokens, cfg["catalog"], cfg["ocr"]["min_confidence"],
                               tuple(cfg["matching"]["diameter_range_mm"]),
                               tuple(cfg["matching"]["length_range_mm"]))
    partial_ref = (cfg.get("visual", {}).get("partial_spine_read", False)
                   and any(r.product == request.product for r in parse_refs(text)))
    # Keep a real partial REF reading instead of repeating the full reader now.
    # The unchanged track gate combines it with other observations and still
    # requires a complete independent printed-size source before green.
    if not partial_ref and (not ident.product_ok or not (ident.diameter_ok or ident.length_ok
                                                        or ident.diameter_hint is not None)):
        return None
    ev = Evidence(text=ident, refs=parse_refs(text), ref_hints=parse_ref_hints(text), dates=tuple(find_dates(text)),
                  codes=(decode_gs1(box, hard=False) if cfg.get("verify", {}).get("decode_barcodes", True)
                         else []) + find_gtins_in_text(text))
    ev.band_read = True
    decision, reason, key, ev.support = redecide(ev, request, cfg, catalog)
    if joint_tokens is not None:
        joint_band["field_tokens"] = field_tokens
        bands = [joint_band]
    else:
        bands = [{"name": "product_ref", "image": product, "tokens": name_tokens},
                 {"name": "size", "image": size, "tokens": size_tokens, "field_tokens": field_tokens}]
    bands += retry_bands
    return (ev, decision, reason, key, (time.perf_counter() - started) * 1000,
            (1 if joint_tokens is not None else 2) + len(retry_bands)), bands

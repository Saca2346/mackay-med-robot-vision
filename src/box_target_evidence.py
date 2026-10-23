"""Observed, same-track verification evidence; never inventory-derived field values."""
from __future__ import annotations

from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import re
import time
from types import SimpleNamespace

import cv2
import numpy as np

from src.box_crops import barcode_with_text, best_date_crop, numeral_crop, pick_number_tiles, upright_date
from src.box_evidence import bright_label_crops, read_evidence, redecide


def date_region_retry(image, tokens, reader):
    """Retry observed date-shaped token pixels, never a request-derived location/value."""
    bands = []
    for token in tokens:
        if (token.poly is None or not re.match(r"^[2Z][0O][\dOoIlDQZSB|./ -]{3,}$", token.text.strip())
                or not any(c in token.text for c in "-/ .")):
            continue
        p = np.asarray(token.poly, np.float32)
        if not np.isfinite(p).all():
            continue
        lo, hi = p.min(0), p.max(0)
        margin = max(8, round(float(hi[1]-lo[1])*.45))
        x0, y0 = np.maximum(0, np.floor(lo-margin).astype(int))
        x1, y1 = np.minimum([image.shape[1], image.shape[0]], np.ceil(hi+margin).astype(int))
        patch = image[y0:y1, x0:x1]
        if not patch.size:
            continue
        patch = cv2.rotate(patch, cv2.ROTATE_180)
        bands.append({"name": f"date_retry_{len(bands)+1}", "image": patch,
                      "tokens": reader.read_once(patch).tokens})
        if len(bands) == 2:
            break
    return bands


_PANEL_READER = []


def panel_reader():
    """One small OCR engine of the evidence thread, made on first use (it only turns display crops upright)."""
    if not _PANEL_READER:
        from src.box_ocr import SizeTextReader
        try:
            _PANEL_READER.append(SizeTextReader(threads=1))
        except Exception:  # noqa: BLE001
            _PANEL_READER.append(None)
    return _PANEL_READER[0]


class EvidenceWriter:
    """One frozen export at a time; disk and panel work never wait in the view loop.

    The caller retries the newest state when busy. Results retain their original
    context key so a completed old export cannot populate another target's panel.
    """
    def __init__(self):
        self.executor = None
        self.pending = None

    def submit(self, context, *args):
        if self.pending is not None:
            return False
        frozen = deepcopy(args)
        if self.executor is None:
            self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="target-evidence")
        self.pending = (context, self.executor.submit(save_evidence, *frozen))
        return True

    def drain(self, wait=False):
        if self.pending is None:
            return None
        context, future = self.pending
        if not wait and not future.done():
            return None
        self.pending = None
        return context, future.result()  # write errors must fail the session

    def close(self):
        if self.executor is not None:
            self.executor.shutdown(wait=True)
        return self.drain(wait=True)


def read_target_evidence(sl_crops, box_crop, too_far, reader, cfg, req, catalog=None, *, attempt=0,
                         identity_verified=False):
    """Keep the actual OCR input alongside its result, not a later camera frame."""
    from src.box_barcode import decode_gs1, find_gtins_in_text
    from src.box_expiry import find_dates
    from src.box_ref import parse_refs, parse_ref_hints
    from src.box_verifier import Evidence, parse_identity, combined_score
    if identity_verified:
        # After acquisition, search for full current observations/details instead
        # of filling the rolling history with intentionally partial fast reads.
        cfg = dict(cfg, visual={**cfg.get("visual", {}), "partial_spine_read": False})
    started = time.perf_counter()
    result = None
    numbers = []
    bands = []
    decode_barcodes = cfg.get("verify", {}).get("decode_barcodes", True)
    from src.box_verifier import NO_SIZE
    if (cfg.get("visual", {}).get("fast_no_size_spine", True)
            and req.product in NO_SIZE and req.diameter is None and req.length is None
            and not identity_verified and not too_far and box_crop is not None
            and attempt % 4 != 3):
        # A product without size fields does not need four label orientations
        # followed by tiled full-spine OCR. Read every pixel of the same spine
        # once; its name/date tokens and decoded code still use the normal gate.
        # Never use this acquisition path for products requiring exact sizes.
        tokens = reader.read_once(box_crop).tokens
        identity_bands = [{"name": "spine_identity", "image": box_crop, "tokens": tokens}]
        if (cfg.get("visual", {}).get("date_region_retry", True) and not decode_barcodes and req.expiry is not None
                and not find_dates(" ".join(t.text for t in tokens if t.conf >= cfg["ocr"]["min_confidence"]))):
            retries = date_region_retry(box_crop, tokens, reader)
            identity_bands.extend(retries)
            tokens = tokens + [t for band in retries for t in band["tokens"]]
        if (not decode_barcodes and req.expiry is not None
                and not find_dates(" ".join(t.text for t in tokens if t.conf >= cfg["ocr"]["min_confidence"]))):
            # With no decoded code date available, do not wait until the fourth
            # attempt for an unreadable printed date's other orientation. Both
            # orientations remain one observation of one frame, not two sources.
            flipped = cv2.rotate(box_crop, cv2.ROTATE_180)
            more = reader.read_once(flipped).tokens
            identity_bands.append({"name": "spine_identity_flipped", "image": flipped, "tokens": more})
            tokens = tokens + more
        ident = parse_identity(tokens, cfg["catalog"], cfg["ocr"]["min_confidence"],
                               tuple(cfg["matching"]["diameter_range_mm"]),
                               tuple(cfg["matching"]["length_range_mm"]))
        if ident.product_ok:
            text = " ".join(t.text for t in tokens)
            dates_text = " ".join(t.text for t in tokens if t.conf >= cfg["ocr"]["min_confidence"])
            ev = Evidence(text=ident, refs=parse_refs(text), ref_hints=parse_ref_hints(text),
                          codes=(decode_gs1(box_crop, hard=True) if decode_barcodes else []) + find_gtins_in_text(text),
                          dates=tuple(find_dates(dates_text)), spine=text)
            ev.band_read = True
            decision, reason, key, ev.support = redecide(ev, req, cfg, catalog)
            result = (ev, decision, reason, key, (time.perf_counter()-started)*1000, len(identity_bands))
            bands = identity_bands
    fast_products = cfg.get("visual", {}).get("fast_spine_products", ["angiolite"])
    if (cfg.get("verify", {}).get("spine_bands", False) and req.product in fast_products
            and req.expiry is None and not too_far and attempt % 4 != 3):
        from src.box_spine_read import read_spine_bands, read_label_regions
        fast_result = read_spine_bands(box_crop, reader, cfg, req, catalog)
        if fast_result is None:
            fast_result = read_label_regions(sl_crops, box_crop, reader, cfg, req, catalog)
        if fast_result is not None:
            result, bands = fast_result
            for band in bands:
                if not band["name"].startswith("size"):
                    continue
                for token in band.get("field_tokens", band["tokens"]):
                    if (token.poly is not None and token.conf >= cfg["ocr"]["min_confidence"]
                            and re.fullmatch(r"\d{1,2}(?:[.,]\d{1,2})?(?:\s*mm)?", token.text, re.I)):
                        image = numeral_crop(band["image"], token.poly)
                        if image.size:
                            numbers.append({"text": token.text, "confidence": token.conf, "image": image})
    # Angiolite's large numerals face across the spine in the current webcam layout.
    # Try this orientation first; it changes acquisition order, never confidence/scoring.
    if result is None and req.product == "angiolite" and sl_crops and not too_far and attempt % 4 != 3:
        tokens = []
        for crop in sl_crops:
            upright = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE) if crop.shape[0] > crop.shape[1] else crop
            found = reader.read_once(upright).tokens
            tokens += found
            for token in found:
                if (token.poly is not None and token.conf >= cfg["ocr"]["min_confidence"]
                        and re.fullmatch(r"\d{1,2}(?:[.,]\d{1,2})?(?:\s*mm)?", token.text, re.I)):
                    crop_number = numeral_crop(upright, token.poly)
                    if crop_number.size:
                        numbers.append({"text": token.text, "confidence": token.conf, "image": crop_number})
        text = " ".join(t.text for t in tokens)
        ident = parse_identity(tokens, cfg["catalog"], cfg["ocr"]["min_confidence"],
                               tuple(cfg["matching"]["diameter_range_mm"]), tuple(cfg["matching"]["length_range_mm"]))
        ev = Evidence(text=ident, refs=parse_refs(text), ref_hints=parse_ref_hints(text),
                      codes=(decode_gs1(box_crop, hard=False) if box_crop is not None and decode_barcodes else []) + find_gtins_in_text(text),
                      dates=tuple(find_dates(text)))
        ev.observed_brand = "iVascular" if any(t.text.lower() == "ivascular"
                                                and t.conf >= cfg["ocr"]["min_confidence"] for t in tokens) else None
        d, reason, key, ev.support = redecide(ev, req, cfg, catalog)
        # Unreadable or wrong-orientation crops use the original orientation/full-spine fallback.
        if combined_score(ev.support[0])[0] >= 1 or ev.support[1]:
            result = (ev, d, reason, key, (time.perf_counter()-started)*1000, len(sl_crops))
    if result is None:
        result = read_evidence(sl_crops, box_crop, too_far, reader, cfg, req, catalog)
        ev = result[0]
        ev.dates = tuple(sorted(set(ev.dates) | set(find_dates(ev.text.text if ev.text else ""))))
    result[0].capture = {"spine": box_crop, "labels": sl_crops, "numbers": numbers, "bands": bands,
                         "decode_barcodes": decode_barcodes, "date_crop": best_date_crop(bands)[0]}
    return result


read_target_evidence.uses_attempt = True


def observed_fields(state):
    """Nullable values and provenance. A REF-derived size is never a size-text read."""
    evs = [e for e in state.evs if e is not None]
    fields = {}
    def put(name, value, source):
        if value is not None and value != "":
            fields.setdefault(name, []).append({"value": str(value), "source": source})
    for ev in evs:
        t = ev.text
        if t:
            for name, value, ok in (("product", t.product, t.product_ok),
                                    ("diameter_mm", t.diameter, t.diameter_ok),
                                    ("length_mm", t.length, t.length_ok)):
                if ok:
                    put(name, value, "spine_band_ocr" if getattr(ev, "band_read", False) else "size_label_ocr")
        # Keep brand observational, rather than filling it from product/catalogue metadata.
        if getattr(ev, "observed_brand", None):
            put("brand", ev.observed_brand, "label_ocr")
        for ref in ev.refs:
            put("REF", ref.text, "ref_ocr")
            put("product", ref.product, "ref_parse")
            put("diameter_mm", ref.diameter, "ref_parse")
            put("length_mm", ref.length, "ref_parse")
        for code in ev.codes:
            put("GTIN", code.gtin, code.source)
            put("LOT", code.lot, code.source)
            put("expiry", code.expiry, code.source)
    # The tracker accepts printed expiry only after the existing date consistency gate.
    if state.expiry_src == "text" and state.expiry in {d for ev in evs for d in ev.dates}:
        put("expiry", state.expiry, "consistent_printed_date_ocr")
    out = {}
    for name in ("product", "brand", "diameter_mm", "length_mm", "REF", "GTIN", "LOT", "expiry"):
        rows = fields.get(name, [])
        values = sorted({r["value"] for r in rows})
        derived = rows and all(r["source"] == "ref_parse" for r in rows)
        out[name] = {"value": values[0] if len(values) == 1 else None,
                     "status": ("derived_from_REF" if derived else "read") if len(values) == 1 else "conflict" if values else "not_read",
                     "observations": [dict(value=v, source=s) for v, s in sorted({(r['value'], r['source']) for r in rows})]}
    return out


def save_evidence(root, run_id, req, tid, state, verified, seconds, time_basis):
    """Save actual reader crops and raw evidence; green panel only for a verified identity."""
    from src.box_crops import panel, pick_crops, rectify
    folder = root / f"{run_id}_{time.time_ns()}_track{tid}_{'verified' if verified else 'pending'}"
    folder.mkdir(parents=True, exist_ok=False)
    evidence = []
    last_crops = None
    for i, ev in enumerate(state.evs):
        capture = getattr(ev, "capture", None)
        if not capture or capture["spine"] is None:
            continue
        sub = folder / f"reading_{i + 1}"
        sub.mkdir()
        spine = capture["spine"]
        h, w = spine.shape[:2]
        group = SimpleNamespace(box=SimpleNamespace(poly=np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])), labels=[])
        barcode_enabled = capture.get("decode_barcodes", True)
        crops = (pick_crops(spine, group) if barcode_enabled else
                 {"spine": rectify(spine, group.box.poly)[0], "barcode": None,
                  "barcode_decoded": False, "barcode_text": "", "barcode_disabled": True})
        crops["size_labels"] = bright_label_crops(capture["labels"])    # not the dark TERUMO logo crop
        cv2.imwrite(str(sub / "spine_ocr_input.png"), spine)
        cv2.imwrite(str(sub / "spine.png"), crops["spine"])
        for j, lab in enumerate(capture["labels"], 1):
            cv2.imwrite(str(sub / f"size_label_{j}.png"), lab)
        number_rows = []
        band_rows = []
        for band in capture.get("bands", []):
            filename = "band_" + band["name"] + ".png"
            cv2.imwrite(str(sub / filename), band["image"])
            def token_row(t):
                return {"text": t.text, "confidence": t.conf,
                        "poly": t.poly.tolist() if t.poly is not None else None}
            band_rows.append({"file": filename, "tokens": [token_row(t) for t in band["tokens"]],
                              "field_tokens": [token_row(t) for t in band.get("field_tokens", [])]})
        for j, number in enumerate(capture.get("numbers", []), 1):
            filename = f"number_{j}.png"
            cv2.imwrite(str(sub / filename), number["image"])
            number_rows.append({"file": filename, "text": number["text"], "confidence": number["confidence"]})
        if crops["barcode"] is not None:
            cv2.imwrite(str(sub / "barcode_candidate.png"), crops["barcode"])
        # Gradient localization can mistake printed letters for bars. Only decoded
        # polygons are displayed as barcode crops in this evidence view.
        crops["barcode"] = None
        for code in ev.codes:
            if code.source == "barcode" and code.poly is not None:
                from src.box_crops import rectify
                crops["barcode"], _ = rectify(spine, code.poly)
                crops["barcode_decoded"], crops["barcode_text"] = True, code.raw
                cv2.imwrite(str(sub / "barcode_decoded.png"), crops["barcode"])
                break
        codes = [{k: v for k, v in asdict(c).items() if k != "poly"} for c in ev.codes]
        evidence.append({"directory": sub.name, "text": asdict(ev.text) if ev.text else None,
                         "refs": [asdict(r) for r in ev.refs], "codes": codes,
                         "number_crops": number_rows,
                         "band_reads": band_rows,
                         "spine_text": ev.spine, "printed_dates": ev.dates,
                         "observed_brand": getattr(ev, "observed_brand", None),
                         "support": sorted(map(str, ev.support[0])) if ev.support else [],
                         "conflicts": bool(ev.support and ev.support[1]),
                         "barcode_crop": "disabled" if not barcode_enabled else
                                         "decoded" if crops["barcode_decoded"] else "not_confirmed"})
        last_crops = crops
    fields = observed_fields(state)
    report = {"request": str(req), "track": tid, "epoch": state.epoch,
              "identity_verified": verified, "retrieval_authorized": False,
              "verified_wall_seconds": seconds, "time_basis": time_basis,
              "timer_origin": "first frame / latest request after model initialization",
              "sources": state.combined_sources(), "fields": fields,
              "all_fields_read": all(f["status"] == "read" for f in fields.values()),
              "barcode_decoded": any(c.source == "barcode" for e in state.evs if e for c in e.codes),
              "evidence": evidence}
    (folder / "result.json").write_text(json.dumps(report, default=str, indent=2), encoding="utf-8")
    summary = {"identity": "VERIFIED" if verified else "PENDING - not verified",
               "sources": state.combined_sources() or "none",
               **{k: (f["value"] + " (REF)" if f["status"] == "derived_from_REF" else f["value"] or f["status"])
                  for k, f in fields.items()}}
    # Enlarge actual numeral regions in the panel, while retaining full labels on disk.
    latest_numbers = next((getattr(e, "capture", {}).get("numbers", []) for e in reversed(state.evs)
                           if getattr(e, "capture", {}).get("numbers")), [])
    latest_date = next((getattr(e, "capture", {}).get("date_crop") for e in reversed(state.evs)
                        if getattr(e, "capture", {}).get("date_crop") is not None), None)
    size = {k: fields[k]["value"] for k in ("diameter_mm", "length_mm")}
    shown = pick_number_tiles(latest_numbers, float(size["diameter_mm"]) if size["diameter_mm"] else None,
                              float(size["length_mm"]) if size["length_mm"] else None)
    if latest_date is not None:
        latest_date = upright_date(latest_date, panel_reader())
    if last_crops is not None:
        # Owner review 23-10: a Sled Bag has no size, so its slot shows the printed date that was read (upright, as the OCR
        # read it); a sized box shows the expiry date next to the size when a complete date was read. Display only.
        if latest_date is not None and not any(size.values()):
            last_crops = dict(last_crops, size_labels=[latest_date], size_title="date (OCR)")
        elif latest_date is not None:
            last_crops = dict(last_crops, dates=[latest_date])
        spine = next((e.capture["spine"] for e in reversed(state.evs)
                      if getattr(e, "capture", None) and e.capture["spine"] is not None), None)
        if spine is not None and last_crops.get("barcode") is None:
            h, w = spine.shape[:2]
            upright = rectify(spine, np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]]))[0]
            located, _ = barcode_with_text(upright)
            if located is not None:
                cv2.imwrite(str(folder / "barcode_located.png"), located)
                last_crops = dict(last_crops, barcode=located, barcode_disabled=False, barcode_decoded=False)
    if shown and last_crops is not None:
        tiles = []
        for number in shown:
            img = number["image"]
            scale = min(100 / img.shape[0], 150 / img.shape[1])
            tile = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            tiles.append(cv2.copyMakeBorder(tile, 0, 100-tile.shape[0], 0, 10,
                                            cv2.BORDER_CONSTANT, value=(245, 245, 242)))
        strip = np.hstack(tiles)
        if strip.shape[1] < 100:
            strip = cv2.copyMakeBorder(strip, 0, 0, 0, 100-strip.shape[1], cv2.BORDER_CONSTANT,
                                       value=(245, 245, 242))
        last_crops = dict(last_crops, size_labels=[strip])
    pnl = panel(last_crops or {}, summary, height=340)
    cv2.rectangle(pnl, (0, 0), (pnl.shape[1]-1, pnl.shape[0]-1),
                  (0, 210, 0) if verified else (255, 220, 30), 3)
    cv2.imwrite(str(folder / "panel.png"), pnl)
    return pnl, folder

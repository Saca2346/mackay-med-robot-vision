"""Focused target tracking and identity verification, independent of PICK / TAKEN.

Visual-only mode can hold one repeated source; verified mode requires the existing
two-source/date gate and exports observed crops. Neither authorizes a pick or instantiates SelectionSession or
write grasp/inventory outputs. Display refinement never changes OCR geometry.
"""
from __future__ import annotations

import collections
import json
import queue
import sys
import threading
import time

import cv2
import numpy as np

from src.box_verifier import EXPIRED, IGNORED, combined_score
from src.box_preview import fit_preview


def verified_ready(state, need=2, required_expiry=None):
    """Use the physical verifier's evidence gate, without authorizing retrieval."""
    from src.box_verifier import MATCH
    return (not state.stale and state.stable(need) == MATCH
            and state.combined_match(need)
            and (required_expiry is None or state.expiry == required_expiry))


def keeps_verified(state, required_expiry=None):
    """A VERIFIED identity is released only by evidence against it, never because a later partial reading did not repeat
    a source. The two independent sources were already established on this track; released by: a doubtful track
    (stale), a reading in the combined window that contradicts or is not the target / expired, or a required printed
    date that no longer holds. (20-10: Accuforce 3.5 x 15 went green -> checking 3 times while only REF stopped
    repeating during the detail reads.)"""
    from src.box_verifier import COMBINE_WINDOW
    if state.stale or len(state.support) != len(state.history) or getattr(state, "look_veto", ""):
        return False
    if required_expiry is not None and state.expiry != required_expiry:
        return False
    rows = list(zip(state.history, state.support))[-COMBINE_WINDOW:]
    return not any(decision in (IGNORED, EXPIRED) or conflicts for (decision, _), (_, conflicts) in rows)


def source_ready(state, need=2, required_expiry=None):
    """Repeated complete support, without contradiction or a doubtful track."""
    if state.stale or len(state.history) < need or len(state.support) < need:
        return False
    if required_expiry is not None and state.expiry != required_expiry:
        return False
    history, support = list(state.history)[-need:], list(state.support)[-need:]
    gtins = set().union(*(combined_score(atoms)[1] for atoms, _ in support))
    if len(gtins) > 1:
        return False
    return all(decision not in (IGNORED, EXPIRED) and not conflicts and combined_score(atoms)[0] >= 1
               for (decision, _), (atoms, conflicts) in zip(history, support))


def supports_request(state, need=2, required_expiry=None):
    """Display only: the latest reading supports the request and nothing on the track contradicts it.

    A box that only came from the colour lock (no supporting evidence yet, a contradicting reading, or a printed date
    that differs twice) is drawn as a neutral tentative outline, not as a blue candidate. 23-10 replays: D455
    "accuforce 3.5 x 15" was blue on the Angiolite, and 8 of 22 requests for boxes that are not on the shelf ended blue
    on a same-product box of another size or date. Reading, scheduling and every verification gate are unchanged."""
    from src.box_verifier import combined_score
    if state.stale or not state.support or not state.history:
        return False
    atoms, conflicts = state.support[-1]
    if conflicts or combined_score(atoms)[0] < 1 or state.history[-1][0] in (IGNORED, EXPIRED):
        return False
    if required_expiry is not None:
        recent = [set(d) for d in list(state.dates)[-need:]]
        if len(recent) >= need and all(recent):
            common = set.intersection(*recent)
            if common and required_expiry not in common:
                return False
    return True


def select_visible(groups, ids, tracker, lock, held=None, required_expiry=None):
    """Choose only a currently detected track; never draw a predicted off-screen box."""
    visible = {tid for tid in ids if tid >= 0 and tid in tracker.tracks}
    if held in visible:
        return held
    candidates = []
    for tid in visible:
        state = tracker.state(tid)
        if state.stale or not state.support:
            continue
        atoms, conflicts = state.support[-1]
        score, _ = combined_score(atoms)
        if score >= 1 and not conflicts and state.history[-1][0] not in (IGNORED, EXPIRED):
            candidates.append((source_ready(state, required_expiry=required_expiry), score, -tid, tid))
    if candidates:
        return max(candidates)[-1]
    return lock.tid if lock is not None and lock.active and lock.tid in visible else None


def draw_target(frame, polygon, request, status, fps=None, detector_ms=0.0, verified=False, verifying=False,
                context=(), transparent=True, tentative=False):
    """Blend the header over the original viewport; source/OCR pixels stay untouched."""
    vis = frame.copy()
    if transparent:
        region = vis[:min(90, vis.shape[0])]
        cv2.addWeighted(region, .82, np.full_like(region, 20), .18, 0, dst=region)
    for tid, outline in context:
        points = np.rint(outline).astype(np.int32)
        cv2.polylines(vis, [points], True, (170, 170, 170), 1, cv2.LINE_AA)
        top = points[np.argmin(points[:, 1])]
        cv2.putText(vis, f"#{tid}", (max(0, int(top[0])), max(15, int(top[1])-4)),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (220, 220, 220), 1, cv2.LINE_AA)
    if polygon is not None:
        cv2.polylines(vis, [np.rint(polygon).astype(np.int32)], True,
                      (0, 210, 0) if verified else (190, 190, 190) if tentative else (255, 220, 30), 2, cv2.LINE_AA)
    if not transparent:
        vis = cv2.copyMakeBorder(vis, 90, 0, 0, 0, cv2.BORDER_CONSTANT, value=(28, 28, 28))
    rows = (f"TARGET: {request} | no retrieval command", status,
            (f"Display {fps:.1f} FPS | " if fps is not None else "")
            + f"Detector {detector_ms:.0f} ms | Q finish | O fresh read | A shelf area | target: CMD")
    text_mask = np.zeros((min(90, vis.shape[0]), vis.shape[1]), np.uint8)
    for y, text in zip((25, 52, 78), rows):
        available = max(1, vis.shape[1] - 24)
        while len(text) > 4 and cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, .40, 1)[0][0] > available:
            text = text[:-4].rstrip(".") + "..."
        width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, .60, 1)[0][0]
        scale = min(.60, .60 * available / max(1, width))
        cv2.putText(text_mask, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, 255, 1, cv2.LINE_AA)
    header = vis[:text_mask.shape[0]]
    if transparent:
        # Current OpenCV changes glyph advances with thickness. Dilate the same
        # glyph mask to keep its halo aligned instead of drawing a second font.
        halo = cv2.dilate(text_mask, np.ones((3, 3), np.uint8))[:, :, None] / 255.0
        header[:] = np.rint(header * (1 - halo) + 15 * halo).astype(np.uint8)
    ink = text_mask[:, :, None] / 255.0
    header[:] = np.rint(header * (1 - ink) + 235 * ink).astype(np.uint8)
    return vis


def validate_args(args):
    if not args.request:
        raise ValueError("Focused mode needs --request (product and size, or a dated product request)")
    if args.rencana or args.until_taken or args.qty != 1 or args.depth or args.no_lock:
        raise ValueError("Focused mode supports one target; omit --rencana, --until-taken, --qty, --depth and --no-lock")


def shelf_requests(entries, box_ids):
    """Number only configured reference requests; never label detected tracks from this list."""
    if len(box_ids) != len(set(box_ids)):
        raise ValueError("Duplicate shelf request IDs")
    rows = [r for r in entries if not r.get("generic") and len(r["boxes"]) == 1
            and r["boxes"][0]["id"] in box_ids]
    if {r["boxes"][0]["id"] for r in rows} != set(box_ids):
        raise ValueError("Shelf request IDs missing from inventory copy")
    return [dict(row, no=str(i+1)) for i, row in enumerate(rows)]


def run(args):
    """Reuse detector/tracker/read scheduling, without running the physical selection state machine."""
    from scripts import test_webcam_box as pipeline
    from src.box_catalog import Catalog
    from src.box_detector import BoxDetector
    from src.box_inventory import Inventory
    from src.box_lock import TargetLock, as_place_map, as_slot_map
    from src.box_motion import MotionEstimator
    from src.box_ocr import SizeTextReader
    from src.box_order import parse_order
    from src.box_outline import refine_box_outline, inset_display_outline
    from src.box_outline_track import OutlineTracker
    from src.box_stream import VideoRecorder
    from src.box_verifier import SimpleTracker
    from src.box_visual_read import colour_similarity, focused_order, read_visual_evidence, reuse_request
    from src.config import load_config, resolve
    from src.box_target_evidence import read_target_evidence, EvidenceWriter, observed_fields
    from src.box_photo_evidence import PhotoEvidence
    from src.box_view_recorder import ViewRecorder

    verifying = getattr(args, "verify_target", False)
    ready = verified_ready if verifying else source_ready

    try:
        validate_args(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if pipeline.is_photo_source(args.source):
        raise SystemExit("--lock-only uses a live camera or video; use the normal photo workflow for still images")
    if args.record and args.source and not pipeline.re.match(r"^[a-z]+://", args.source, pipeline.re.I):
        raise SystemExit("--record is only for a live camera / stream")
    cfg = load_config(resolve(args.config))
    if args.imgsz:
        size = [int(v) for v in args.imgsz.split(",")]
        cfg["model"]["imgsz"] = size[0] if len(size) == 1 else size
    if args.device:
        cfg["model"]["device"] = args.device
    model = cfg["model"]
    weights = resolve(args.weights or model["weights"])
    if not weights.exists():
        raise SystemExit(f"Model not found: {weights}")
    screens = resolve(cfg["output"]["screens_dir"])
    screens.mkdir(parents=True, exist_ok=True)
    output = resolve(cfg["output"].get("verify_log_csv", "results/verify_log.csv")).parent
    output.mkdir(parents=True, exist_ok=True)
    inventory = Inventory(resolve(args.inventaris)) if args.inventaris else None
    if inventory is not None and not inventory.boxes:
        raise SystemExit(f"Inventory missing/empty: {inventory.path}")

    def request_from(text):
        items = parse_order(text, cfg.get("catalog"))
        if len(items) != 1 or items[0].qty != 1:
            raise ValueError("Visual mode takes one request at a time")
        return items[0].req

    try:
        request = request_from(args.request)
    except ValueError as exc:
        raise SystemExit(f"Request not understood: {exc}") from exc
    print(f"[target] {'two-source verification' if verifying else 'visual lock only'}; loading {weights.name}; no PICK/TAKEN or grasp output", flush=True)
    detector = BoxDetector(str(weights), model["task"], model["imgsz"], model["device"], model["conf"],
                           model["iou"], cfg["classes"], fixed_input=bool(model.get("fixed_input", False)))
    from src.box_product_id import load as load_look
    pipeline.LOOK = load_look(cfg, resolve("."))
    if pipeline.LOOK is not None:
        print(f"[look] product look model on (order + cross-check only): {len(pipeline.LOOK.classes)} products", flush=True)
    fast = cfg.get("visual", {}).get("fast_read", True)
    workers = min(2, max(1, int(cfg.get("visual", {}).get("workers", 1)))) if fast else 1
    def new_reader():
        return SizeTextReader(cfg["ocr"]["engine"], cfg["ocr"]["retry_flip_below"],
                              threads=1 if workers == 2 else int(cfg["ocr"].get("threads", 2)),
                              backend=cfg["ocr"].get("backend", "onnxruntime"))
    reader = new_reader()
    catalog = Catalog(resolve(cfg.get("verify", {}).get("catalog_csv", "data/gtin_catalog.csv")))
    readers = [reader] + [new_reader() for _ in range(workers - 1)]
    if cfg.get("visual", {}).get("warmup_text", False):
        # Warm the recognition graph as part of initialization, before any timed frame.
        # Synthetic text is discarded and never enters evidence/tracker history.
        warmup = np.full((64, 360, 3), 255, np.uint8)
        cv2.putText(warmup, "Warmup 12345", (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        for engine in readers:
            if callable(getattr(engine, "read_once", None)):
                engine.read_once(warmup)
    verification_active = threading.Event()
    def target_read(*args, attempt=0):
        return read_target_evidence(*args, attempt=attempt,
                                    identity_verified=verification_active.is_set())
    target_read.uses_attempt = True
    worker = pipeline.OcrWorker(readers, cfg, request, catalog,
                                read_fn=target_read if verifying else read_visual_evidence if fast else None)
    tracker = SimpleTracker(history=cfg["matching"]["history"])
    motion = MotionEstimator()
    outline_tracker = OutlineTracker()
    stabilize_outline = cfg.get("visual", {}).get("stabilize_outline", True)
    show_all = cfg.get("visual", {}).get("show_all", False)
    need = max(2, int(cfg["matching"].get("require_consistent", 2)))
    pending = queue.Queue()
    if sys.stdin is not None and sys.stdin.isatty():
        def read_terminal():
            for line in sys.stdin:
                if line.strip():
                    pending.put(line.strip())
        threading.Thread(target=read_terminal, daemon=True).start()
    if inventory is not None:
        from scripts.demo_daftar_kotak import groups as inventory_groups, show as show_inventory
        numbered = inventory_groups(inventory.path)
        shelf_ids = cfg.get("visual", {}).get("request_ids")
        if shelf_ids:
            numbered = shelf_requests(numbered, shelf_ids)
            def show_inventory(entries):
                print("\n[shelf] Reference requests; visible identities must still be read", flush=True)
                for entry in entries:
                    print(f"  {entry['no']:>2}  {entry['boxes'][0]['id']}  {entry['text']}", flush=True)
                print("[shelf] Type a number or request in CMD; ? lists these requests again", flush=True)
            show_inventory(numbered)
    else:
        numbered = []

    def target_context(req):
        pipeline.box_verifier.REQUIRED_EXPIRY = getattr(req, "expiry", None)
        candidates = inventory.candidates(req) if inventory is not None else []
        pipeline.EXPIRY_ALLOWED = {b.expiry_date() for b in candidates if b.expiry_date()} or None
        pipeline.EXPIRY_NEED = need
        if pipeline.LOOK is not None:
            pipeline.LOOK.set_request(req.product)
            print(f"[look] request product {'known' if pipeline.LOOK.active else 'NOT known: model off'}", flush=True)
        hint_boxes = ([b for b in inventory.boxes.values() if b.product == req.product and b.status == "terverifikasi"]
                      if fast and inventory is not None else candidates)
        targets = [(b.box_id, b.signature()) for b in hint_boxes if b.signature() is not None]
        boxes = inventory.boxes.values() if inventory is not None else []
        places = as_place_map(boxes)
        print(f"[visual] request {req}; {len(targets)} inventory appearances; {len(places)} mapped places", flush=True)
        if targets and not places:
            print("[visual] no shelf position map: appearance is provisional until target text is read", flush=True)
        lock = TargetLock(n_shelf=args.expected or 0, min_similarity=.68 if fast else .80) if targets else None
        return targets, as_slot_map(boxes), places, lock

    targets, slots, places, lock = target_context(request)
    recorder = VideoRecorder(resolve(args.record), cfg["camera"].get("fps", 30)) if args.record else None
    info = {"recorder": recorder}
    source = pipeline.iter_source(args, cfg, info)
    stats = {"ocr_calls": 0, "ocr_ms": 0.0, "reads": 0, "blur": 0.0}
    roi = pipeline.load_roi(args.roi, False) if args.roi else None
    shown = collections.deque(maxlen=100)
    held, held_epoch, selected_before = None, None, None
    started, previous, frame_idx, writer = None, None, 0, None
    started_wall = None
    recording_started_wall = None
    request_generation = 0
    text_reports = set()
    report_key, report_panel, verified_seconds = None, None, None
    run_id = time.time_ns()
    details_complete = False
    evidence_writer = EvidenceWriter()
    latency_target = float(cfg.get("visual", {}).get("target_seconds", 2))
    defer_pending = float(cfg.get("visual", {}).get("defer_pending_seconds", 0))
    evidence_mode = "fresh request"
    photo_dir = getattr(args, "evidence_photos", None)
    photos = PhotoEvidence(resolve(photo_dir)) if photo_dir else None
    last_photo = None
    time_basis = "video" if args.source and not pipeline.re.match(r"^[a-z]+://", args.source, pipeline.re.I) else "live"
    debug = None
    if args.debug_readings:
        path = resolve(args.debug_readings)
        path.parent.mkdir(parents=True, exist_ok=True)
        debug = path.open("w", encoding="utf-8")
        debug.write("t,track,job_epoch,epoch,decision,identity,teks,ocr_ms,ocr_calls\n")
        pipeline.READ_LOG = debug
    log = (output / "target_view.jsonl").open("a", encoding="utf-8")
    outline_log = (output / "outline_frames.jsonl").open("a", encoding="utf-8")
    latency_log = (output / "frame_latency.jsonl").open("a", encoding="utf-8")
    detections_log = (output / "shelf_detections.jsonl").open("a", encoding="utf-8") if show_all else None

    def collect_export(current_key=None, wait=False):
        nonlocal report_panel
        completed = evidence_writer.drain(wait=wait)
        if completed is None:
            return False
        context, (panel, path) = completed
        is_current = context["key"] == current_key
        if is_current:
            report_panel = panel
        log.write(json.dumps({"state": "EVIDENCE_SAVED", "request": context["request"],
                              "track": context["track"], "identity_verified": context["verified"],
                              "current_context": is_current, "path": str(path),
                              "wall_seconds": round(time.perf_counter() - context["origin"], 3),
                              "time_basis": time_basis}) + "\n")
        log.flush()
        print(f"[evidence] {path}", flush=True)
        return is_current
    print("[visual] type a new product/size or inventory number; ? lists requests. q closes the view.", flush=True)
    try:
        for _, frame, _ in source:
            photo_event = None
            if frame is None:
                if not args.no_window and cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                continue
            frame_tick = time.perf_counter()
            now = info.get("t", time.monotonic())
            started = now if started is None else started
            started_wall = time.perf_counter() if started_wall is None else started_wall
            recording_started_wall = started_wall if recording_started_wall is None else recording_started_wall
            while not pending.empty():
                text = pending.get_nowait()
                if text in ("?", "list") and numbered:
                    show_inventory(numbered)
                    continue
                if text.isdigit() and numbered:
                    hit = next((row for row in numbered if row["no"] == str(int(text))), None)
                    if hit is None:
                        print("[visual] unknown list number; ? lists requests")
                        continue
                    text = hit["text"]
                try:
                    next_request = request_from(text)
                except ValueError as exc:
                    print(f"[visual] {exc}; current request unchanged")
                    continue
                request = worker.req = next_request
                request_generation += 1
                verification_active.clear()
                reuse_request(tracker, request, cfg, catalog)
                evidence_mode = "continuous-track cached observations"
                print("[target] cached observations rechecked for the new request; warm request timing", flush=True)
                held = held_epoch = selected_before = None
                report_key = report_panel = verified_seconds = None
                details_complete = False
                started = now
                started_wall = time.perf_counter()
                targets, slots, places, lock = target_context(request)
                outline_tracker.reset()
            if held is not None and (held not in tracker.tracks or tracker.state(held).stale
                                     or tracker.state(held).epoch != held_epoch):
                held = held_epoch = None
            process_tick = time.perf_counter()
            _, groups, _, ids, detect_ms, _ = pipeline.process(
                frame, detector, reader, tracker, request, cfg, frame_idx, worker=worker, catalog=catalog,
                motion_est=motion, need=need, dt=None if previous is None else now - previous,
                now=now, targets=targets, roi=roi, lock=lock, slots=slots, places=places, stats=stats,
                read_enabled=held is None or (verifying and not details_complete),
                hint_compare=colour_similarity if fast else None,
                read_order=(lambda gs, ts, tr, lk, nd: ([held] if frame_idx % 10 == 0 else [])
                            if verifying and held is not None else
                            focused_order(gs, ts, tr, lk, nd, required_expiry=request.expiry)) if fast else None)
            process_end = time.perf_counter()
            previous = now
            if held is not None and (held not in ids or tracker.state(held).stale
                                     or tracker.state(held).epoch != held_epoch
                                     or not (ready(tracker.state(held), need, request.expiry)
                                             or (verifying and keeps_verified(tracker.state(held), request.expiry)))):
                held = held_epoch = None
            selected = select_visible(groups, ids, tracker, lock, held, request.expiry)
            if selected is not None and ready(tracker.state(selected), need, request.expiry):
                held, held_epoch = selected, tracker.state(selected).epoch
            if held is None:
                verification_active.clear()
            else:
                verification_active.set()
            polygon = None
            raw = None
            text_read = False
            tentative = False
            status = "SEARCHING - reading target identity"
            if selected is not None:
                index = ids.index(selected)
                raw = groups[index].box.poly
                neighbours = [g.box.poly for j, g in enumerate(groups) if j != index]
                refine = lambda: refine_box_outline(frame, raw, neighbours)
                polygon = raw
                # Pixel-supported display geometry is independent of identity.
                # A pending target keeps reading and remains blue while its
                # visible outline can already be refined and stabilized.
                if stabilize_outline:
                    polygon = outline_tracker.update(frame, polygon,
                                (str(request), selected, tracker.state(selected).epoch, held is not None),
                                neighbours, observed_poly=raw, refine=refine)
                else:
                    outline_tracker.reset()
                    polygon = refine()
                polygon = inset_display_outline(polygon)
                status = (f"{'VERIFIED' if verifying else 'LOCKED'} #{selected} - "
                          + ("reading missing details" if verifying and not details_complete else "tracking; OCR paused") if held is not None
                          else f"CHECKING #{selected} - reading target text/date")
                if (verifying and held is None and not cfg.get("verify", {}).get("decode_barcodes", True)
                        and source_ready(tracker.state(selected), need, request.expiry)):
                    text_read = True
                    status = f"TEXT READ #{selected} - waiting for second independent source"
                if (held is None and status.startswith("CHECKING") and cfg.get("visual", {}).get("tentative_outline", True)
                        and not supports_request(tracker.state(selected), need, request.expiry)):
                    tentative = True      # colour-lock fallback or contradicted track: neutral outline, not a blue candidate
                    status = f"SEARCHING #{selected} - no evidence for the request yet (tentative box)"
            else:
                outline_tracker.reset()
            outline_end = time.perf_counter()
            outline_log.write(json.dumps({"frame": frame_idx, "source_seconds": round(now-started, 6),
                             "track": selected, "verified": verifying and held is not None,
                             "evidence_supported": None if selected is None else not tentative,
                             "mode": outline_tracker.mode if stabilize_outline else "disabled",
                             "raw": None if raw is None else np.asarray(raw).tolist(),
                             "display": None if polygon is None else np.asarray(polygon).tolist()}) + "\n")
            state_key = (selected, held is not None)
            first_green = False
            if state_key != selected_before:
                report_panel = report_key = None
                details_complete = False
                elapsed = round(now - started, 3)
                state_name = ("VERIFIED" if verifying else "VISUAL_LOCK") if held is not None else "CHECKING" if selected is not None else "SEARCHING"
                photo_event = state_name
                verified_seconds = round(time.perf_counter() - started_wall, 3) if verifying and held is not None else None
                first_green = verified_seconds is not None
                row = {"request": str(request), "state": state_name,
                       "track": selected, "seconds": elapsed,
                       "ocr_paused": held is not None and not verifying, "verified_pick": False,
                       "time_basis": time_basis, "wall_seconds": round(time.perf_counter() - started_wall, 3),
                       "identity_verified": verifying and held is not None,
                       "latency_target_seconds": latency_target,
                       "within_target": verified_seconds <= latency_target if verified_seconds is not None else None,
                       "evidence_mode": evidence_mode,
                       "sources": tracker.state(selected).combined_sources() if selected is not None else "",
                       "look": (tracker.state(selected).look_top, round(tracker.state(selected).look_p, 3),
                                tracker.state(selected).look_veto) if selected is not None else None,
                       "outline_px": None if polygon is None else np.asarray(polygon).round(1).tolist(),
                       "evidence_supported": None if selected is None else not tentative,
                       "outline_use": "display only"}
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(f"[visual] {time_basis} {elapsed:.2f}s; wall {row['wall_seconds']:.2f}s {status}; no pick authorization", flush=True)
                selected_before = state_key
            if verifying and selected is not None:
                st = tracker.state(selected)
                # Save observed candidate evidence too; a missing second source must remain reviewable.
                fields = observed_fields(st)
                details_complete = held is not None and all(f["status"] == "read" for f in fields.values())
                key = (request_generation, str(request), selected, st.epoch, held is not None,
                       tuple((k, f["value"], f["status"]) for k, f in fields.items()), st.combined_sources())
                if collect_export(key):
                    photo_event = "VERIFIED" if held is not None else "PENDING"
                if (st.evs and any(getattr(e, "capture", None) for e in st.evs) and key != report_key
                        and (held is not None or time.perf_counter() - started_wall >= defer_pending)):
                    context = {"key": key, "request": str(request), "track": selected,
                               "verified": held is not None, "origin": started_wall}
                    if evidence_writer.submit(context, output / "verification", run_id, request, selected,
                                               st, held is not None, verified_seconds, time_basis):
                        report_key = key
            else:
                collect_export()
                report_panel = None
            evidence_end = time.perf_counter()
            if lock is not None:
                lock.log.clear()
            rates = pipeline.rates(shown, info)
            if verifying:
                status += (f" | verified in {verified_seconds:.2f}s" if verified_seconds is not None
                           else f" | {time.perf_counter() - started_wall:.1f}s; waiting for independent evidence")
                if verified_seconds is not None:
                    status += f" | {'within' if verified_seconds <= latency_target else 'over'} {latency_target:g}s target"
            context = []
            if show_all:
                context = [(tid, inset_display_outline(g.box.poly)) for g, tid in zip(groups, ids)
                           if tid != selected or polygon is None]
                status += f" | boxes {len(groups)}/{args.expected} expected"
                detections_log.write(json.dumps({"frame": frame_idx, "source_seconds": round(now-started, 6),
                    "expected": args.expected, "detected": len(groups), "selected": selected,
                    "boxes": [{"track": tid, "raw": np.asarray(g.box.poly).tolist(),
                               "identity_verified": tid == held and held is not None and verifying}
                              for g, tid in zip(groups, ids)],
                    "scope": "Count/context detections are not identity verification or accepted box labels"})+"\n")
            vis = draw_target(frame, polygon, str(request), status, rates.get("real"), detect_ms,
                              verified=verifying and held is not None, verifying=verifying, context=context,
                              transparent=cfg.get("visual", {}).get("transparent_header", True), tentative=tentative)
            if verifying and report_panel is not None:
                # Append outside the camera image so neither end of the target is obscured.
                h = 300
                vis = np.vstack([vis, cv2.resize(report_panel, (vis.shape[1], h))])
            elif verifying:
                vis = np.vstack([vis, np.full((300, vis.shape[1], 3), 28, np.uint8)])
            render_end = time.perf_counter()
            if photos is not None:
                metadata = {"request": str(request), "track": selected, "frame_index": frame_idx,
                            "identity_verified": verifying and held is not None,
                            "status": status, "time_basis": time_basis, "source": args.source or "live camera",
                            "source_seconds": round(now - started, 3),
                            "wall_seconds": round(time.perf_counter() - started_wall, 3),
                            "verified_wall_seconds": verified_seconds,
                            "latency_target_seconds": latency_target, "evidence_mode": evidence_mode,
                            "all_fields_read": details_complete,
                            "raw_recording": args.record, "view_recording": args.record_vis}
                last_photo = (frame, vis, metadata)
                if photo_event is not None:
                    photos.submit(frame, vis, photo_event, metadata)
            if args.record_vis:
                if writer is None:
                    path = resolve(args.record_vis)
                    writer = ViewRecorder(path, args.record_vis_fps,
                                          max_width=cfg.get("visual", {}).get("record_width"))
                writer.write(vis, time.perf_counter() - recording_started_wall)
            presentation = "rendered frame ready"
            if not args.no_window:
                width = args.view_width or 1280
                show = fit_preview(vis, width)
                cv2.imshow("Target lock - external webcam (q quit)", show)
                presentation = "window submitted"
            present_end = time.perf_counter()
            latency = {"frame": frame_idx, "request": str(request), "track": selected,
                       "request_generation": request_generation,
                       "identity_verified": verifying and held is not None,
                       "source_seconds": round(now-started, 6),
                       "wall_seconds": round(present_end-started_wall, 6),
                       "detector_ms": detect_ms, "process_ms": (process_end-process_tick)*1000,
                       "outline_ms": (outline_end-process_end)*1000,
                       "evidence_submit_ms": (evidence_end-outline_end)*1000,
                       "render_ms": (render_end-evidence_end)*1000,
                       "photo_video_present_ms": (present_end-render_end)*1000,
                       "frame_ms": (present_end-frame_tick)*1000, "presentation": presentation,
                       "look_ms": pipeline.LOOK.last_ms if pipeline.LOOK is not None else None,
                       "look_error": (pipeline.LOOK.error or None) if pipeline.LOOK is not None else None}
            latency_log.write(json.dumps(latency) + "\n")
            text_key = (request_generation, selected, tracker.state(selected).epoch) if text_read else None
            if text_read and text_key not in text_reports:
                text_reports.add(text_key)
                log.write(json.dumps({"state": "TEXT_READY", **latency,
                                      "sources": tracker.state(selected).combined_sources(),
                                      "evidence_mode": evidence_mode}) + "\n")
                log.flush()
                print(f"[target] text frame ready in {latency['wall_seconds']:.3f}s; "
                      "identity still unverified; barcode/QR decoding OFF", flush=True)
            if first_green:
                log.write(json.dumps({"state": "DISPLAY_READY", **latency,
                                      "verified_wall_seconds": verified_seconds,
                                      "within_target": latency["wall_seconds"] <= latency_target,
                                      "sources": tracker.state(selected).combined_sources(),
                                      "evidence_mode": evidence_mode}) + "\n")
                log.flush()
                print(f"[target] green frame ready in {latency['wall_seconds']:.3f}s "
                      f"({'within' if latency['wall_seconds'] <= latency_target else 'over'} {latency_target:g}s); {presentation}", flush=True)
            shown.append(time.perf_counter())
            frame_idx += 1
            if args.max_frames and frame_idx >= args.max_frames:
                break
            if args.no_window:
                continue
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("o"):
                request_generation += 1
                verification_active.clear()
                outline_tracker.reset()
                tracker.reset()
                held = held_epoch = selected_before = None
                report_key = report_panel = verified_seconds = None
                details_complete = False
                evidence_mode = "fresh request"
                targets, slots, places, lock = target_context(request)
                started = now
                started_wall = time.perf_counter()
            if key == ord("a"):
                outline_tracker.reset()
                x, y, w, h = cv2.selectROI("Shelf area (Enter to accept)", frame, showCrosshair=False)
                cv2.destroyWindow("Shelf area (Enter to accept)")
                if w > 20 and h > 20:
                    request_generation += 1
                    verification_active.clear()
                    roi = (x / frame.shape[1], y / frame.shape[0], (x+w) / frame.shape[1], (y+h) / frame.shape[0])
                    tracker.reset()
                    held = held_epoch = selected_before = None
                    report_key = report_panel = verified_seconds = None
                    details_complete = False
                    evidence_mode = "fresh request"
                    started = now
                    started_wall = time.perf_counter()
                    targets, slots, places, lock = target_context(request)
            if key == ord("s"):
                if photos is not None:
                    photos.submit(frame, vis, "MANUAL", metadata)
                else:
                    cv2.imwrite(str(screens / f"target_{frame_idx:06d}.png"), vis)
    finally:
        source.close()
        if recorder is not None:
            recorder.close()
        if writer is not None:
            writer.close()
        try:
            collect_export(wait=True)
        finally:
            evidence_writer.close()
            log.close()
            outline_log.close()
            latency_log.close()
            if detections_log is not None:
                detections_log.close()
        if debug is not None:
            pipeline.READ_LOG = None
            debug.close()
        cv2.destroyAllWindows()
        if photos is not None:
            try:
                if last_photo is not None:
                    photos.submit(last_photo[0], last_photo[1], "FINAL", last_photo[2])
            finally:
                photos.close()
    print(f"[visual] finished: {frame_idx} frames, {stats['reads']} OCR reads; log {output / 'target_view.jsonl'}")
    if not frame_idx:
        raise SystemExit("No camera/video frames received")

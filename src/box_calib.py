"""
Camera-position calibration card for a recording session (pure measurement, no camera or model in here).

Input: the box polygons detected in a few still frames, the frame size, sharpness readings and the frame-to-frame
change. Output: a card with the numbers that decide whether the shelf data of this session is usable, a PASS / WARN /
FAIL per check with advice, and (given the card of an earlier session) how far the camera moved since then.

Limits come from our own measurements (docs/DATA_RAK_PROTOCOL.md): box width 24 px -> size text read in 7 % of readings;
36 px -> 34 % and PICK for iVascular; 40 px is the config value that switches the whole-spine read on (never reached in
a live test). Also: sharpness >= 300, camera still, target boxes away from the image edge.
"""
from __future__ import annotations

import math

import numpy as np

WIDTH_MIN, WIDTH_IDEAL = 28.0, 40.0       # px of the box's short side: < 28 FAIL (24 px = 7 % reads), 28-40 WARN
WIDTH_MAX = 140.0                          # wider = too close for a wide view (few boxes), only a warning
SHARP_MIN = 300.0
STILL_MAX = 4.0                            # camera shift in px (full frame) during the 3 s measurement (camera_motion)
MOVING_WARN = 5.0                          # % of the picture that changed (a hand, a person passing): warning only
TILT_WARN, TILT_FAIL = 3.0, 8.0            # degrees the boxes lean from vertical (= camera roll)
MARGIN_MIN = 0.03                          # free image width left of the first / right of the last box
WIDE_X = 2.5                               # detections wider than this x the median width are not spines
LYING_DEG = 30.0                           # axis further than this from vertical = a lying box, not used for tilt


def _poly(p) -> np.ndarray:
    return np.asarray(p, dtype=float).reshape(4, 2)


def _short_long(p: np.ndarray) -> tuple[float, float]:
    e = sorted((float(np.linalg.norm(p[1] - p[0])), float(np.linalg.norm(p[2] - p[1]))))
    return e[0], e[1]


def _axis_deg_from_vertical(p: np.ndarray) -> float:
    """Signed angle (deg) of the box's long axis from vertical; + = leans right at the top."""
    a, b = (p[1] - p[0], p[2] - p[1])
    v = a if np.linalg.norm(a) >= np.linalg.norm(b) else b
    ang = math.degrees(math.atan2(v[0], -v[1]))
    while ang > 90:
        ang -= 180
    while ang < -90:
        ang += 180
    return ang


def spines(polys: list, ) -> list[np.ndarray]:
    """Drop detections that are far wider than the typical spine (monitor, shelf post area)."""
    ps = [_poly(p) for p in polys]
    if len(ps) < 3:
        return ps
    med = float(np.median([_short_long(p)[0] for p in ps]))
    return [p for p in ps if _short_long(p)[0] <= WIDE_X * med]


def _norm(gray: np.ndarray) -> np.ndarray:
    """Grey frame at 320 px wide, brightness and contrast removed (auto-exposure and lamp flicker are not motion)."""
    g = np.asarray(gray, dtype=np.float32)
    if g.shape[1] != 320:
        import cv2
        g = cv2.resize(g, (320, max(1, round(g.shape[0] * 320 / g.shape[1]))), interpolation=cv2.INTER_AREA)
    return (g - g.mean()) / (g.std() + 1e-6)


def camera_motion(grays: list, full_width: int) -> tuple[float, float]:
    """Grey frames of the 3 s measurement -> (camera shift in px of the full frame, % of the picture that changed).

    Before 14 Oct the check was the mean grey difference of the first and last frame: the D455 auto-exposure and
    lamp flicker alone gave values far above the limit on a camera standing still on a tripod. Now: the image
    shift is measured with phase correlation on brightness-normalised frames (largest shift against the first frame),
    and a moving hand or person shows up separately as the share of changed pixels."""
    import cv2
    if len(grays) < 2:
        return 0.0, 0.0
    ref = _norm(grays[0])
    win = cv2.createHanningWindow((ref.shape[1], ref.shape[0]), cv2.CV_32F)
    k = full_width / ref.shape[1]
    shifts, moving = [], 0.0
    for g in grays[1:]:
        n = _norm(g)
        (dx, dy), _ = cv2.phaseCorrelate(ref, n, win)
        shifts.append(math.hypot(dx, dy) * k)
        moving = max(moving, float((np.abs(cv2.GaussianBlur(n - ref, (5, 5), 0)) > 1.0).mean() * 100))
    return round(max(shifts), 2), round(moving, 1)


def measure(polys: list, size: tuple[int, int], sharp: list[float], still_diff: float, moving_pct: float = 0.0) -> dict:
    """-> card (numbers only). polys = box polygons of ONE representative frame."""
    W, H = size
    ps = spines(polys)
    shorts = [_short_long(p)[0] for p in ps]
    angs = [_axis_deg_from_vertical(p) for p in ps]
    up = [a for a in angs if abs(a) <= LYING_DEG]
    xs = [(float(p[:, 0].min()), float(p[:, 0].max())) for p in ps]
    ys = [(float(p[:, 1].min()), float(p[:, 1].max())) for p in ps]
    cut = sum(1 for p in ps if (min(p[:, 0].min(), p[:, 1].min()) < -0.1 * min(_short_long(p))
                                or p[:, 0].max() > W * 1.0 + 0.1 * min(_short_long(p))
                                or p[:, 1].max() > H * 1.0 + 0.1 * _short_long(p)[1]))
    return {
        "frame": [int(W), int(H)],
        "boxes": len(ps),
        "width_px": round(float(np.median(shorts)), 1) if shorts else 0.0,
        "width_min_px": round(float(min(shorts)), 1) if shorts else 0.0,
        "tilt_deg": round(float(np.median(up)), 1) if up else 0.0,
        "lying": len(angs) - len(up),
        "margin_left": round(min(x[0] for x in xs) / W, 3) if xs else 0.0,
        "margin_right": round(1 - max(x[1] for x in xs) / W, 3) if xs else 0.0,
        "margin_top": round(min(y[0] for y in ys) / H, 3) if ys else 0.0,
        "margin_bottom": round(1 - max(y[1] for y in ys) / H, 3) if ys else 0.0,
        "edge_cut": int(cut),
        "sharpness": round(float(np.median(sharp)), 0) if sharp else 0.0,
        "still_diff": round(float(still_diff), 2),          # camera shift, px (camera_motion)
        "moving_pct": round(float(moving_pct), 1),
        "edge_left_px": round(min(x[0] for x in xs), 1) if xs else 0.0,
        "edge_right_px": round(max(x[1] for x in xs), 1) if xs else 0.0,
        "edge_top_px": round(min(y[0] for y in ys), 1) if ys else 0.0,
        "edge_bottom_px": round(max(y[1] for y in ys), 1) if ys else 0.0,
    }


def parse_region(value: str):
    """Normalized x1,y1,x2,y2; keep one validated region for calibration and reading."""
    if value.lower() == "none":
        return None
    try:
        region = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise ValueError("region must be x1,y1,x2,y2 between 0 and 1") from error
    if (len(region) != 4 or not all(math.isfinite(v) for v in region)
            or not (0 <= region[0] < region[2] <= 1 and 0 <= region[1] < region[3] <= 1)):
        raise ValueError("region must satisfy 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1")
    return region


def region_polygons(polys, size, region):
    """Select by centre in the original frame, matching the target reader's ROI policy."""
    if region is None:
        return polys
    width, height = size
    return [p for p in polys if region[0] <= _poly(p)[:, 0].mean() / width <= region[2]
            and region[1] <= _poly(p)[:, 1].mean() / height <= region[3]]


def judge(card: dict, min_boxes: int = 3) -> list[dict]:
    """PASS / WARN / FAIL per check, each with the value, the limit and what to do."""
    if min_boxes < 1:
        raise ValueError("minimum box count must be positive")
    out = []

    def add(name, value, limit, status, advice=""):
        out.append({"check": name, "value": value, "limit": limit, "status": status, "advice": advice})

    n, w = card["boxes"], card["width_px"]
    add("boxes found", n, f">= {min_boxes}", "PASS" if n >= min_boxes else "FAIL",
        "" if n >= min_boxes else "point the camera at the target area; check lighting and the selected region")
    if n:
        st = "PASS" if w >= WIDTH_IDEAL else "WARN" if w >= WIDTH_MIN else "FAIL"
        advice = ("" if st == "PASS" else
                  "move the camera closer: at this width the size text is read in only ~7 % of readings"
                  if st == "FAIL" else
                  "enough for iVascular (REF inside the size label); other brands need a close-up for their 2nd source")
        if w > WIDTH_MAX:
            st, advice = "WARN", "very close: fewer boxes per frame; fine for a close-up session"
        add("box width", f"{w:g} px", f">= {WIDTH_MIN:g} to read, {WIDTH_IDEAL:g}+ for the spine read", st, advice)
    s = card["sharpness"]
    add("sharpness", f"{s:g}", f">= {SHARP_MIN:g}", "PASS" if s >= SHARP_MIN else "FAIL",
        "" if s >= SHARP_MIN else "refocus / change the distance; check for glare and a dirty lens")
    d = card["still_diff"]
    add("camera still", f"{d:g} px", f"shift < {STILL_MAX:g} px", "PASS" if d < STILL_MAX else "FAIL",
        "" if d < STILL_MAX else "the camera itself moved: tripod, do not touch the laptop / cable during the countdown")
    mv = card.get("moving_pct", 0.0)
    if mv >= MOVING_WARN:
        add("nothing moving in view", f"{mv:g} %", f"< {MOVING_WARN:g} % of the picture changed", "WARN",
            "a hand, a person or loose paper / plastic moved in the picture; measure again when the view is quiet")
    t = abs(card["tilt_deg"])
    add("camera roll", f"{card['tilt_deg']:+g} deg", f"|x| < {TILT_WARN:g}",
        "PASS" if t < TILT_WARN else "WARN" if t < TILT_FAIL else "FAIL",
        "" if t < TILT_WARN else "level the camera: the boxes should stand vertical in the image")
    ml, mr = card["margin_left"], card["margin_right"]
    ok = min(ml, mr) >= MARGIN_MIN and card["edge_cut"] == 0
    add("shelf inside the frame", f"left {ml:.0%}, right {mr:.0%}, cut {card['edge_cut']}",
        f">= {MARGIN_MIN:.0%} free each side, none cut", "PASS" if ok else "WARN",
        "" if ok else "centre the shelf: boxes at the image edge are read badly (measured: ~200 s vs 17 s)")
    mt, mb = card.get("margin_top"), card.get("margin_bottom")
    ends_ok = mt is not None and mb is not None and min(mt, mb) >= MARGIN_MIN
    value = "not measured" if mt is None or mb is None else f"top {mt:.1%}, bottom {mb:.1%}"
    add("top and bottom inside frame", value, f">= {MARGIN_MIN:.0%} free at both ends",
        "PASS" if ends_ok else "WARN", "" if ends_ok else
        "leave room above and below the actual cartons; a clipped end cannot be reconstructed")
    return out


def compare(card: dict, ref: dict) -> dict:
    """How the camera moved since the reference card: shift (px), scale change (distance) and roll change.

    The shift is taken from the OUTER edges of the shelf (left edge of the left-most box, right edge of the right-most
    box). Matching box to nearest box fails on a shelf of near-identical, evenly spaced spines: a camera moved by 60 px
    still "matches" the neighbour 10 px away. The edges are only comparable when both views hold the same boxes
    (+-1); otherwise the shift is None (a different part of the shelf was filmed)."""
    same_frame = card["frame"] == ref["frame"]
    same_view = same_frame and abs(card["boxes"] - ref["boxes"]) <= 1 and card["boxes"] > 0 and ref["boxes"] > 0
    if same_view:
        shift = float(np.median([card["edge_left_px"] - ref["edge_left_px"], card["edge_right_px"] - ref["edge_right_px"]]))
    else:
        shift = None
    scale = (card["width_px"] / ref["width_px"] - 1.0) if same_view and ref["width_px"] else None
    return {"shift_px": None if shift is None else round(abs(shift), 1),
            "scale_change": None if scale is None else round(scale, 3),
            "comparison_basis": "same-resolution detected shelf proxy; not physical distance or camera identity",
            "roll_change_deg": round(card["tilt_deg"] - ref["tilt_deg"], 1)}


def judge_move(cmp: dict, same_position: bool) -> dict:
    """With the intent 'same position as the reference' the camera should not have moved."""
    sh = cmp["shift_px"]
    scale = cmp["scale_change"]
    shown = ("view or resolution differs" if sh is None else f"shift {sh:g} px") + (
        ", scale unavailable" if scale is None else f", scale {scale:+.0%}") + f", roll {cmp['roll_change_deg']:+g} deg"
    if not same_position:
        return {"check": "position vs reference", "value": shown, "limit": "(different position wanted)",
                "status": "INFO", "advice": ""}
    if sh is None or scale is None:
        return {"check": "same position as the reference", "value": shown, "limit": "same boxes in view",
                "status": "WARN", "advice": "use the same resolution and physical boxes; pixel scale is not a distance measurement"}
    bad = sh > 25 or abs(cmp["scale_change"]) > 0.08 or abs(cmp["roll_change_deg"]) > 2
    return {"check": "same position as the reference", "value": shown,
            "limit": "shift <= 25 px, scale +-8 %, roll +-2 deg", "status": "WARN" if bad else "PASS",
            "advice": "put the camera back where the reference session was taken" if bad else ""}


def verdict(checks: list[dict]) -> str:
    s = {c["status"] for c in checks}
    return "FAIL" if "FAIL" in s else "WARN" if "WARN" in s else "PASS"

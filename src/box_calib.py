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
STILL_MAX = 3.0                            # mean grey difference between the first and last frame
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


def measure(polys: list, size: tuple[int, int], sharp: list[float], still_diff: float) -> dict:
    """-> card (numbers only). polys = box polygons of ONE representative frame."""
    W, H = size
    ps = spines(polys)
    shorts = [_short_long(p)[0] for p in ps]
    angs = [_axis_deg_from_vertical(p) for p in ps]
    up = [a for a in angs if abs(a) <= LYING_DEG]
    xs = [(float(p[:, 0].min()), float(p[:, 0].max())) for p in ps]
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
        "edge_cut": int(cut),
        "sharpness": round(float(np.median(sharp)), 0) if sharp else 0.0,
        "still_diff": round(float(still_diff), 2),
        "edge_left_px": round(min(x[0] for x in xs), 1) if xs else 0.0,
        "edge_right_px": round(max(x[1] for x in xs), 1) if xs else 0.0,
    }


def judge(card: dict) -> list[dict]:
    """PASS / WARN / FAIL per check, each with the value, the limit and what to do."""
    out = []

    def add(name, value, limit, status, advice=""):
        out.append({"check": name, "value": value, "limit": limit, "status": status, "advice": advice})

    n, w = card["boxes"], card["width_px"]
    add("boxes found", n, ">= 3", "PASS" if n >= 3 else "FAIL",
        "" if n >= 3 else "point the camera at the shelf; check the lighting and the shelf area")
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
    add("camera still", f"{d:g}", f"< {STILL_MAX:g}", "PASS" if d < STILL_MAX else "FAIL",
        "" if d < STILL_MAX else "hold the camera on a tripod; something moved in the frame")
    t = abs(card["tilt_deg"])
    add("camera roll", f"{card['tilt_deg']:+g} deg", f"|x| < {TILT_WARN:g}",
        "PASS" if t < TILT_WARN else "WARN" if t < TILT_FAIL else "FAIL",
        "" if t < TILT_WARN else "level the camera: the boxes should stand vertical in the image")
    ml, mr = card["margin_left"], card["margin_right"]
    ok = min(ml, mr) >= MARGIN_MIN and card["edge_cut"] == 0
    add("shelf inside the frame", f"left {ml:.0%}, right {mr:.0%}, cut {card['edge_cut']}",
        f">= {MARGIN_MIN:.0%} free each side, none cut", "PASS" if ok else "WARN",
        "" if ok else "centre the shelf: boxes at the image edge are read badly (measured: ~200 s vs 17 s)")
    return out


def compare(card: dict, ref: dict) -> dict:
    """How the camera moved since the reference card: shift (px), scale change (distance) and roll change.

    The shift is taken from the OUTER edges of the shelf (left edge of the left-most box, right edge of the right-most
    box). Matching box to nearest box fails on a shelf of near-identical, evenly spaced spines: a camera moved by 60 px
    still "matches" the neighbour 10 px away. The edges are only comparable when both views hold the same boxes
    (+-1); otherwise the shift is None (a different part of the shelf was filmed)."""
    same_view = abs(card["boxes"] - ref["boxes"]) <= 1 and card["boxes"] > 0 and ref["boxes"] > 0
    if same_view:
        shift = float(np.median([card["edge_left_px"] - ref["edge_left_px"], card["edge_right_px"] - ref["edge_right_px"]]))
    else:
        shift = None
    scale = (card["width_px"] / ref["width_px"] - 1.0) if ref["width_px"] else 0.0
    return {"shift_px": None if shift is None else round(abs(shift), 1), "scale_change": round(scale, 3),
            "roll_change_deg": round(card["tilt_deg"] - ref["tilt_deg"], 1)}


def judge_move(cmp: dict, same_position: bool) -> dict:
    """With the intent 'same position as the reference' the camera should not have moved."""
    sh = cmp["shift_px"]
    shown = ("different part of the shelf" if sh is None else f"shift {sh:g} px") +         f", scale {cmp['scale_change']:+.0%}, roll {cmp['roll_change_deg']:+g} deg"
    if not same_position:
        return {"check": "position vs reference", "value": shown, "limit": "(different position wanted)",
                "status": "INFO", "advice": ""}
    if sh is None:
        return {"check": "same position as the reference", "value": shown, "limit": "same boxes in view",
                "status": "WARN", "advice": "the view holds other boxes than the reference: aim at the same part of the shelf"}
    bad = sh > 25 or abs(cmp["scale_change"]) > 0.08 or abs(cmp["roll_change_deg"]) > 2
    return {"check": "same position as the reference", "value": shown,
            "limit": "shift <= 25 px, scale +-8 %, roll +-2 deg", "status": "WARN" if bad else "PASS",
            "advice": "put the camera back where the reference session was taken" if bad else ""}


def verdict(checks: list[dict]) -> str:
    s = {c["status"] for c in checks}
    return "FAIL" if "FAIL" in s else "WARN" if "WARN" in s else "PASS"

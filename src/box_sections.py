"""
Shelf sections (meeting 29-10, Sub-project 1): each wide shelf level is split into 2-3 fixed sections in the inventory
(data/rak/posisi.csv column `seksi`), sized from the camera's field of view, so the robot drives straight to the
target's section and the box is chosen in ONE camera frame, without a visual search along the shelf.

  plan_sections   shelf width + camera distance + horizontal FOV -> number of sections, their edges, spine size in px
  distance_for    camera distance at which one section (plus a free margin each side) fills the frame
  measure_hfov    horizontal FOV from a photo of a known width at a known distance (the maker's figure is a guess)
  assign_sections columns of one level -> {box: (section, slot within the section)}
  slot_in_view    one frame of a section -> the detected box in the target's slot (counted from the section's left end)

The slot is counted from the section's end, not from where the box stood in the camera frame: on the 29-10 Task 3
recordings the camera stood 0.9 box pitch aside and 32 % closer in the second run, the frame-place map chose the right
box for 2 of 11 requests, counting from the row end for 11 of 11 (results/sections_20261030). The count is only used
when every box of the section is detected; the section's ends must therefore be visible (shelf end or divider).

Positions only choose WHERE to look and which box to read first. PICK still needs two independent sources on that box
(src/box_verifier); a box that is not where the inventory says is read like any other.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# Horizontal FOV (deg) from the makers' data sheets, replaced by measure_hfov once measured:
# Logitech C930e 90 deg diagonal -> 82.1 deg horizontal at 16:9; RealSense D455 colour 90 x 65 deg.
CAMERA_HFOV = {"c930e": 82.1, "d455": 90.0}
MARGIN = 0.10            # free frame width each side of a section (camera stop error, boxes leaning out)
MAX_SECTIONS = 3         # meeting 29-10: 2-3 sections per level


def visible_width(distance: float, hfov_deg: float) -> float:
    """Width the camera sees at `distance` (same unit) across the whole frame."""
    return 2.0 * distance * math.tan(math.radians(hfov_deg) / 2.0)


def distance_for(width: float, hfov_deg: float, margin: float = MARGIN) -> float:
    """Camera distance at which `width` fills the frame less `margin` of it on each side."""
    return width / (1.0 - 2.0 * margin) / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))


def measure_hfov(width: float, width_px: float, frame_px: int, distance: float) -> float:
    """Horizontal FOV (deg) from a photo: something `width` wide (e.g. the shelf, a ruler) spans `width_px` of a
    `frame_px` wide frame at `distance` from the lens (pinhole camera, object square to the camera)."""
    f_px = width_px * distance / width
    return math.degrees(2.0 * math.atan(frame_px / (2.0 * f_px)))


@dataclass
class SectionPlan:
    shelf: float                         # shelf level width (cm)
    distance: float                      # camera to box fronts (cm)
    hfov_deg: float
    frame_px: int
    sections: int                        # sections needed so each fits the frame with the margin
    visible: float                       # width seen across the frame (cm)
    edges: list = field(default_factory=list)       # [(left, right)] cm from the shelf's left end
    centres: list = field(default_factory=list)     # camera stop per section (cm from the left end)
    px_per_cm: float = 0.0
    spine_px: float | None = None        # a spine `spine` cm wide in px at this distance

    @property
    def fits(self) -> bool:
        return self.sections <= MAX_SECTIONS


def plan_sections(shelf: float, distance: float, hfov_deg: float, frame_px: int = 1920, spine: float | None = None,
                  margin: float = MARGIN, sections: int | None = None) -> SectionPlan:
    """Fixed equal sections of one shelf level. `sections` = None: as few as fit the frame at `distance`."""
    visible = visible_width(distance, hfov_deg)
    n = sections or max(1, math.ceil(shelf / (visible * (1.0 - 2.0 * margin)) - 1e-9))
    step = shelf / n
    ppc = frame_px / visible
    return SectionPlan(shelf, distance, hfov_deg, frame_px, n, visible,
                       [(round(k * step, 1), round((k + 1) * step, 1)) for k in range(n)],
                       [round((k + 0.5) * step, 1) for k in range(n)], ppc,
                       None if spine is None else spine * ppc)


def assign_sections(columns: dict, n_columns: int, sections: int) -> dict:
    """{box: column (1 = leftmost position of the level, empty positions counted)} -> {box: (section, slot)}.
    Sections split the columns equally (fixed, not moved when a box is taken); slot = rank from the section's left end
    among the boxes listed, i.e. the boxes on the shelf (a taken box's empty place is not counted)."""
    sec = {b: min(sections, (int(c) - 1) * sections // n_columns + 1) for b, c in columns.items()}
    out = {}
    for s in set(sec.values()):
        for k, b in enumerate(sorted((b for b in sec if sec[b] == s), key=lambda b: int(columns[b]))):
            out[b] = (s, k + 1)
    return out


def slot_in_view(xs, expected: int, slot: int):
    """Spine centres along the row (px, any order) of ONE section in view -> index of the box in `slot` (1 = leftmost)
    or None when the count differs from `expected` (a box missed, a neighbour section in view): then the program
    looks for the box by its appearance as before."""
    if slot < 1 or len(xs) != expected or slot > expected:
        return None
    return sorted(range(len(xs)), key=lambda i: xs[i])[slot - 1]

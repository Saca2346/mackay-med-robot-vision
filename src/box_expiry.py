"""
Expiry dates printed as text on a box spine (next to the hourglass "use by" symbol), e.g. "2029-03-10".

Why: identical boxes (the four Permanent Sled Bags K01/K03/K05/K17: same product, same GTIN, no size) differ ONLY in
the printed expiry date. The date never counts as evidence for MATCH (src/box_verifier.decide_multi is unchanged); it
only decides WHICH of several matching boxes is picked (FEFO) and which inventory row is marked taken.

The GS1 barcode date (17)YYMMDD stays the stronger source: a printed date that disagrees with the barcode is ignored.
"""
from __future__ import annotations

import datetime as dt
import re

# OCR confuses these with digits on the thin spine font
_DIGIT = str.maketrans({"O": "0", "o": "0", "D": "0", "Q": "0", "I": "1", "l": "1", "|": "1", "Z": "2", "S": "5",
                        "B": "8"})
# 2029-03-10, 2029 - 03 - 10, 2029.03.10, 2029/03/10 (separator may be read as space); characters that OCR swaps
# for digits are allowed and translated afterwards
_D = r"[0-9OoDQIl|ZSB]"
_DATE = re.compile(rf"(?<![0-9])(2{_D}{_D}{_D})\s*[-./\s]\s*({_D}{_D})\s*[-./\s]\s*({_D}{_D})(?![0-9])")
YEARS = (2020, 2045)


def find_dates(text: str) -> list[dt.date]:
    """All plausible YYYY-MM-DD dates in OCR text, in reading order, without duplicates."""
    out: list[dt.date] = []
    for m in _DATE.finditer(text or ""):
        y, mo, d = (int(g.translate(_DIGIT)) for g in m.groups())
        if not YEARS[0] <= y <= YEARS[1]:
            continue
        try:
            day = dt.date(y, mo, d)
        except ValueError:
            continue
        if day not in out:
            out.append(day)
    return out


def trusted_expiry(reads: list[tuple[dt.date, ...]], need: int = 2,
                   allowed: set[dt.date] | None = None) -> dt.date | None:
    """Printed expiry of ONE box from its last OCR reads -> the date to use for FEFO / inventory, or None.

    reads   : one tuple per read of this box (oldest first) with the dates found in that read's spine text;
              an empty tuple = no date read that time (too far, hand in front, date not in view).
    need    : how many reads must show the date (same as matching.require_consistent).
    allowed : expiry dates the inventory has for the requested product (None = no inventory loaded).
    """
    # Strictest rule: K03 / K17 differ by one digit, so ANY doubt -> None (FEFO then falls back to the inventory
    # order and the doctor still confirms). Empty reads are not counter-evidence.
    seen = [r for r in reads if r]
    if len(seen) < need or any(len(r) > 1 for r in seen):    # too few reads, or two dates in one read
        return None
    days = {r[0] for r in seen}
    if len(days) != 1:                                        # a different date read even once
        return None
    day = days.pop()
    if allowed is not None and day not in allowed:            # not a date this product has on the shelf
        return None
    return day

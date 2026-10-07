"""
Katalog GTIN -> produk, diameter, panjang (data/gtin_catalog.csv).

Hanya entri dengan verified = 1 yang dipakai untuk memutuskan MATCH. Entri baru (dari
scripts/enroll_gtin.py atau draf dari foto) mulai sebagai verified = 0 sampai seseorang mencocokkan
GTIN dengan tulisan di kemasan dan menandainya benar.
"""
from __future__ import annotations

import csv
import datetime as dt
import pathlib
import re
from dataclasses import asdict, dataclass, fields

FIELDS = ["gtin", "product", "diameter_mm", "length_mm", "brand", "ref", "verified", "verified_by",
          "verified_at", "note"]


def norm_product(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower()).split())


@dataclass
class CatalogEntry:
    gtin: str
    product: str
    diameter_mm: float | None = None
    length_mm: float | None = None
    brand: str = ""
    ref: str = ""
    verified: bool = False
    verified_by: str = ""
    verified_at: str = ""
    note: str = ""

    def key(self):
        return (norm_product(self.product), self.diameter_mm, self.length_mm)

    def __str__(self) -> str:
        size = "" if self.diameter_mm is None else f" {self.diameter_mm:g} x {self.length_mm:g} mm"
        return f"{self.product}{size}"


def _num(v):
    v = (v or "").strip().replace(",", ".")
    return float(v) if v else None


class Catalog:
    def __init__(self, path: str | pathlib.Path | None = None):
        self.path = pathlib.Path(path) if path else None
        self.entries: dict[str, CatalogEntry] = {}
        if self.path and self.path.exists():
            with open(self.path, newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    g = (r.get("gtin") or "").strip()
                    if not g:
                        continue
                    self.entries[g] = CatalogEntry(
                        gtin=g, product=norm_product(r.get("product", "")),
                        diameter_mm=_num(r.get("diameter_mm")), length_mm=_num(r.get("length_mm")),
                        brand=r.get("brand", "") or "", ref=r.get("ref", "") or "",
                        verified=str(r.get("verified", "")).strip() in ("1", "true", "True", "ya"),
                        verified_by=r.get("verified_by", "") or "", verified_at=r.get("verified_at", "") or "",
                        note=r.get("note", "") or "")

    def get(self, gtin: str) -> CatalogEntry | None:
        return self.entries.get(gtin)

    def gtins_for(self, product: str, diameter: float | None, length: float | None) -> list[str]:
        p = norm_product(product)
        return [g for g, e in self.entries.items() if e.product == p and e.diameter_mm == diameter
                and e.length_mm == length]

    def upsert(self, entry: CatalogEntry) -> None:
        entry.product = norm_product(entry.product)
        self.entries[entry.gtin] = entry

    def verify(self, gtin: str, by: str) -> None:
        e = self.entries[gtin]
        e.verified, e.verified_by, e.verified_at = True, by, dt.date.today().isoformat()

    def save(self, path: str | pathlib.Path | None = None) -> None:
        path = pathlib.Path(path or self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            for g in sorted(self.entries):
                row = asdict(self.entries[g])
                row["verified"] = 1 if row["verified"] else 0
                row["diameter_mm"] = "" if row["diameter_mm"] is None else f"{row['diameter_mm']:g}"
                row["length_mm"] = "" if row["length_mm"] is None else f"{row['length_mm']:g}"
                w.writerow({k: row[k] for k in FIELDS})


assert [f.name for f in fields(CatalogEntry)] == FIELDS

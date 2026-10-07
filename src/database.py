"""Local medicine database (SQLite) — the "source of truth" from design doc section 2.

Stores, per SKU: expected label text (for OCR match), an HSV color-histogram
signature, and a shape signature (Hu moments via cv2.matchShapes reference contour
is computed on the fly from a reference image, not stored raw, to keep rows small).
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

import cv2
import numpy as np


SCHEMA = """
CREATE TABLE IF NOT EXISTS medicines (
    sku TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    dose TEXT,
    label_text TEXT NOT NULL,       -- expected OCR text (lowercase, whitespace-normalized)
    color_hist TEXT NOT NULL,       -- JSON list, HSV histogram signature
    ref_image_path TEXT,            -- reference crop used to derive the shape signature
    stock_count INTEGER DEFAULT 0
);
"""


@dataclass
class MedicineRecord:
    sku: str
    name: str
    dose: str
    label_text: str
    color_hist: np.ndarray
    ref_image_path: str | None
    stock_count: int = 0


class MedicineDB:
    def __init__(self, path: str):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.execute(SCHEMA)
        self.conn.commit()

    def upsert(self, rec: MedicineRecord):
        self.conn.execute(
            """INSERT INTO medicines (sku, name, dose, label_text, color_hist, ref_image_path, stock_count)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(sku) DO UPDATE SET
                 name=excluded.name, dose=excluded.dose, label_text=excluded.label_text,
                 color_hist=excluded.color_hist, ref_image_path=excluded.ref_image_path,
                 stock_count=excluded.stock_count""",
            (
                rec.sku,
                rec.name,
                rec.dose,
                rec.label_text,
                json.dumps(rec.color_hist.tolist()),
                rec.ref_image_path,
                rec.stock_count,
            ),
        )
        self.conn.commit()

    def all_records(self) -> list[MedicineRecord]:
        rows = self.conn.execute(
            "SELECT sku, name, dose, label_text, color_hist, ref_image_path, stock_count FROM medicines"
        ).fetchall()
        out = []
        for r in rows:
            out.append(
                MedicineRecord(
                    sku=r[0],
                    name=r[1],
                    dose=r[2],
                    label_text=r[3],
                    color_hist=np.array(json.loads(r[4]), dtype=np.float32),
                    ref_image_path=r[5],
                    stock_count=r[6],
                )
            )
        return out

    def increment_count(self, sku: str, by: int = 1):
        self.conn.execute(
            "UPDATE medicines SET stock_count = stock_count + ? WHERE sku = ?", (by, sku)
        )
        self.conn.commit()

    def close(self):
        self.conn.close()


def color_histogram(img_bgr: np.ndarray, bins: int = 16) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [bins, bins], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist.flatten()

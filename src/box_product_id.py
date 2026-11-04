"""
Product look model, role A (owner decision 2 Nov 2026): reading ORDER and a CROSS-CHECK, never identity evidence.

An image classifier over the straightened spine (results/id_model_20261102, trained on Leo's Drive photos only) gives a
probability per PRODUCT (not size, date or lot). It is used in two ways, both on the side of caution:

  order        a box the model is sure is ANOTHER product is not locked and is read last; a box the model gives a high
               probability for the requested product is read first (TrackState.hint, the same scale as the colour hint)
  cross-check  `veto_reads` model reads in a row sure of another product -> TrackState.look_veto; stable() then turns
               a MATCH into CONFIRM (yellow, not picked). The model can only hold a green back, never make one.

A product the model was not trained on (no class contains its name) switches the model off for that request: it is
not asked about products it cannot know. The two-source rule (src/box_verifier.combined_match) is untouched.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np

from src.box_crops import rectify

INPUT_HW = (448, 64)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def classes_for(product: str, classes: list[str]) -> list[int]:
    """Model classes of a catalogue product name ("sapphire" -> Sapphire 3 and Sapphire NC24). Empty = unknown."""
    p = norm(product)
    return [i for i, c in enumerate(classes) if p and p in norm(c)] if p else []


def to_input(spine: np.ndarray) -> np.ndarray:
    h, w = INPUT_HW
    x = cv2.resize(spine, (w, h), interpolation=cv2.INTER_AREA)[:, :, ::-1].astype(np.float32) / 255
    return ((x - MEAN) / STD).transpose(2, 0, 1)


class ProductLook:
    def __init__(self, cfg: dict, root: Path):
        import openvino as ov
        self.cfg = cfg
        meta = json.loads((root / cfg["classes"]).read_text(encoding="utf-8"))
        self.classes = meta["classes"]
        core = ov.Core()
        self.models = []
        device = cfg.get("device", "CPU")
        if device not in core.available_devices and device != "CPU":
            device = "CPU"                               # no iGPU on this machine: the CPU still works
        self.device = device
        for path in ([cfg["model"]] if isinstance(cfg["model"], str) else cfg["model"]):   # list = ensemble (mean)
            m = core.read_model(str(root / path))
            props = {"PERFORMANCE_HINT": "LATENCY"}
            if device == "CPU" and cfg.get("threads"):
                props["INFERENCE_NUM_THREADS"] = int(cfg["threads"])   # leave the other cores to OCR (critical path)
            self.models.append(core.compile_model(m, device, props))
        self.veto_top = float(cfg.get("veto_top", 0.85))
        self.veto_req = float(cfg.get("veto_req", 0.02))
        self.veto_reads = int(cfg.get("veto_reads", 3))
        self.order_min = float(cfg.get("order_min", 0.5))
        self.req_classes: list[int] = []
        self.last_ms = 0.0
        self._busy, self._done, self.error = False, None, ""
        self.every = int(cfg.get("every_frames", 5))      # live loop: submit the boxes at most every N frames
        self.last_submit = -1000

    def set_request(self, product: str) -> None:
        self.req_classes = classes_for(product, self.classes)
        self._done = None                                # a job for the previous request is not applied

    @property
    def active(self) -> bool:
        return bool(self.req_classes)

    def probs(self, frame: np.ndarray, polys: list) -> np.ndarray:
        """Class probabilities, one row per polygon (spines straightened as in training)."""
        import time
        t = time.perf_counter()
        x = np.stack([to_input(rectify(frame, p)[0]) for p in polys]).astype(np.float32)
        out = []
        for c in self.models:
            z = np.asarray(c(x)[0], np.float32)
            z = np.exp(z - z.max(axis=1, keepdims=True))
            out.append(z / z.sum(axis=1, keepdims=True))
        self.last_ms = (time.perf_counter() - t) * 1000
        return np.mean(out, axis=0)

    def judge(self, p: np.ndarray) -> tuple[float, str, float, bool]:
        """-> (p of the requested product, top class, its p, sure of another product)."""
        p_req = float(p[self.req_classes].sum())
        top = int(np.argmax(p))
        other = top not in self.req_classes and float(p[top]) >= self.veto_top and p_req < self.veto_req
        return p_req, self.classes[top], float(p[top]), other

    def update(self, states: list, frame: np.ndarray, polys: list) -> None:
        """Model look of the given boxes -> reading order hint and the cross-check flag (blocking; tests, offline)."""
        if not self.active or not polys:
            return
        self.apply(states, [st.epoch for st in states], self.probs(frame, polys))

    def submit(self, states: list, frame: np.ndarray, polys: list) -> bool:
        """Live loop: run the model in a background thread so detection and OCR never wait for it (in the live loop
        it took 0.5-2 s per refresh while sharing the CPU with OCR). One job at a time; a refresh that comes while
        the model is busy is skipped (the next one carries newer boxes). -> False when skipped."""
        import threading
        if not self.active or not polys or self._busy:
            return False
        self._busy = True
        job = (list(states), [st.epoch for st in states], frame.copy(), [np.asarray(p, np.float32).copy() for p in polys])

        def work():
            try:
                self._done = (job[0], job[1], self.probs(job[2], job[3]))
            except Exception as exc:                     # never takes the live loop down; logged by the caller
                self._done, self.error = None, repr(exc)
            finally:
                self._busy = False
        threading.Thread(target=work, daemon=True).start()
        return True

    def drain(self) -> None:
        """Main thread, every frame: apply a finished background job. Boxes whose number was doubted since (epoch
        changed) are skipped: the look belonged to the crop taken before."""
        done, self._done = self._done, None
        if done is not None and self.active:
            self.apply(*done)

    def apply(self, states: list, epochs: list, probs: np.ndarray) -> None:
        for st, ep, p in zip(states, epochs, probs):
            if st.epoch != ep:
                continue
            p_req, top, p_top, other = self.judge(p)
            st.look_p, st.look_top = p_req, f"{top} {p_top:.2f}"
            st.look_other = (st.look_other + 1) if other else 0
            st.look_veto = f"look: {top} {p_top:.2f}" if st.look_other >= self.veto_reads else ""
            self.adjust(st)

    def adjust(self, st) -> None:
        """Reading order from the latest look; also called after every colour-hint refresh, which overwrites hint."""
        if not self.active or st.look_p < 0:
            return
        if st.look_other:
            st.hint = 0.0                                      # sure of another product: never locked, read last
        elif st.look_p >= self.order_min:
            st.hint = max(st.hint, 0.5 + 0.5 * st.look_p)      # read before boxes that look like other products


def load(cfg: dict, root: Path) -> ProductLook | None:
    c = (cfg or {}).get("product_look") or {}
    return ProductLook(c, root) if c.get("enabled") else None

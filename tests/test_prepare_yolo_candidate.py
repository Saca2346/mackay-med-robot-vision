"""Candidate checks preserve evidence and compare polygons one to one."""
import csv

import numpy as np
import pytest

from scripts.prepare_yolo_candidate import fresh_output, match_detections, photo_log_summary
from src.box_detector import Det


def test_candidate_output_rejects_evidence_and_existing_run(tmp_path):
    results = tmp_path / "results"
    previous = results / "previous"
    previous.mkdir(parents=True)
    for path in (tmp_path / "models", results, previous, results / "bukti_demo/new"):
        with pytest.raises(ValueError):
            fresh_output(path, tmp_path)
    assert fresh_output(results / "new", tmp_path) == results / "new"


def test_matching_exposes_duplicate_and_wrong_class_instead_of_count_parity():
    box = np.array([[0, 0], [10, 0], [10, 20], [0, 20]], dtype=np.float32)
    left = [Det("box", 0.9, box), Det("box", 0.9, box + 50)]
    right = [Det("box", 0.9, box), Det("box", 0.8, box.copy()), Det("size_label", 0.9, box + 50)]
    matched = match_detections(left, right)
    assert len(matched["matches"]) == 1
    assert matched["unmatched_left"] == [1]
    assert len(matched["unmatched_right"]) == 2


def test_photo_summary_does_not_confuse_brand_only_candidate_with_requested_size(tmp_path):
    with (tmp_path / "bacaan_photo.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["decision", "identity"])
        writer.writeheader()
        writer.writerows([{"decision": "candidate", "identity": "chroma||||"},
                          {"decision": "candidate", "identity": "chroma|2.75|33||"}])
    summary = photo_log_summary(tmp_path, "chroma|2.75|33|")
    assert summary["pick_rows"] == []
    assert len(summary["requested_identity_readings"]) == 1

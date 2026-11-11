"""Session evidence must retain console input/errors and truthful paired photos."""
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

from src.box_photo_evidence import PhotoEvidence


def test_transcript_keeps_stdout_stderr_input_and_failure_code(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/run_logged.py"
    logfile = tmp_path / "cmd.txt"
    child = "import sys; print('request='+input()); print('reader failed',file=sys.stderr); sys.exit(7)"
    result = subprocess.run([sys.executable, str(script), "--log", str(logfile), "--", sys.executable, "-c", child],
                            input="angiolite 4 x 19\n", capture_output=True, text=True, timeout=20)
    assert result.returncode == 7
    transcript = logfile.read_text(encoding="utf-8")
    for expected in ("request=angiolite 4 x 19", "reader failed", "exit_code: 7", "[session] command:"):
        assert expected in transcript and expected in result.stdout


def test_launch_failure_is_written_to_transcript(tmp_path):
    from scripts.run_logged import run_logged
    logfile = tmp_path / "cmd.txt"
    assert run_logged([str(tmp_path / "missing.exe")], logfile) == 1
    assert "launch failed" in logfile.read_text()


def test_photos_preserve_raw_pixels_status_and_do_not_overwrite_between_runs(tmp_path):
    camera = np.full((40, 60, 3), 60, np.uint8)
    view = np.full((80, 60, 3), 170, np.uint8)
    for verified in (False, True):
        photos = PhotoEvidence(tmp_path)
        row = photos.save(camera, view, "VERIFIED" if verified else "PENDING",
                          {"identity_verified": verified, "request": "angiolite 4 x 19", "wall_seconds": 2.3})
        assert np.array_equal(cv2.imread(str(tmp_path / row["photos"]["camera"])), camera)
        assert np.array_equal(cv2.imread(str(tmp_path / row["photos"]["view"])), view)
    rows = [json.loads(s) for s in (tmp_path / "index.jsonl").read_text().splitlines()]
    assert [r["identity_verified"] for r in rows] == [False, True]
    assert len(list(tmp_path.glob("*.png"))) == 4


def test_photo_write_failure_is_not_logged_as_success(tmp_path, monkeypatch):
    monkeypatch.setattr(cv2, "imwrite", lambda *a: False)
    with pytest.raises(OSError, match="Could not save"):
        PhotoEvidence(tmp_path).save(np.zeros((2, 2, 3), np.uint8), None, "PENDING", {})
    assert not (tmp_path / "index.jsonl").exists()


def test_queued_photos_freeze_camera_pixels_and_finish_before_close(tmp_path):
    photos = PhotoEvidence(tmp_path)
    frame = np.full((30, 40, 3), 55, np.uint8)
    for _ in range(6):
        photos.submit(frame, frame, "MANUAL", {"identity_verified": False})
    frame[:] = 0
    photos.close()
    rows = [json.loads(s) for s in (tmp_path / "index.jsonl").read_text().splitlines()]
    assert len(rows) == 6
    assert all(np.all(cv2.imread(str(tmp_path / r["photos"]["camera"])) == 55) for r in rows)


def test_background_write_failure_surfaces_at_close(tmp_path, monkeypatch):
    monkeypatch.setattr(cv2, "imwrite", lambda *a: False)
    photos = PhotoEvidence(tmp_path)
    frame = np.zeros((2, 2, 3), np.uint8)
    photos.submit(frame, frame, "PENDING", {})
    with pytest.raises(OSError, match="Could not save"):
        photos.close()

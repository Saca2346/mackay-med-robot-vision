"""Session isolation and preflight checks use the settings the demo will run."""
import pathlib
import sys
from types import SimpleNamespace

import pytest
import yaml

from scripts import demo_preflight, sesi_config


def test_session_outputs_are_isolated_without_changing_source_or_pipeline(tmp_path):
    source = tmp_path / "config.yaml"
    config = {
        "model": {"weights": "models/box_v4s_960_openvino_model", "imgsz": [544, 960]},
        "verify": {"min_score": 2, "trust_unverified_catalog": False},
        "catalog": ["chroma", "sequent please neo"],
        "output": {"view_width": 900, "screens_dir": "results/bukti_old/screens",
                   "log_csv": "results/bukti_old/selection.csv", "verify_log_csv": "results/bukti_old/verify.csv"},
    }
    source.write_text(yaml.safe_dump(config), encoding="utf-8")
    original = source.read_bytes()
    destinations = []
    for name in ("first", "second"):
        folder = tmp_path / name
        target = sesi_config.create_session_config(source, folder)
        saved = yaml.safe_load(target.read_text(encoding="utf-8"))
        for key in ("model", "verify", "catalog"):
            assert saved[key] == config[key]
        assert saved["output"]["view_width"] == 900
        paths = [pathlib.Path(saved["output"][key])
                 for key in ("screens_dir", "log_csv", "verify_log_csv")]
        assert all(path.parent == folder for path in paths)
        # The pipeline derives PICK crops / grasp output from these same roots.
        assert (paths[2].parent / "grasp.jsonl").is_relative_to(folder)
        destinations.extend(paths)
    assert len(set(destinations)) == 6
    assert source.read_bytes() == original


@pytest.mark.parametrize("contents", ["", "- invalid\n", "output: []\n", "output: [\n"])
def test_invalid_config_fails_before_creating_session(tmp_path, capsys, contents):
    source = tmp_path / "bad.yaml"
    source.write_text(contents, encoding="utf-8")
    folder = tmp_path / "session"
    assert sesi_config.main([str(source), str(folder)]) == 1
    assert "Session config failed:" in capsys.readouterr().err
    assert not folder.exists()


def test_missing_config_fails_without_replacing_existing_session(tmp_path, capsys):
    folder = tmp_path / "session"
    folder.mkdir()
    target = folder / "config_sesi.yaml"
    target.write_text("existing session\n", encoding="utf-8")
    assert sesi_config.main([str(tmp_path / "missing.yaml"), str(folder)]) == 1
    assert "Session config failed:" in capsys.readouterr().err
    assert target.read_text(encoding="utf-8") == "existing session\n"


def test_preflight_checks_supplied_config_and_inventory_copy(tmp_path, monkeypatch, capsys):
    weights = tmp_path / "chosen_model.pt"
    config = tmp_path / "config_sesi.yaml"
    config.write_text(yaml.safe_dump({
        "model": {"weights": str(weights), "imgsz": [544, 960]},
        "camera": {"backend": "realsense", "width": 1280, "height": 800},
    }), encoding="utf-8")
    inventory = tmp_path / "inventaris_demo.csv"
    inventory.write_text("box_id,status\n" + "".join(f"K{i:02d},terverifikasi\n" for i in range(1, 21)),
                         encoding="utf-8")
    original = inventory.read_bytes()
    device = SimpleNamespace(supports=lambda _: True,
                             get_info=lambda field: "3.2" if field == "usb" else "test camera")
    monkeypatch.setitem(sys.modules, "pyrealsense2", SimpleNamespace(
        context=lambda: SimpleNamespace(devices=[device]),
        camera_info=SimpleNamespace(usb_type_descriptor="usb", name="name")))
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(
        virtual_memory=lambda: SimpleNamespace(available=4e9)))
    args = ["--config", str(config), "--inventaris", str(inventory)]
    assert demo_preflight.main(args) == 1
    output = capsys.readouterr().out
    assert str(config) in output and str(weights) in output and str(inventory) in output
    assert "20 verified, 0 taken" in output
    weights.touch()
    assert demo_preflight.main(args) == 0
    assert "READY: 0 FAIL" in capsys.readouterr().out
    assert inventory.read_bytes() == original


@pytest.mark.parametrize("backend", ["msmf", "dshow"])
def test_webcam_preflight_works_without_realsense_sdk(monkeypatch, backend):
    from src import box_camera
    closed = []
    cam = SimpleNamespace(release=lambda: closed.append(True))
    config = {"backend": backend, "index": 2, "width": 1920, "height": 1080}
    received = []

    def open_camera(cfg):
        received.append(cfg)
        return cam, 2, (1920, 1080)

    monkeypatch.setattr(box_camera, "open_camera", open_camera)
    monkeypatch.setitem(sys.modules, "pyrealsense2", None)
    demo_preflight.results.clear()
    demo_preflight.check_camera(config)
    assert received == [config] and closed == [True]
    assert not any(status == "FAIL" or "RealSense" in what for status, what, _ in demo_preflight.results)
    assert any(status == "OK" and "camera #2" in detail for status, _, detail in demo_preflight.results)


def test_webcam_preflight_fails_when_no_readable_camera(monkeypatch):
    from src import box_camera
    monkeypatch.setattr(box_camera, "open_camera", lambda cfg: (None, None, None))
    demo_preflight.results.clear()
    demo_preflight.check_camera({"backend": "msmf"})
    assert any(status == "FAIL" and what == "USB webcam" for status, what, _ in demo_preflight.results)


def test_webcam_preflight_warns_on_laptop_resolution_and_releases_camera(monkeypatch):
    from src import box_camera
    closed = []
    cam = SimpleNamespace(release=lambda: closed.append(True))
    monkeypatch.setattr(box_camera, "open_camera", lambda cfg: (cam, 0, (640, 480)))
    demo_preflight.results.clear()
    demo_preflight.check_camera({"backend": "msmf", "width": 1920, "height": 1080})
    assert closed == [True]
    assert any(status == "WARN" and "external webcam" in detail for status, _, detail in demo_preflight.results)

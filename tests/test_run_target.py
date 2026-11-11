"""Both cameras keep separate calibrated detector profiles and isolated evidence paths."""
import hashlib

import pytest
import yaml

from scripts.run_target import ROOT, parser, prepare_session


@pytest.mark.parametrize("camera", ["webcam", "realsense"])
def test_camera_profile_keeps_expected_model_and_inventory_copy(tmp_path, camera):
    before = hashlib.sha256((ROOT / "data/rak/inventaris.csv").read_bytes()).hexdigest()
    args = parser().parse_args(["--camera", camera, "--output", str(tmp_path / camera), "--prepare-only"])
    folder, config, inventory = prepare_session(args)
    cfg = yaml.safe_load(config.read_text())
    if camera == "webcam":
        assert cfg["camera"]["backend"] == "dshow" and cfg["camera"]["index"] == 1
        assert cfg["model"]["weights"] == "models/box_y26_maxB_544x960_openvino_model"   # 20-10: webcam model trained on all recordings
    else:
        assert cfg["camera"]["backend"] == "realsense"
        assert cfg["model"]["weights"] == "models/box_v4s_960_openvino_model"
    assert cfg["verify"]["min_score"] == 2 and not cfg["verify"]["trust_unverified_catalog"]
    assert cfg["visual"]["target_seconds"] == 2 and cfg["visual"]["workers"] == 2
    assert inventory.read_bytes() == (ROOT / "data/rak/inventaris.csv").read_bytes()
    assert hashlib.sha256((ROOT / "data/rak/inventaris.csv").read_bytes()).hexdigest() == before
    assert str(folder) in cfg["output"]["screens_dir"]


def test_realsense_rejects_webcam_index_and_existing_output(tmp_path):
    args = parser().parse_args(["--camera", "realsense", "--index", "1", "--output", str(tmp_path / "bad")])
    with pytest.raises(ValueError, match="only for a webcam"):
        prepare_session(args)
    args = parser().parse_args(["--camera", "webcam", "--output", str(tmp_path)])
    with pytest.raises(FileExistsError):
        prepare_session(args)


def test_candidate_profile_is_isolated_and_rejects_wrong_camera(tmp_path):
    source = tmp_path / "candidate.yaml"
    cfg = yaml.safe_load((ROOT / "config/box_pipeline_config_focus.yaml").read_text())
    cfg["model"]["weights"] = "results/candidate/model"
    source.write_text(yaml.safe_dump(cfg))
    original = source.read_bytes()
    args = parser().parse_args(["--camera", "webcam", "--config", str(source), "--output", str(tmp_path / "run")])
    folder, config, _ = prepare_session(args)
    assert yaml.safe_load(config.read_text())["model"]["weights"] == "results/candidate/model"
    assert source.read_bytes() == original
    assert folder.is_dir()
    args = parser().parse_args(["--camera", "realsense", "--config", str(source), "--output", str(tmp_path / "wrong")])
    with pytest.raises(ValueError, match="does not match"):
        prepare_session(args)
    assert not (tmp_path / "wrong").exists()


@pytest.mark.parametrize("camera", ["webcam", "realsense"])
def test_no_barcode_is_saved_only_in_session_with_verification_rules_intact(tmp_path, camera):
    import json
    source = ROOT / "config" / ("box_pipeline_config_focus.yaml" if camera == "webcam"
                                else "box_pipeline_config_focus_realsense.yaml")
    original = source.read_bytes()
    args = parser().parse_args(["--camera", camera, "--no-barcode", "--output", str(tmp_path / camera)])
    folder, config, _ = prepare_session(args)
    cfg = yaml.safe_load(config.read_text())
    assert cfg["verify"]["decode_barcodes"] is False
    assert cfg["verify"]["min_score"] == cfg["matching"]["require_consistent"] == 2
    assert not json.loads((folder / "session.json").read_text())["barcode_decoding"]
    assert source.read_bytes() == original


@pytest.mark.parametrize("source", [None, "recording.mp4"])
def test_take_routes_all_outputs_and_transcript_to_its_own_folder(tmp_path, monkeypatch, source):
    from pathlib import Path
    from scripts import run_target
    argv = ["--camera", "webcam", "--output", str(tmp_path / "session"), "--max-frames", "5", "--no-window"]
    if source:
        argv += ["--source", source]
    args = parser().parse_args(argv)
    folder, config, inventory = prepare_session(args)
    commands = []
    monkeypatch.setattr(run_target.subprocess, "call", lambda cmd, **kwargs: commands.append(cmd) or 0)
    assert run_target.run_request(args, folder, config, inventory, "angiolite 4 x 19") == 0
    command = commands[0]
    transcript = Path(command[command.index("--log") + 1])
    take = transcript.parent
    assert take.parent == folder and take.name.startswith("take_")
    for option in ("--record-vis", "--evidence-photos", "--debug-readings"):
        assert Path(command[command.index(option) + 1]).parent == take
    saved = yaml.safe_load(Path(command[command.index("--config") + 1]).read_text())
    assert Path(saved["output"]["screens_dir"]).parent == take
    assert command[command.index("--inventaris") + 1] == str(inventory)
    assert "--verify-target" in command and "--no-window" in command
    if source:
        assert "--record" not in command and command[command.index("--source") + 1] == source
    else:
        assert Path(command[command.index("--record") + 1]).parent == take


def test_explicit_region_is_recorded_and_forwarded_without_changing_identity_policy(tmp_path, monkeypatch):
    import json
    from scripts import run_target
    args = parser().parse_args(['--camera','realsense','--roi','0.45,0.025,0.55,0.93',
                               '--output',str(tmp_path/'session')])
    folder, config, inventory = prepare_session(args)
    assert json.loads((folder/'session.json').read_text())['roi'] == [.45,.025,.55,.93]
    commands=[]
    monkeypatch.setattr(run_target.subprocess,'call',lambda cmd,**kwargs: commands.append(cmd) or 0)
    assert run_target.run_request(args,folder,config,inventory,'permanent sled bag 2029-03-24') == 0
    cmd=commands[0]
    assert cmd[cmd.index('--roi')+1] == '0.45,0.025,0.55,0.93'
    assert '--verify-target' in cmd
    assert yaml.safe_load(config.read_text())['verify']['min_score'] == 2


@pytest.mark.parametrize('camera', ['webcam', 'realsense'])
def test_eleven_profile_records_count_and_show_all_without_changing_camera_or_verifier(tmp_path, monkeypatch, camera):
    import json
    from scripts import run_target
    args = parser().parse_args(['--camera', camera, '--config',
        str(ROOT/f'config/box_pipeline_config_{camera}_eleven.yaml'), '--output', str(tmp_path/'run')])
    folder, config, inventory = prepare_session(args)
    cfg = yaml.safe_load(config.read_text())
    assert cfg['camera']['backend'] == ('realsense' if camera == 'realsense' else 'dshow')
    assert cfg['selection']['expected_count'] == 11 and cfg['visual']['show_all']
    assert len(set(cfg['visual']['request_ids'])) == 11
    assert not cfg['verify']['decode_barcodes'] and cfg['verify']['min_score'] == 2
    meta = json.loads((folder/'session.json').read_text())
    assert meta['show_all'] and meta['expected_count'] == 11
    calls = []
    monkeypatch.setattr(run_target.subprocess, 'call', lambda cmd, **kw: calls.append(cmd) or 0)
    run_target.run_request(args, folder, config, inventory, 'angiolite 4 x 19')
    assert calls[0][calls[0].index('--expected')+1] == '11'


def test_show_all_and_count_overrides_are_explicit_and_validated(tmp_path):
    args = parser().parse_args(['--camera','webcam','--show-all','--expected','11','--output',str(tmp_path/'run')])
    _, config, _ = prepare_session(args)
    cfg = yaml.safe_load(config.read_text())
    assert cfg['visual']['show_all'] and cfg['selection']['expected_count'] == 11
    args = parser().parse_args(['--camera','webcam','--expected','0','--output',str(tmp_path/'bad')])
    with pytest.raises(ValueError, match='expected count'):
        prepare_session(args)


@pytest.mark.parametrize('camera', ['webcam', 'realsense'])
def test_calibration_first_keeps_camera_roi_count_and_unique_evidence(tmp_path, monkeypatch, camera):
    import json
    from pathlib import Path
    from scripts import run_target
    args = parser().parse_args(['--camera',camera,'--calibrate-first','--expected','11',
        '--roi','0.1,0.1,0.9,0.95','--output',str(tmp_path/camera)])
    folder, config, _ = prepare_session(args)
    cfg = yaml.safe_load(config.read_text())
    assert cfg['visual']['transparent_header']
    calls = []
    def launch(command, **kwargs):
        calls.append(command)
        out = Path(command[command.index('--out')+1]);out.mkdir()
        name = command[command.index('--sesi')+1]
        (out/(name+'_kalibrasi.json')).write_text(json.dumps({'verdict':'PASS'}))
        return 0
    monkeypatch.setattr(run_target.subprocess, 'call', launch)
    assert run_target.run_calibration(args, folder, config) == 0
    command = calls[0]
    assert command[command.index('--config')+1] == str(config)
    assert command[command.index('--min-boxes')+1] == '11'
    assert command[command.index('--roi')+1] == '0.1,0.1,0.9,0.95'
    assert command[command.index('--log')+1] == str(folder/'cmd_calibration.txt')
    metadata = json.loads((folder/'session.json').read_text())
    assert metadata['calibration']['verdict'] == 'PASS' and metadata['calibration_first']


@pytest.mark.parametrize('exit_code,verdict', [(1,None),(0,None),(0,'FAIL')])
def test_calibration_failure_or_missing_result_cannot_report_success(tmp_path, monkeypatch, exit_code, verdict):
    import json
    from pathlib import Path
    from scripts import run_target
    args = parser().parse_args(['--camera','webcam','--calibrate-first','--output',str(tmp_path/'run')])
    folder, config, _ = prepare_session(args)
    def launch(command, **kwargs):
        if verdict:
            out=Path(command[command.index('--out')+1]);out.mkdir()
            (out/(folder.name+'_kalibrasi.json')).write_text(json.dumps({'verdict':verdict}))
        return exit_code
    monkeypatch.setattr(run_target.subprocess,'call',launch)
    assert run_target.run_calibration(args,folder,config) != 0


@pytest.mark.parametrize('calibration_code', [0,1])
def test_main_calibrates_before_detection_and_stops_after_cancel(tmp_path, monkeypatch, calibration_code):
    from scripts import run_target
    events=[]
    monkeypatch.setattr(run_target,'run_calibration',lambda *a: events.append('calibration') or calibration_code)
    monkeypatch.setattr(run_target,'run_request',lambda *a: events.append('detection') or 0)
    code=run_target.main(['--camera','webcam','--calibrate-first','--output',str(tmp_path/'run')])
    assert events == (['calibration','detection'] if calibration_code == 0 else ['calibration'])
    assert code == calibration_code


def test_replay_cannot_open_unrelated_live_calibration(tmp_path):
    args=parser().parse_args(['--camera','webcam','--calibrate-first','--source','recorded.mp4',
                             '--output',str(tmp_path/'run')])
    with pytest.raises(ValueError,match='live camera'):
        prepare_session(args)
    assert not (tmp_path/'run').exists()

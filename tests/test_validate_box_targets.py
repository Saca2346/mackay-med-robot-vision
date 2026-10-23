"""Individual evidence must preserve printed dates, failures and pending owner review."""
import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.box_target_review import write_review
from scripts.validate_box_targets import inventory_plan, summarize


def inventory(path, dates):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["box_id", "product", "diameter", "length", "expiry", "status"])
        writer.writeheader()
        for i, date in enumerate(dates):
            writer.writerow(dict(box_id=f"K{i+1:02d}", product="permanent sled bag", expiry=date, status="terverifikasi"))


def test_each_identical_box_requires_its_own_printed_date_and_inventory_is_unchanged(tmp_path):
    path = tmp_path / "inventory.csv"
    inventory(path, ["2029-03-12", "2029-04-26"])
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    plan = inventory_plan(path)
    assert [r["request"] for r in plan] == ["permanent sled bag 2029-03-12", "permanent sled bag 2029-04-26"]
    assert all(r["owner_validation"] == "PENDING" for r in plan)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


@pytest.mark.parametrize("dates", [["", ""], ["", "2029-03-12"]])
def test_indistinguishable_rows_are_not_claimed_as_separate_physical_tests(tmp_path, dates):
    path = tmp_path / "inventory.csv"
    inventory(path, dates)
    with pytest.raises(ValueError, match="Indistinguishable"):
        inventory_plan(path)


def trial(tmp_path, *, lost=False):
    take = tmp_path / "take_1"
    take.mkdir()
    states = [{"state": "CHECKING", "wall_seconds": .2, "sources": ""},
              {"state": "VERIFIED", "wall_seconds": 1.6, "sources": "text + REF"},
              {"state": "DISPLAY_READY", "wall_seconds": 1.7}]
    if lost:
        states.append({"state": "CHECKING", "wall_seconds": 3.0, "sources": "REF"})
    result = take / "verification/result.json"
    result.parent.mkdir()
    result.write_text(json.dumps({"fields": {"diameter_mm": {"value": None, "status": "not_read"}},
                                  "sources": "REF", "all_fields_read": False}))
    states.append({"state": "EVIDENCE_SAVED", "path": str(result.parent), "identity_verified": not lost,
                   "wall_seconds": 3.1})
    (take / "target_view.jsonl").write_text("\n".join(map(json.dumps, states)))
    return {"box_id": "K10", "request": "angiolite 4 x 19", "owner_validation": "PENDING"}


def test_later_loss_is_visible_and_no_unknown_field_is_filled_from_request(tmp_path):
    case = trial(tmp_path, lost=True)
    row = summarize(case, tmp_path, 0, {"ac": 1}, {"ac": 1}, True)
    assert row["status"] == "VERIFIED_THEN_LOST"
    assert row["machine_verified_once"] and not row["machine_verified_at_end"]
    assert row["later_verification_losses"] == 1
    assert row["observed_fields"]["diameter_mm"]["value"] is None
    assert row["owner_validation"] == "PENDING"


@pytest.mark.parametrize("ac,real_times", [(0, True), (1, False), (None, True)])
def test_unsupported_timing_cannot_be_reported_as_a_deadline_pass(tmp_path, ac, real_times):
    case = trial(tmp_path)
    row = summarize(case, tmp_path, 0, {"ac": ac}, {"ac": ac}, real_times)
    assert row["machine_verified_once"] and not row["timing_eligible"]
    assert row["status"] == "VERIFIED_TIMING_INELIGIBLE"


def test_failed_process_is_never_relabelled_passed_from_a_partial_green_log(tmp_path):
    case = trial(tmp_path)
    row = summarize(case, tmp_path, 1, {"ac": 1}, {"ac": 1}, True)
    assert row["status"] == "PROCESS_FAILED" and row["exit_code"] == 1


def test_review_escapes_data_and_cannot_turn_machine_results_into_owner_approval(tmp_path):
    case = {"box_id": "K10", "request": "</script><script>alert('x')</script>", "required_printed_date": None}
    write_review(tmp_path, [case], [{**case, "status": "VERIFIED_WITHIN_2S", "owner_validation": "ACCEPT"}])
    text = (tmp_path / "REVIEW.html").read_text(encoding="utf-8")
    assert case["request"] not in text
    assert json.loads((tmp_path / "review_index.json").read_text())["cases"][0]["owner_validation"] == "PENDING"
    with (tmp_path / "review_template.csv").open(encoding="utf-8-sig") as f:
        assert next(csv.DictReader(f))["owner_validation"] == "PENDING"


@pytest.mark.parametrize("remove", [False, True])
def test_changed_or_missing_runtime_input_stops_batch_with_auditable_reason(tmp_path, remove):
    from scripts.validate_box_targets import check_inputs, digest
    source = tmp_path / "runtime.py"
    source.write_text("before")
    expected = {str(source): digest(source)}
    check_inputs(expected, tmp_path, "before K01")
    if remove:
        source.unlink()
    else:
        source.write_text("after")
    with pytest.raises(SystemExit, match="Input/code drift"):
        check_inputs(expected, tmp_path, "after K01")
    reason = json.loads((tmp_path / "input_drift.json").read_text())
    assert reason["changed"] == [str(source)] and reason["stage"] == "after K01"


def test_text_timing_is_not_green_and_target_switch_cannot_pass_the_deadline(tmp_path):
    case = trial(tmp_path)
    log = tmp_path / 'take_1/target_view.jsonl'
    log.write_text(json.dumps({'state':'TEXT_READY', 'request':'angiolite 4 x 19 mm', 'wall_seconds':1.2}))
    row = summarize(case, tmp_path, 0, {'ac':1}, {'ac':1}, True)
    assert row['status'] == 'TEXT_READ_NOT_VERIFIED' and row['text_frame_seconds'] == 1.2
    assert not row['machine_verified_once'] and row['green_frame_seconds'] is None
    log.write_text('\n'.join(json.dumps(r) for r in [
        {'state':'VERIFIED','request':'angiolite 2.5 x 29 mm','wall_seconds':1.3},
        {'state':'DISPLAY_READY','request':'angiolite 2.5 x 29 mm','wall_seconds':1.4}]))
    row = summarize(case, tmp_path, 0, {'ac':1}, {'ac':1}, True)
    assert row['status'] == 'REQUEST_CHANGED' and not row['timing_eligible']


def test_manual_timer_reset_cannot_certify_initial_request_latency(tmp_path):
    case = trial(tmp_path)
    log=tmp_path/'take_1/target_view.jsonl'
    events=[json.loads(s) for s in log.read_text().splitlines()]
    events[2]['request_generation']=1
    log.write_text('\n'.join(map(json.dumps,events)))
    row=summarize(case,tmp_path,0,{'ac':1},{'ac':1},True)
    assert row['status']=='REQUEST_RESTARTED' and row['request_restarted']
    assert not row['timing_eligible']


@pytest.mark.parametrize('live', [False, True])
@pytest.mark.parametrize('camera', ['webcam', 'realsense'])
@pytest.mark.parametrize('headless', [False, True])
def test_sequential_trials_use_profile_targets_camera_and_fresh_evidence(tmp_path, monkeypatch, live, camera, headless):
    from scripts import validate_box_targets as batch, run_target
    import yaml
    profile = tmp_path/'profile.yaml'
    profile.write_text(yaml.safe_dump({'camera':{'backend':'realsense' if camera=='realsense' else 'dshow'},
                                     'visual':{'request_ids':['K01','K03']}}))
    source = tmp_path/'source.mp4'; source.write_bytes(b'video')
    Path(str(source)+'.times.csv').write_text('frame,t\n0,0\n')
    plan = [{'box_id':k,'request':r,'owner_validation':'PENDING'} for k,r in
            [('K01','permanent sled bag 2029-04-26'),('K02','angiolite 4 x 19'),
             ('K03','permanent sled bag 2029-03-10')]]
    monkeypatch.setattr(batch,'inventory_plan',lambda _:plan)
    monkeypatch.setattr(batch,'fingerprint',lambda *a:{})
    monkeypatch.setattr(batch,'check_inputs',lambda *a:None)
    monkeypatch.setattr(batch,'digest',lambda *a:'same')
    # Keep the final read-only inventory assertion meaningful with the mocked fingerprint.
    inventory_path = (batch.ROOT/'data/rak/inventaris.csv').resolve()
    monkeypatch.setattr(batch,'fingerprint',lambda *a:{str(inventory_path):'same'})
    calls=[]
    def launch(command, log, timeout):
        calls.append(command)
        folder=Path(command[command.index('--output')+1]); take=folder/'take_1';take.mkdir(parents=True)
        (take/'target_view.jsonl').write_text('')
        if live:
            (take/'camera_raw.mp4.times.csv').write_text('frame,t\n0,0\n')
        return 0
    monkeypatch.setattr(batch,'run_case',launch)
    monkeypatch.setattr(batch,'power_status',lambda:{'ac':1})
    order=[]
    monkeypatch.setattr(run_target,'prepare_session',lambda a:(tmp_path,tmp_path/'cfg',inventory_path))
    def calibrate(args, *rest):
        assert args.no_window == headless
        order.append('calibration')
        return 0
    monkeypatch.setattr(run_target,'run_calibration',calibrate)
    out=tmp_path/'out'
    argv=['--camera',camera,'--config',str(profile),'--output',str(out)]
    argv += ['--live','--calibrate-first'] if live else ['--source',str(source)]
    if headless: argv += ['--no-window']
    assert batch.main(argv)==0
    assert len(calls)==2 and [Path(c[c.index('--output')+1]).name for c in calls]==['K01','K03']
    assert all(c[c.index('--camera')+1]==camera for c in calls)
    assert all(('--source' in c)==(not live) and ('--no-window' in c)==(not live or headless) for c in calls)
    assert order==(['calibration'] if live else [])
    rows=json.loads((out/'results.json').read_text())
    assert all(r['owner_validation']=='PENDING' and not r['machine_verified_once'] for r in rows)
    html=(out/'REVIEW.html').read_text(encoding='utf-8')
    assert ('Live camera trials' in html)==live


def test_live_calibration_failure_stops_before_any_target(tmp_path, monkeypatch):
    from scripts import validate_box_targets as batch, run_target
    monkeypatch.setattr(batch,'inventory_plan',lambda p:[{'box_id':'K10','request':'angiolite 4 x 19'}])
    monkeypatch.setattr(batch,'fingerprint',lambda *a:{})
    monkeypatch.setattr(run_target,'prepare_session',lambda a:(tmp_path,tmp_path/'config',tmp_path/'inv'))
    monkeypatch.setattr(run_target,'run_calibration',lambda *a:1)
    monkeypatch.setattr(batch,'run_case',lambda *a:pytest.fail('detection must not start'))
    assert batch.main(['--live','--camera','webcam','--config',str(batch.ROOT/'config/box_pipeline_config_focus.yaml'),
                       '--calibrate-first','--output',str(tmp_path/'out')])==1

"""Visual lock must pause reading without authorizing physical selection."""
import datetime as dt
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from src.box_target_view import draw_target, run, select_visible, source_ready, validate_args


def test_transparent_header_halo_follows_the_same_glyph_advances():
    frame = np.full((800,1280,3),255,np.uint8)
    status = 'VERIFIED #4 - reading missing details | verified in 4.65s | boxes 11/11 expected'
    view = draw_target(frame,None,'essential pro 3 x 40',status,verifying=True)
    line=view[35:60,:,0]
    white=np.where(line>225)[1]; dark=np.where(line<80)[1]
    assert len(white)>100 and len(dark)>100
    assert dark.min() >= white.min()-2 and dark.max() <= white.max()+2
    assert np.array_equal(view[90:],frame[90:])
from src.box_verifier import CANDIDATE, EXPIRED, IGNORED, TrackState


def test_all_visible_context_boxes_remain_neutral_and_do_not_mutate_source():
    frame = np.zeros((120, 450, 3), np.uint8)
    poly = np.float32([[10, 10], [30, 10], [30, 100], [10, 100]])
    context = [(i+2, poly + [35*(i+1), 0]) for i in range(11)]
    result = draw_target(frame, poly, 'test', 'status', verified=True, verifying=True, context=context)
    assert not frame.any()
    assert tuple(result[95, 10]) == (0, 210, 0)
    for i in range(11):
        pixel = result[95, 10+35*(i+1)]
        assert pixel.min() > 0 and pixel[0] == pixel[1] == pixel[2]


def test_shelf_menu_keeps_each_dated_request_without_inventing_track_identity():
    from src.box_target_view import shelf_requests
    entries = [{'no': str(i+5), 'text': f'permanent sled bag 2029-03-{i+10}',
                'boxes': [{'id': f'K{i}', 'expiry': f'2029-03-{i+10}'}], 'generic': False} for i in range(3)]
    before = [dict(r) for r in entries]
    result = shelf_requests(entries, ['K0', 'K2'])
    assert [r['no'] for r in result] == ['1', '2']
    assert [r['text'] for r in result] == [entries[0]['text'], entries[2]['text']]
    assert entries == before
    with pytest.raises(ValueError, match='missing'):
        shelf_requests(entries, ['K99'])
    with pytest.raises(ValueError, match='Duplicate'):
        shelf_requests(entries, ['K0', 'K0'])


def supported(atoms=frozenset({"tp", "td", "tl"}), count=2):
    state = TrackState()
    for _ in range(count):
        state.add(CANDIDATE, ("target",), (atoms, False))
    return state


def test_one_repeated_complete_source_can_hold_visual_target():
    assert source_ready(supported())
    assert not source_ready(supported(count=1))
    assert not source_ready(supported(frozenset({"tp", "td"})))


@pytest.mark.parametrize("reason", ["stale", "conflict", IGNORED, EXPIRED])
def test_uncertain_or_contradicted_track_keeps_reading(reason):
    state = supported()
    if reason == "stale":
        state.mark_stale()
    elif reason == "conflict":
        state.support[-1] = (state.support[-1][0], True)
    else:
        state.history[-1] = (reason, None)
    assert not source_ready(state)


def test_dated_request_requires_actual_matching_date_before_pausing():
    state = supported()
    requested = dt.date(2029, 3, 24)
    assert not source_ready(state, required_expiry=requested)
    state.expiry = dt.date(2029, 3, 12)
    assert not source_ready(state, required_expiry=requested)
    state.expiry = requested
    assert source_ready(state, required_expiry=requested)


def test_different_gtins_in_recent_readings_do_not_freeze_target():
    state = supported(frozenset({("bc", "111")}))
    state.support[-1] = (frozenset({("bc", "222")}), False)
    assert not source_ready(state)


def test_disappeared_target_is_not_drawn_and_held_target_wins_when_visible():
    states = {1: supported(), 2: supported()}
    tracker = NS(tracks=states, state=states.__getitem__)
    lock = NS(active=True, tid=2)
    assert select_visible([], [1, 2], tracker, lock, held=1) == 1
    assert select_visible([], [], tracker, lock, held=1) is None
    assert select_visible([], [-1], tracker, lock) is None


def test_dated_siblings_prioritize_the_observed_requested_date():
    states = {1: supported(), 2: supported()}
    states[1].expiry = dt.date(2029, 3, 12)
    states[2].expiry = dt.date(2029, 3, 24)
    tracker = NS(tracks=states, state=states.__getitem__)
    lock = NS(active=True, tid=1)
    assert select_visible([], [1, 2], tracker, lock, required_expiry=states[2].expiry) == 2


def test_transparent_header_preserves_native_viewport_and_source_pixels():
    frame = np.full((180, 300, 3), 100, np.uint8)
    original = frame.copy()
    poly = np.float32([[230, 100], [260, 100], [260, 170], [230, 170]])
    vis = draw_target(frame, poly, "conqueror 3.5 x 15", "LOCKED")
    assert vis.shape == frame.shape
    assert np.array_equal(frame, original)
    assert tuple(vis[0, 0]) == (86, 86, 86)  # 82% original +18% dark panel; no opaque strip
    assert np.array_equal(vis[95:, :200], frame[95:, :200])
    assert tuple(vis[120, 230]) == (255, 220, 30)


@pytest.mark.parametrize("change", [{"request": ""}, {"qty": 2}, {"depth": True},
                                    {"rencana": "plan.txt"}, {"until_taken": True}, {"no_lock": True}])
def test_physical_or_multi_target_arguments_rejected(change):
    args = NS(request="angiolite 2.5 x 29", rencana=None, until_taken=False, qty=1, depth=False, no_lock=False)
    vars(args).update(change)
    with pytest.raises(ValueError):
        validate_args(args)


@pytest.mark.parametrize("verifying", [False, True])
def test_run_pauses_ocr_resumes_after_loss_and_has_no_pick_outputs(tmp_path, monkeypatch, verifying):
    from scripts import test_webcam_box as pipeline
    from src import box_catalog, box_detector, box_ocr, box_session, config

    cfg = config.load_config(config.resolve("config/box_pipeline_config_webcam.yaml"))
    weights = tmp_path / "fake_model.pt"
    weights.touch()
    cfg["model"]["weights"] = str(weights)
    cfg["output"] = {"screens_dir": str(tmp_path / "screens"),
                     "verify_log_csv": str(tmp_path / "verify_log.csv")}
    monkeypatch.setattr(config, "load_config", lambda _: cfg)
    monkeypatch.setattr(box_detector, "BoxDetector", lambda *a, **kw: object())
    monkeypatch.setattr(box_ocr, "SizeTextReader", lambda *a, **kw: object())
    monkeypatch.setattr(box_catalog, "Catalog", lambda *a: object())
    monkeypatch.setattr(pipeline, "OcrWorker", lambda *a, **kw: NS(req=a[2]))
    monkeypatch.setattr("sys.stdin", None)

    def forbidden(*a, **kw):
        pytest.fail("Visual mode entered the physical PICK workflow")

    monkeypatch.setattr(box_session, "SelectionSession", forbidden)
    monkeypatch.setattr(pipeline, "write_grasp", forbidden)
    enabled = []
    frame = np.zeros((120, 180, 3), np.uint8)
    poly = np.float32([[60, 5], [85, 5], [85, 115], [60, 115]])
    groups = [NS(box=NS(poly=poly))]

    def frames(args, cfg, info):
        for i in range(5):
            info["t"] = i / 30
            yield i, frame, None

    def process(frame, detector, reader, tracker, request, cfg, frame_idx, **kwargs):
        assert kwargs.get("session") is None
        enabled.append(kwargs["read_enabled"])
        if frame_idx == 0:
            tracker.tracks[1] = {"state": supported(frozenset({"tp", "td", "tl", "ref"}))}
        if frame_idx == 2:
            tracker.state(1).mark_stale()
            return [], [], [], [], 0, 0
        if frame_idx == 3:
            tracker.tracks[1] = {"state": supported(frozenset({"tp", "td", "tl", "ref"}))}
        return [], groups, [], [1], 0, 0

    monkeypatch.setattr(pipeline, "iter_source", frames)
    monkeypatch.setattr(pipeline, "process", process)
    args = NS(request="angiolite 2.5 x 29", rencana=None, until_taken=False, qty=1, depth=False,
              no_lock=False, source="test.mp4", record=None, config="unused", imgsz=None, device=None,
              weights=None, inventaris=None, expected=20, roi=None, debug_readings=None,
              no_window=True, record_vis=None, max_frames=5, verify_target=verifying,
              evidence_photos=str(tmp_path / "photos") if verifying else None)
    run(args)
    assert enabled == ([True] * 5 if verifying else [True, False, False, True, False])
    rows = [json.loads(line) for line in (tmp_path / "target_view.jsonl").read_text().splitlines()]
    ready = "VERIFIED" if verifying else "VISUAL_LOCK"
    displays = [row for row in rows if row["state"] == "DISPLAY_READY"]
    rows = [row for row in rows if row["state"] != "DISPLAY_READY"]
    assert len(displays) == (2 if verifying else 0)
    assert all(row["wall_seconds"] >= row["verified_wall_seconds"] - .001 for row in displays)
    assert [row["state"] for row in rows] == [ready, "SEARCHING", ready]
    assert [row["identity_verified"] for row in rows] == [verifying, False, verifying]
    assert all(row["verified_pick"] is False and row["time_basis"] == "video" for row in rows)
    assert all("wall_seconds" in row for row in rows)
    expected = ["fake_model.pt", "screens", "target_view.jsonl", "outline_frames.jsonl", "frame_latency.jsonl"]
    outlines = [json.loads(s) for s in (tmp_path / "outline_frames.jsonl").read_text().splitlines()]
    assert len(outlines) == 5 and outlines[2]["display"] is None
    assert outlines[3]["mode"] == "acquiring"
    if verifying:
        expected.append("photos")
        photos = [json.loads(s) for s in (tmp_path / "photos/index.jsonl").read_text().splitlines()]
        assert [p["event"] for p in photos] == ["VERIFIED", "SEARCHING", "VERIFIED", "FINAL"]
        assert [p["identity_verified"] for p in photos] == [True, False, True, True]
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(expected)


def test_process_pause_skips_crop_and_ocr_but_keeps_detection(monkeypatch):
    from scripts import test_webcam_box as pipeline
    from src.box_detector import Det
    from src.box_verifier import SimpleTracker, parse_request
    from src.config import load_config, resolve

    frame = np.zeros((150, 100, 3), np.uint8)
    poly = np.float32([[30, 10], [60, 10], [60, 130], [30, 130]])
    det = NS(detect=lambda frame: ([Det("box", .99, poly)], 10.))
    tracker = SimpleTracker()
    cfg = load_config(resolve("config/box_pipeline_config_webcam.yaml"))
    calls = []

    def prepare(*args):
        calls.append(True)
        return ()

    monkeypatch.setattr(pipeline, "prepare_crops", prepare)
    worker = NS(drain=lambda: [], submit=lambda *a: True)
    args = (frame, det, None, tracker, parse_request("angiolite 2.5 x 29"), cfg)
    result = pipeline.process(*args, 0, worker=worker, read_enabled=False)
    assert len(result[1]) == 1 and result[3][0] > 0
    assert calls == []
    pipeline.process(*args, 1, worker=worker, read_enabled=True)
    assert calls == [True]


def test_pending_blue_target_refines_without_identity_or_source_geometry_changes(tmp_path, monkeypatch):
    from scripts import test_webcam_box as pipeline
    from src import box_catalog, box_detector, box_ocr, box_outline, config
    cfg = config.load_config(config.resolve('config/box_pipeline_config_webcam.yaml'))
    weights = tmp_path/'fake.pt';weights.touch()
    cfg['model']['weights'] = str(weights)
    cfg['output'] = {'screens_dir':str(tmp_path/'screens'),'verify_log_csv':str(tmp_path/'verify.csv')}
    monkeypatch.setattr(config,'load_config',lambda _:cfg)
    monkeypatch.setattr(box_detector,'BoxDetector',lambda *a,**kw:object())
    monkeypatch.setattr(box_ocr,'SizeTextReader',lambda *a,**kw:object())
    monkeypatch.setattr(box_catalog,'Catalog',lambda *a:object())
    monkeypatch.setattr(pipeline,'OcrWorker',lambda *a,**kw:NS(req=a[2]))
    monkeypatch.setattr('sys.stdin',None)
    frame = np.zeros((150,180,3),np.uint8)
    raw = np.float32([[60,8],[85,8],[85,125],[60,125]])
    pixels,original = frame.copy(),raw.copy()
    corrected = np.float32([[61,12],[84,12],[84,120],[61,120]])
    calls=[]
    def refine(image,poly,neighbours):
        calls.append(True)
        np.testing.assert_array_equal(poly,original)
        return corrected.copy()
    monkeypatch.setattr(box_outline,'refine_box_outline',refine)
    state = supported()  # repeated text remains a single source
    history = list(state.history)
    def frames(args,cfg,info):
        for i in range(3):
            info['t']=i/30
            yield i,frame,None
    def process(image,detector,reader,tracker,request,cfg,frame_idx,**kw):
        assert kw['read_enabled']
        tracker.tracks[1]={'state':state}
        return [],[NS(box=NS(poly=raw))],[],[1],0,0
    monkeypatch.setattr(pipeline,'iter_source',frames)
    monkeypatch.setattr(pipeline,'process',process)
    args = NS(request='angiolite 2.5 x 29',rencana=None,until_taken=False,qty=1,depth=False,
              no_lock=False,source='test.mp4',record=None,config='unused',imgsz=None,device=None,
              weights=None,inventaris=None,expected=1,roi=None,debug_readings=None,no_window=True,
              record_vis=None,max_frames=3,verify_target=True)
    run(args)
    assert calls and list(state.history)==history and not state.combined_match(2)
    rows=[json.loads(s) for s in (tmp_path/'target_view.jsonl').read_text().splitlines()]
    assert all(not r.get('identity_verified',False) for r in rows)
    assert any(r['state']=='CHECKING' for r in rows)
    outlines=[json.loads(s) for s in (tmp_path/'outline_frames.jsonl').read_text().splitlines()]
    for row in outlines:
        np.testing.assert_array_equal(row['raw'],original)
        assert np.asarray(row['display'])[:,1].min() >= 12
        assert np.asarray(row['display'])[:,1].max() <= 120
    np.testing.assert_array_equal(raw,original)
    np.testing.assert_array_equal(frame,pixels)


def test_read_again_resets_request_timer_but_video_clock_keeps_advancing(tmp_path, monkeypatch):
    import itertools
    import cv2
    from scripts import test_webcam_box as pipeline
    from src import box_catalog, box_detector, box_ocr, box_view_recorder, config
    cfg = config.load_config(config.resolve("config/box_pipeline_config_webcam.yaml"))
    weights = tmp_path / "fake.pt"
    weights.touch()
    cfg["model"]["weights"] = str(weights)
    cfg["output"] = {"screens_dir": str(tmp_path / "screens"), "verify_log_csv": str(tmp_path / "verify.csv")}
    monkeypatch.setattr(config, "load_config", lambda _: cfg)
    monkeypatch.setattr(box_detector, "BoxDetector", lambda *a, **kw: object())
    monkeypatch.setattr(box_ocr, "SizeTextReader", lambda *a, **kw: object())
    monkeypatch.setattr(box_catalog, "Catalog", lambda *a: object())
    monkeypatch.setattr(pipeline, "OcrWorker", lambda *a, **kw: NS(req=a[2]))
    monkeypatch.setattr("sys.stdin", None)
    ticks = itertools.count(1, .01)
    monkeypatch.setattr("time.perf_counter", lambda: next(ticks))
    keys = iter((ord("o"), 0))
    monkeypatch.setattr(cv2, "waitKey", lambda _: next(keys))
    monkeypatch.setattr(cv2, "imshow", lambda *a: None)
    recorded = []
    class Recorder:
        def __init__(self, *a, **kw): pass
        def write(self, frame, seconds): recorded.append(seconds)
        def close(self): pass
    monkeypatch.setattr(box_view_recorder, "ViewRecorder", Recorder)
    frame = np.zeros((120, 180, 3), np.uint8)
    poly = np.float32([[60, 5], [85, 5], [85, 115], [60, 115]])
    def frames(args, cfg, info):
        for i in range(3):
            info["t"] = i / 30
            yield i, frame, None
    def process(frame, detector, reader, tracker, request, cfg, frame_idx, **kw):
        if 1 not in tracker.tracks:
            tracker.tracks[1] = {"state": supported()}
        return [], [NS(box=NS(poly=poly))], [], [1], 0, 0
    monkeypatch.setattr(pipeline, "iter_source", frames)
    monkeypatch.setattr(pipeline, "process", process)
    args = NS(request="angiolite 2.5 x 29", rencana=None, until_taken=False, qty=1, depth=False,
              no_lock=False, source="test.mp4", record=None, config="unused", imgsz=None, device=None,
              weights=None, inventaris=None, expected=20, roi=None, debug_readings=None,
              no_window=False, view_width=1280, record_vis=str(tmp_path / "view.mp4"), record_vis_fps=6,
              max_frames=3, verify_target=False, evidence_photos=None)
    run(args)
    assert len(recorded) == 3 and recorded[0] < recorded[1] < recorded[2]
    rows = [json.loads(s) for s in (tmp_path / "frame_latency.jsonl").read_text().splitlines()]
    assert [r["request_generation"] for r in rows] == [0, 1, 1]


def test_verified_identity_survives_later_partial_readings_but_not_evidence_against_it():
    """20-10, Accuforce 3.5 x 15: VERIFIED with text + REF, then text-only detail reads pushed REF out of the combined
    window and the target went back to CHECKING 3 times. Only evidence against it may release it."""
    from src.box_target_view import keeps_verified
    state = TrackState()
    state.add(CANDIDATE, ("t",), (frozenset({"tp", "td", "tl", "ref"}), False))
    for _ in range(5):                                    # later readings: text only, no contradiction
        state.add(CANDIDATE, ("t",), (frozenset({"tp", "td", "tl"}), False))
    assert keeps_verified(state)
    contradicted = TrackState()
    contradicted.add(CANDIDATE, ("t",), (frozenset({"tp", "td", "tl", "ref"}), False))
    contradicted.add(CANDIDATE, ("t",), (frozenset({"tp"}), True))      # another size / REF read
    assert not keeps_verified(contradicted)
    other = TrackState()
    other.add(IGNORED, ("t",), (frozenset(), False))                    # read as not the target
    assert not keeps_verified(other)
    stale = TrackState()
    stale.add(CANDIDATE, ("t",), (frozenset({"tp", "td", "tl"}), False))
    stale.stale = True
    assert not keeps_verified(stale)
    dated = TrackState()
    dated.add(CANDIDATE, ("t",), (frozenset({"tp", "td", "tl"}), False))
    assert not keeps_verified(dated, required_expiry=dt.date(2029, 3, 10))   # required printed date not held


def test_supports_request_needs_matching_evidence_and_no_contradiction():
    from src.box_target_view import supports_request
    from src.box_verifier import Evidence
    assert not supports_request(TrackState())                                   # colour lock only: no reading yet
    assert supports_request(supported())
    assert not supports_request(supported(frozenset({"tp"})))                   # product only (another size of the product)
    assert supports_request(supported(frozenset({"ref"})))                      # REF alone is one source
    contradicted = supported()
    contradicted.support[-1] = (contradicted.support[-1][0], True)
    assert not supports_request(contradicted)
    for decision in (IGNORED, EXPIRED):
        ignored = supported()
        ignored.history[-1] = (decision, None)
        assert not supports_request(ignored)
    stale = supported()
    stale.mark_stale()
    assert not supports_request(stale)
    wanted, other = dt.date(2029, 3, 24), dt.date(2029, 3, 10)
    bag = TrackState()
    for count in (1, 2):
        bag.add(CANDIDATE, ("bag",), (frozenset({"tp", "td", "tl"}), False), Evidence(dates=(other,)))
        assert supports_request(bag, required_expiry=wanted) is (count == 1)    # one other date is not enough
    bag.add(CANDIDATE, ("bag",), (frozenset({"tp", "td", "tl"}), False), Evidence(dates=(wanted,)))
    assert supports_request(bag, required_expiry=wanted)                        # the requested date was read again


def test_tentative_outline_is_neutral_grey_not_the_blue_candidate_colour():
    frame = np.zeros((120, 90, 3), np.uint8)
    poly = np.float32([[10, 10], [60, 10], [60, 100], [10, 100]])
    blue = draw_target(frame, poly, 'test', 'status')
    grey = draw_target(frame, poly, 'test', 'status', tentative=True)
    verified = draw_target(frame, poly, 'test', 'status', verified=True, verifying=True, tentative=True)
    assert tuple(blue[95, 10]) == (255, 220, 30)
    assert tuple(grey[95, 10]) == (190, 190, 190)
    assert tuple(verified[95 + 0, 10]) == (0, 210, 0)                          # a verified box is never tentative


@pytest.mark.parametrize("other_date_twice", [True, False])
def test_pending_target_is_logged_as_tentative_when_its_printed_date_differs_twice(tmp_path, monkeypatch, other_date_twice):
    from scripts import test_webcam_box as pipeline
    from src import box_catalog, box_detector, box_ocr, config
    from src.box_verifier import Evidence
    monkeypatch.setattr('src.box_verifier.REQUIRED_EXPIRY', None)   # run() sets this module global for a dated request
    cfg = config.load_config(config.resolve('config/box_pipeline_config_webcam.yaml'))
    weights = tmp_path/'fake.pt';weights.touch()
    cfg['model']['weights'] = str(weights)
    cfg['output'] = {'screens_dir':str(tmp_path/'screens'),'verify_log_csv':str(tmp_path/'verify.csv')}
    monkeypatch.setattr(config,'load_config',lambda _:cfg)
    monkeypatch.setattr(box_detector,'BoxDetector',lambda *a,**kw:object())
    monkeypatch.setattr(box_ocr,'SizeTextReader',lambda *a,**kw:object())
    monkeypatch.setattr(box_catalog,'Catalog',lambda *a:object())
    monkeypatch.setattr(pipeline,'OcrWorker',lambda *a,**kw:NS(req=a[2]))
    monkeypatch.setattr('sys.stdin',None)
    frame = np.zeros((150,180,3),np.uint8)
    raw = np.float32([[60,8],[85,8],[85,125],[60,125]])
    state = TrackState()
    for date in ((dt.date(2029,3,10),)*2 if other_date_twice else (dt.date(2029,3,10), dt.date(2029,3,24))):
        state.add(CANDIDATE, ("bag",), (frozenset({"tp","td","tl"}), False), Evidence(dates=(date,)))
    def frames(args,cfg,info):
        for i in range(3):
            info['t']=i/30
            yield i,frame,None
    def process(image,detector,reader,tracker,request,cfg,frame_idx,**kw):
        tracker.tracks[1]={'state':state}
        return [],[NS(box=NS(poly=raw))],[],[1],0,0
    monkeypatch.setattr(pipeline,'iter_source',frames)
    monkeypatch.setattr(pipeline,'process',process)
    args = NS(request='permanent sled bag 2029-03-24',rencana=None,until_taken=False,qty=1,depth=False,
              no_lock=False,source='test.mp4',record=None,config='unused',imgsz=None,device=None,
              weights=None,inventaris=None,expected=1,roi=None,debug_readings=None,no_window=True,
              record_vis=None,max_frames=3,verify_target=True)
    run(args)
    outlines=[json.loads(s) for s in (tmp_path/'outline_frames.jsonl').read_text().splitlines()]
    assert outlines and all(row['evidence_supported'] is (not other_date_twice) for row in outlines)
    rows=[json.loads(s) for s in (tmp_path/'target_view.jsonl').read_text().splitlines()]
    assert all(not r.get('identity_verified', False) for r in rows)            # display only: never a verification
    assert rows[0]['evidence_supported'] is (not other_date_twice)

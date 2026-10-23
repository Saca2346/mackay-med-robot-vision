"""pytest tests/test_target_parallel_reads.py -q  (the locked target may have one OCR read per worker in flight)"""
import importlib.util
from types import SimpleNamespace as NS
import numpy as np

from src.box_verifier import CANDIDATE, Evidence, TrackState, Request
import pytest


def load():
    spec = importlib.util.spec_from_file_location("test_webcam_box", "scripts/test_webcam_box.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_each_finished_read_frees_one_slot_and_pending_stays_until_the_last():
    m = load()
    st = TrackState()
    st.inflight, st.pending = 2, True
    m.apply_reading(st, Evidence(), CANDIDATE, "", ("k",), 10.0, 1)
    assert st.inflight == 1 and st.pending is True
    m.apply_reading(st, Evidence(), CANDIDATE, "", ("k",), 10.0, 1)
    assert st.inflight == 0 and st.pending is False


def test_a_synchronous_read_never_goes_negative():
    m = load()
    st = TrackState()
    m.apply_reading(st, Evidence(), CANDIDATE, "", ("k",), 10.0, 1)
    assert st.inflight == 0 and st.pending is False


def test_worker_knows_how_many_readers_it_has():
    m = load()
    # the real worker starts daemon threads that wait on the queue; only the attribute is checked
    w = m.OcrWorker([object(), object()], {}, None, None, read_fn=lambda *a, **k: None)
    assert w.n_readers == 2


def test_queued_reads_of_other_boxes_are_dropped_once_the_target_is_chosen():
    m = load()
    w = m.OcrWorker([], {}, None, None, max_pending=5, read_fn=lambda *a, **k: None)   # no threads: jobs stay queued
    target, other1, other2 = TrackState(), TrackState(), TrackState()   # equal by value: identity must decide
    for st in (other1, target, other2):
        assert w.submit(st, ("crop",))
        st.inflight, st.pending = 1, True
    assert w.keep_only([target]) == 2
    assert [job[0] is target for job in list(w.jobs.queue)] == [True]
    assert (other1.inflight, other1.pending, other2.inflight) == (0, False, 0)
    assert (target.inflight, target.pending) == (1, True)


def test_early_colour_refresh_responds_before_frame_ten_without_identity_evidence():
    m=load();st=TrackState();tracker=NS(state=lambda t:st)
    poly=np.float32([[10,10],[50,10],[50,190],[10,190]])
    groups=[NS(box=NS(poly=poly))];frame=np.full((200,80,3),100,np.uint8)
    similarity=[.66]
    compare=lambda a,b:similarity[0]
    m.update_hints(frame,groups,[1],tracker,[('target',np.zeros(1))],0,compare=compare,early=True)
    assert st.hint==.66 and st.hint_frame==0
    similarity[0]=.96
    m.update_hints(frame,groups,[1],tracker,[('target',np.zeros(1))],1,compare=compare,early=True)
    assert st.hint==.66
    m.update_hints(frame,groups,[1],tracker,[('target',np.zeros(1))],2,compare=compare,early=True)
    assert st.hint==.96 and st.hint_frame==2
    assert not st.history and not st.support and not st.evs and st.reads==0


def test_normal_colour_refresh_keeps_existing_interval_and_skips_invalid_tracks():
    m=load();st=TrackState(hint=.66,hint_frame=0);tracker=NS(state=lambda t:st)
    poly=np.float32([[10,10],[50,10],[50,190],[10,190]])
    group=NS(box=NS(poly=poly));frame=np.full((200,80,3),100,np.uint8)
    compare=lambda a,b:.96
    m.update_hints(frame,[group,group],[-1,1],tracker,[('target',np.zeros(1))],9,compare=compare)
    assert st.hint==.66
    m.update_hints(frame,[group],[1],tracker,[('target',np.zeros(1))],10,compare=compare)
    assert st.hint==.96 and st.hint_frame==10


@pytest.mark.parametrize('sized,frame_idx,enabled,focused,expected',[
    (True,2,True,True,True),(False,2,True,True,False),
    (True,15,True,True,False),(True,2,False,True,False),(True,2,True,False,False)])
def test_only_focused_sized_requests_refresh_early(monkeypatch,sized,frame_idx,enabled,focused,expected):
    m=load();st=TrackState();poly=np.float32([[10,10],[50,10],[50,190],[10,190]])
    group=NS(box=NS(poly=poly));det=NS(detect=lambda f:([],0))
    tracker=NS(tracks={1:{'state':st}},state=lambda t:st,update=lambda *a:[1])
    monkeypatch.setattr(m,'group_labels',lambda *a:([group],[]))
    calls=[];monkeypatch.setattr(m,'update_hints',lambda *a,**kw:calls.append(kw['early']))
    req=Request('angiolite',2.5,29) if sized else Request('permanent sled bag')
    cfg={'grouping':{},'ocr':{},'visual':{'early_hint_refresh':enabled}}
    m.process(np.zeros((200,80,3),np.uint8),det,None,tracker,req,cfg,frame_idx,
              targets=[('target',np.zeros(1))],read_enabled=False,
              read_order=(lambda *a:None) if focused else None)
    assert calls==[expected] and not st.support

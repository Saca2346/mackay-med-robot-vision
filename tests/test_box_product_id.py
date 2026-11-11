"""Product look model, role A: it orders reading and can hold a green back, never make one."""
import numpy as np

from src.box_product_id import ProductLook, classes_for
from src.box_verifier import CONFIRM, IGNORED, MATCH, TrackState

CLASSES = ["APTMedical_ConquerorII", "APTMedical_ConquerorNCPro", "BostonScientific_Agent", "BostonScientific_NCEmerge",
           "OrbusNeich_Sapphire3", "OrbusNeich_SapphireNC24", "Terumo_Accuforce"]


def _look(product, veto_reads=3):
    look = ProductLook.__new__(ProductLook)          # no model file: probabilities are given by the test
    look.classes, look.veto_top, look.veto_req, look.veto_reads, look.order_min = CLASSES, 0.85, 0.02, veto_reads, 0.5
    look.last_ms, look._busy, look._done, look.error, look.every, look.last_submit = 0.0, False, None, "", 5, -1000
    look.set_request(product)
    return look


def _probs(**p):
    v = np.full(len(CLASSES), 1e-4)
    for name, x in p.items():
        v[CLASSES.index(name)] = x
    return v / v.sum()


def _feed(look, states, rows):
    look.probs = lambda frame, polys: np.stack(rows)
    look.update(states, None, [None] * len(rows))


def _match_state():
    st = TrackState()
    st.history.extend([(MATCH, ("k",)), (MATCH, ("k",))])
    return st


def test_catalogue_names_map_to_model_classes():
    assert classes_for("conqueror nc pro", CLASSES) == [1]               # not Conqueror II
    assert classes_for("sapphire", CLASSES) == [4, 5]                    # both Sapphire looks count for "sapphire"
    assert classes_for("nc emerge", CLASSES) == [3]
    assert classes_for("permanent sled bag", CLASSES) == []              # unknown product


def test_unknown_product_switches_the_model_off():
    look = _look("permanent sled bag")
    st = TrackState()
    st.hint = 0.7
    _feed(look, [st], [_probs(Terumo_Accuforce=0.99)])                   # update() returns before using the model
    assert not look.active and st.hint == 0.7 and st.look_p == -1.0 and not st.look_veto


def test_order_hint_up_for_the_product_down_for_another():
    look = _look("accuforce")
    same, other = TrackState(), TrackState()
    same.hint = other.hint = 0.6
    _feed(look, [same, other], [_probs(Terumo_Accuforce=0.9), _probs(OrbusNeich_Sapphire3=0.95)])
    assert same.hint >= 0.9 and other.hint == 0.0


def test_unsure_model_changes_nothing():
    look = _look("conqueror nc pro")
    st = TrackState()
    st.hint = 0.7
    for _ in range(5):                                                    # sibling confusion below veto_top: no veto
        _feed(look, [st], [_probs(APTMedical_ConquerorII=0.6, APTMedical_ConquerorNCPro=0.3)])
    assert st.hint == 0.7 and not st.look_veto


def test_veto_needs_reads_in_a_row_and_holds_match_as_confirm():
    look = _look("agent")
    st = _match_state()
    sure_other = _probs(BostonScientific_NCEmerge=0.97)
    _feed(look, [st], [sure_other])
    _feed(look, [st], [sure_other])
    assert st.stable(2) == MATCH                                          # two reads: not yet
    _feed(look, [st], [_probs(BostonScientific_Agent=0.5, BostonScientific_NCEmerge=0.5)])
    _feed(look, [st], [sure_other])
    _feed(look, [st], [sure_other])
    assert st.stable(2) == MATCH                                          # the run was broken: counting starts again
    _feed(look, [st], [sure_other])
    assert st.look_veto and st.stable(2) == CONFIRM                       # three in a row: held, not picked
    _feed(look, [st], [_probs(BostonScientific_Agent=0.9)])
    assert not st.look_veto and st.stable(2) == MATCH                     # the model changed its mind: released


def test_veto_never_creates_a_match_and_is_cleared_with_a_doubted_track():
    look = _look("agent", veto_reads=1)
    st = TrackState()
    st.history.extend([(IGNORED, ("x",)), (IGNORED, ("x",))])
    _feed(look, [st], [_probs(BostonScientific_Agent=0.99)])              # model sure it IS the product
    assert st.stable(2) == IGNORED                                        # reading still decides
    _feed(look, [st], [_probs(BostonScientific_NCEmerge=0.99)])
    assert st.look_veto
    st.mark_stale()
    assert not st.look_veto and st.look_other == 0


def test_request_change_resets_the_look():
    st = TrackState()
    st.history.extend([(MATCH, ("k",)), (MATCH, ("k",))])
    st.look_p, st.look_veto, st.look_other = 0.01, "look: x", 3
    st.redecide(lambda ev: (MATCH, "", ("k",), (frozenset(), False)))
    assert st.look_p == -1.0 and not st.look_veto and st.look_other == 0


def test_background_look_is_applied_only_to_unchanged_boxes():
    import time
    look = _look("accuforce", veto_reads=1)

    def slow(frame, polys):
        time.sleep(0.2)
        return np.stack([_probs(OrbusNeich_Sapphire3=0.97)] * len(polys))
    look.probs = slow
    kept, doubted = TrackState(), TrackState()
    kept.hint = doubted.hint = 0.7
    frame = np.zeros((4, 4, 3), np.uint8)
    assert look.submit([kept, doubted], frame, [np.zeros((4, 2))] * 2)
    assert not look.submit([kept], frame, [np.zeros((4, 2))])            # busy: skipped, not queued
    for _ in range(200):
        if look._done is not None:
            break
        time.sleep(0.01)
    doubted.history.append((MATCH, ("k",)))
    doubted.mark_stale()                                                   # its number was doubted meanwhile
    look.drain()
    assert kept.look_veto and kept.hint == 0.0
    assert not doubted.look_veto and doubted.hint == 0.7 and doubted.look_p == -1.0


def test_colour_refresh_keeps_the_look_order():
    look = _look("accuforce")
    st = TrackState()
    _feed(look, [st], [_probs(OrbusNeich_Sapphire3=0.97)])
    st.hint = 0.9                                                          # colour hint refreshed afterwards
    look.adjust(st)
    assert st.hint == 0.0


def test_a_failing_model_never_raises_into_the_live_loop():
    import time
    look = _look("accuforce")

    def boom(frame, polys):
        raise RuntimeError("device lost")
    look.probs = boom
    st = TrackState()
    assert look.submit([st], np.zeros((4, 4, 3), np.uint8), [np.zeros((4, 2))])
    for _ in range(200):
        if not look._busy:
            break
        time.sleep(0.01)
    look.drain()
    assert "device lost" in look.error and st.look_p == -1.0

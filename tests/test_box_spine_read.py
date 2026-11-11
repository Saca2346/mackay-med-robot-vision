"""Separate printed bands preserve independent sources and numeric unit geometry."""
import numpy as np
import pytest

from src.box_ocr import OcrResult, OcrToken
from src.box_spine_read import column_tokens, read_spine_bands, read_label_regions, size_field_tokens
from src.box_verifier import MATCH, combined_score, parse_request
from src.config import load_config, resolve


def token(text, x, y, conf=.99):
    half = 3 if text == "mm" else 10
    return OcrToken(text, conf, np.float32([[x-5, y-half], [x+5, y-half], [x+5, y+half], [x-5, y+half]]))


def test_columns_keep_each_printed_unit_with_its_number():
    tokens = [token("4", 20, 10), token("19", 100, 10), token("mm", 20, 30), token("mm", 100, 30)]
    assert [t.text for t in column_tokens(tokens, 50)] == ["4", "mm", "19", "mm"]
    missing = [OcrToken("4", .99), OcrToken("19", .99)]
    assert column_tokens(missing, 50) is missing


@pytest.mark.parametrize('ref,retry_value,retry_conf,original_length,expected', [
    ('SCCDSR14150250029', '2.5', .99, '29', True),
    ('', '2.5', .99, '29', False),
    ('SCCDSR14150250029', '2.5', .89, '29', False),
    ('SCCDSR14150250029', '4', .99, '29', False),
    ('SCCDSR14150250029', '2.5', .99, '19', False),
])
@pytest.mark.parametrize('enabled', [False, True])
def test_uncertain_size_retry_keeps_confidence_conflicts_and_one_text_source(
        ref, retry_value, retry_conf, original_length, expected, enabled):
    cfg = load_config(resolve('config/box_pipeline_config_realsense_eleven.yaml'))
    cfg['visual']['size_token_retry'] = enabled
    class Reader:
        calls = 0
        images = []
        def read_once(self, image):
            self.calls += 1
            self.images.append(image.copy())
            if self.calls == 1:
                values = [token('angiolite', 40, 15), token('2.5', 250, 15, .87),
                          token(original_length, 330, 15)]
                if ref: values.append(token(ref, 150, 15))
            else:
                values = [token(retry_value, 12, 15, retry_conf)]
            return OcrResult(values)
    reader = Reader()
    box = np.arange(50*700*3,dtype=np.uint8).reshape(50,700,3)
    (ev, decision, *_), bands = read_spine_bands(box, reader, cfg, parse_request('angiolite 2.5 x 29'))
    assert reader.calls == (2 if enabled else 1)
    assert (decision == MATCH) is (expected and enabled)
    assert len(bands) == (2 if enabled else 1)
    if enabled:
        joint = box[:,round(700*.16):round(700*.73)]
        assert np.array_equal(reader.images[1][4:-4,4:-4], joint[3:28,242:259])
        assert np.all(reader.images[1][:4] == 255)
    if not ref: assert combined_score(ev.support[0])[0] <= 1


@pytest.mark.parametrize('ref,diameter,length,confidence,enabled,expected',[
    ('SCCDSR14150250029','2.5','29',.99,True,True),
    ('','2.5','29',.99,True,False),
    ('SCCDSR14150250029','4','19',.99,True,False),
    ('SCCDSR14150250029','2.5','29',.50,True,False),
    ('SCCDSR14150250029','2.5','29',.99,False,False)])
def test_merged_numeric_retry_reads_real_cells_without_splitting_values_or_sources(monkeypatch,ref,diameter,length,confidence,enabled,expected):
    import src.box_barcode as barcode
    monkeypatch.setattr(barcode,'decode_gs1',lambda *a,**k:[])
    cfg=load_config(resolve('config/box_pipeline_config_realsense_eleven.yaml'))
    cfg['visual']['size_column_retry']=enabled
    class Reader:
        calls=0
        images=[]
        def read_once(self,image):
            self.calls+=1;self.images.append(image.copy())
            if self.calls==1:
                values=[token('angiolite',40,15),token('2.529',280,15)]
                if ref:values.append(token(ref,150,15))
            else:
                values=[token(diameter if self.calls==2 else length,30,15,confidence)]
            return OcrResult(values)
    reader=Reader();box=np.arange(50*700*3,dtype=np.uint8).reshape(50,700,3)
    result,bands=read_spine_bands(box,reader,cfg,parse_request('angiolite 2.5 x 29'))
    ev,decision,*_=result
    assert (decision==MATCH) is expected
    assert (combined_score(ev.support[0])[0]>=2 and not ev.support[1]) is expected
    assert reader.calls==(3 if enabled else 1)
    if enabled:
        size=box[:,round(700*.48):round(700*.73)]
        assert np.array_equal(reader.images[1],size[:,:size.shape[1]//2])
        assert np.array_equal(reader.images[2],size[:,size.shape[1]//2:])
        assert [b['name'] for b in bands[-2:]]==['size_retry_left','size_retry_right']
    if not ref:assert combined_score(ev.support[0])[0]<=1


@pytest.mark.parametrize("ref,diameter,length,confidence,expected", [
    ("SCCDSR14150400019", "4", "19", .99, True),
    ("SCCDSR14150250029", "2.5", "29", .99, True),
    ("SCCDSR14150250029", "4", "19", .99, False),
    ("SCCDSR14150250029", "2.5", "29", .5, False),
])
def test_band_identity_uses_read_numbers_not_ref_dimensions(monkeypatch, ref, diameter, length, confidence, expected):
    import src.box_barcode as barcode
    monkeypatch.setattr(barcode, "decode_gs1", lambda *a, **k: [])
    cfg = load_config(resolve("config/box_pipeline_config_focus.yaml"))
    class Reader:
        calls = 0
        def read_once(self, image):
            self.calls += 1
            values = ([OcrToken("angiolite", .99), OcrToken(ref, .99)] if self.calls == 1 else
                      [token(diameter, 20, 10, confidence), token(length, 100, 10), token("mm", 20, 30), token("mm", 100, 30)])
            return OcrResult(values)
    result, bands = read_spine_bands(np.zeros((50, 700, 3), np.uint8), Reader(), cfg,
                                    parse_request("angiolite 2.5 x 29" if "250" in ref else "angiolite 4 x 19"))
    ev, decision, *_ = result
    assert (decision == MATCH) is expected
    assert (combined_score(ev.support[0])[0] >= 2 and not ev.support[1]) is expected
    assert len(bands) == 2


def test_product_is_not_filled_from_ref_or_request(monkeypatch):
    import src.box_barcode as barcode
    monkeypatch.setattr(barcode, "decode_gs1", lambda *a, **k: [])
    cfg = load_config(resolve("config/box_pipeline_config_focus.yaml"))
    class Reader:
        calls = 0
        def read_once(self, image):
            self.calls += 1
            return OcrResult([OcrToken("SCCDSR14150400019", .99)] if self.calls == 1 else
                             [token("4", 20, 10), token("19", 100, 10), token("mm", 20, 30)])
    assert read_spine_bands(np.zeros((50, 700, 3), np.uint8), Reader(), cfg, parse_request("angiolite 4 x 19")) is None


def test_tiny_date_fragment_does_not_replace_large_size_numerals():
    values = [token("4", 20, 10), token("19", 100, 10), token("mm", 20, 30), token("mm", 100, 30),
              OcrToken("22", .99, np.float32([[150, 2], [165, 2], [165, 8], [150, 8]]))]
    selected = size_field_tokens(values, 60)
    assert [t.text for t in column_tokens(selected, 60)] == ["4", "mm", "19", "mm"]


def test_observed_unit_columns_exclude_tall_unpaired_date_digits():
    values = [token("4", 20, 15), token("19", 100, 15),
              token("25", 165, 10), token("20", 165, 30),
              token("mm", 20, 35), token("mm", 100, 35)]
    selected = size_field_tokens(values, 60)
    assert [t.text for t in column_tokens(selected, 60)] == ["4", "mm", "19", "mm"]
    # Without the two observed units this helper must not guess which digits
    # belong to the requested size.
    assert len(size_field_tokens(values[:4], 60)) == 4


def test_real_third_unit_column_is_not_hidden_to_fit_a_requested_size():
    values = [token("4", 20, 15), token("19", 100, 15), token("25", 180, 15),
              token("mm", 20, 35), token("mm", 100, 35), token("mm", 180, 35)]
    assert len(size_field_tokens(values, 60)) == 6


def test_name_and_size_without_current_ref_remain_one_source(monkeypatch):
    import src.box_barcode as barcode
    monkeypatch.setattr(barcode, "decode_gs1", lambda *a, **k: [])
    cfg = load_config(resolve("config/box_pipeline_config_focus.yaml"))
    class Reader:
        calls = 0
        def read_once(self, image):
            self.calls += 1
            return OcrResult([OcrToken("angiolite", .99)] if self.calls == 1 else
                             [token("4", 20, 10), token("19", 100, 10), token("mm", 20, 30), token("mm", 100, 30)])
    result, _ = read_spine_bands(np.zeros((50, 700, 3), np.uint8), Reader(), cfg, parse_request("angiolite 4 x 19"))
    ev, decision, *_ = result
    assert decision != MATCH and not ev.refs
    assert combined_score(ev.support[0])[0] == 1


@pytest.mark.parametrize("name,ref,confidence,expected", [
    ("angiolite", "SCCDSR14150400019", .99, True),
    ("angiolite", "SCCDSR14150250029", .99, False),
    ("angiolite", "SCCDSR14150400019", .5, False),
    ("unreadable", "SCCDSR14150400019", .99, False),
])
def test_label_geometry_requires_actual_name_numbers_and_independent_ref(monkeypatch, name, ref, confidence, expected):
    import src.box_barcode as barcode
    monkeypatch.setattr(barcode, "decode_gs1", lambda *a, **k: [])
    cfg = load_config(resolve("config/box_pipeline_config_focus.yaml"))
    class Reader:
        def read_once(self, image):
            return OcrResult([
                token("4", 20, 10, confidence), token("19", 100, 10),
                token("mm", 20, 30), token("mm", 100, 30),
                OcrToken("22", .99, np.float32([[150, 2], [165, 2], [165, 8], [150, 8]])),
                token(name, 200, 10), token(ref, 250, 10),
            ])
    result, bands = read_label_regions([np.zeros((50, 350, 3), np.uint8)], None,
                                      Reader(), cfg, parse_request("angiolite 4 x 19"))
    ev, decision, *_ = result
    assert (decision == MATCH) is expected
    assert ev.text.diameter == 4 and ev.text.length == 19
    assert ev.text.product_ok is (name == "angiolite")
    assert [t.text for t in bands[0]["field_tokens"]] == ["4", "19", "mm", "mm"]


@pytest.mark.parametrize("name,ref,confidence,expected", [
    ("angiolite", "SCCDSR14150400019", .99, True),
    ("angiolite", "SCCDSR14150250029", .99, False),
    ("angiolite", "SCCDSR14150400019", .5, False),
    ("unreadable", "SCCDSR14150400019", .99, False),
    ("angiolite", "", .99, False),
])
def test_joint_regions_preserve_independent_printed_sources(monkeypatch, name, ref, confidence, expected):
    import src.box_barcode as barcode
    monkeypatch.setattr(barcode, "decode_gs1", lambda *a, **k: [])
    cfg = load_config(resolve("config/box_pipeline_config_focus.yaml"))
    cfg.setdefault("visual", {})["joint_spine_read"] = True
    calls = []
    class Reader:
        def read_once(self, image):
            calls.append(True)
            return OcrResult([token(name, 40, 10), token(ref, 100, 30),
                              token("4", 240, 10, confidence), token("19", 310, 10),
                              token("mm", 240, 30), token("mm", 310, 30)])
    result = read_spine_bands(np.zeros((50, 700, 3), np.uint8), Reader(), cfg,
                              parse_request("angiolite 4 x 19"))
    assert calls == [True]
    if result is None:
        assert not expected
    else:
        (ev, decision, *_), bands = result
        assert (decision == MATCH) is expected
        assert (combined_score(ev.support[0])[0] >= 2 and not ev.support[1]) is expected
        assert len(bands) == 1


@pytest.mark.parametrize("name", ["angiolite", "unreadable"])
def test_partial_ref_returns_observed_source_but_cannot_verify_by_itself(monkeypatch, name):
    from src.box_target_view import verified_ready
    from src.box_verifier import TrackState
    import src.box_barcode as barcode
    monkeypatch.setattr(barcode, "decode_gs1", lambda *a, **k: [])
    cfg = load_config(resolve("config/box_pipeline_config_focus.yaml"))
    cfg.setdefault("visual", {}).update(joint_spine_read=True, partial_spine_read=True)
    class Reader:
        def read_once(self, image):
            return OcrResult([token(name, 40, 10), token("SCCDSR14150400019", 100, 30)])
    req = parse_request("angiolite 4 x 19")
    (ev, decision, reason, key, *_), _ = read_spine_bands(np.zeros((50, 700, 3), np.uint8), Reader(), cfg, req)
    assert ev.text.diameter is None and ev.text.length is None
    assert combined_score(ev.support[0])[0] == 1
    state = TrackState()
    for _ in range(2):
        state.add(decision, key, ev.support, ev)
    assert not verified_ready(state)


@pytest.mark.parametrize("product,ref,diameter,length", [
    ("essential pro", "BCDPR14N150300040", "3", "40"),
    ("xperience pro", "BCPR14N150250015", "2.5", "15"),
])
def test_other_supported_spines_require_their_actual_name_size_and_ref(monkeypatch, product, ref, diameter, length):
    import src.box_barcode as barcode
    monkeypatch.setattr(barcode, "decode_gs1", lambda *a, **k: [])
    cfg = load_config(resolve("config/box_pipeline_config_focus.yaml"))
    class Reader:
        calls = 0
        def read_once(self, image):
            self.calls += 1
            return OcrResult([OcrToken(product, .99), OcrToken(ref, .99)] if self.calls == 1 else
                             [token(diameter, 20, 10), token(length, 100, 10),
                              token("mm", 20, 30), token("mm", 100, 30)])
    req = parse_request(f"{product} {diameter} x {length}")
    (ev, decision, *_), _ = read_spine_bands(np.zeros((50, 700, 3), np.uint8), Reader(), cfg, req)
    assert decision == MATCH and combined_score(ev.support[0])[0] == 2

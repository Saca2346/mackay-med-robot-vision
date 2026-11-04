import math

import numpy as np

from src.box_demo import load_positions, lookup, section_slot, to_text, to_tm_message
from src.box_inventory import STATUS_DIAMBIL, STATUS_OK, InvBox, Inventory
from src.box_lock import place_candidates
from src.box_sections import (assign_sections, distance_for, measure_hfov, plan_sections, slot_in_view,
                              visible_width)


def test_fov_geometry_round_trips():
    w = visible_width(50, 90)
    assert math.isclose(w, 100, rel_tol=1e-9)                      # 90 deg: twice the distance
    assert math.isclose(measure_hfov(100, 1920, 1920, 50), 90, rel_tol=1e-9)
    assert math.isclose(measure_hfov(50, 960, 1920, 50), 90, rel_tol=1e-9)   # half the width over half the frame
    d = distance_for(40, 90, margin=0.1)
    assert math.isclose(visible_width(d, 90) * 0.8, 40, rel_tol=1e-9)


def test_plan_takes_as_few_sections_as_fit_with_the_margin():
    p = plan_sections(120, 50, 90, frame_px=1920, spine=3)         # sees 100 cm, 80 cm usable -> 2 sections
    assert p.sections == 2 and p.fits
    assert p.edges == [(0, 60), (60, 120)] and p.centres == [30, 90]
    assert math.isclose(p.spine_px, 3 * 19.2)
    assert plan_sections(80, 50, 90).sections == 1                 # exactly the usable width
    assert not plan_sections(300, 50, 90).fits                     # 4 sections: camera too close for 2-3
    assert plan_sections(120, 50, 90, sections=3).edges[1] == (40, 80)


def test_sections_are_fixed_and_a_taken_box_is_not_counted():
    cols = {f"K{c:02d}": c for c in range(1, 21) if c != 2}        # K02 taken: its place stays empty
    a = assign_sections(cols, 20, 2)
    assert a["K01"] == (1, 1) and a["K03"] == (1, 2)               # slot = boxes on the shelf from the left end
    assert a["K10"] == (1, 9) and a["K11"] == (2, 1) and a["K20"] == (2, 10)
    b = assign_sections(cols, 20, 3)
    assert {s for s, _ in b.values()} == {1, 2, 3}
    assert b["K07"] == (1, 6) and b["K08"] == (2, 1)


def test_slot_in_view_only_with_every_box_of_the_section_detected():
    xs = [500, 100, 300, 700]
    assert slot_in_view(xs, 4, 1) == 1 and slot_in_view(xs, 4, 3) == 0
    assert slot_in_view(xs, 5, 1) is None                          # one box missed
    assert slot_in_view(xs, 4, 5) is None and slot_in_view(xs, 4, 0) is None


def test_frame_place_map_misses_a_closer_camera_and_the_slot_count_does_not():
    # 29-10 Task 3 runs: second run 0.9 pitch aside and 32 % closer. Shelf of 20 boxes, pitch 0.04 of the frame.
    mapped = {f"K{i}": (0.25 + 0.04 * i, 0.5, 0.025) for i in range(20)}
    seen = np.array([(0.5 + 1.32 * (x - 0.5) - 0.036, 0.5) for x, _, _ in mapped.values()], np.float32)
    right = [next(place_candidates(seen, mapped, [f"K{i}"]), (None,))[0] == i for i in range(20)]
    assert sum(right) <= 5                                         # shift taken out, scale not: none or a neighbour
    assert all(slot_in_view(seen[:, 0].tolist(), 20, i + 1) == i for i in range(20))


def test_position_table_reads_the_section(tmp_path):
    p = tmp_path / "posisi.csv"
    p.write_text("box_id,rak,level,kolom,sumber,seksi\nK01,R1,2,5,tim,1\nK02,R1,2,6,tim\n", encoding="utf-8")
    pos = load_positions(p)
    assert str(pos["K01"]) == "rack R1, level 2, section 1, column 5"
    assert str(pos["K02"]) == "rack R1, level 2, column 6"


def test_robot_message_carries_section_and_the_slot_among_the_boxes_still_there(tmp_path):
    inv = Inventory(tmp_path / "inventaris.csv")
    inv.boxes = {b: InvBox(b, "angiolite", 2.5, 29.0 + k, "ivascular", status=STATUS_OK)
                 for k, b in enumerate(("K11", "K12", "K13", "K14"))}
    inv.boxes["K12"].status = STATUS_DIAMBIL                      # taken: its place is empty now
    p = tmp_path / "posisi.csv"
    p.write_text("box_id,rak,level,kolom,sumber,seksi\nK11,R1,2,11,tim,2\nK12,R1,2,12,tim,2\nK13,R1,2,13,tim,2\n"
                 "K14,R1,2,14,tim,2\nK10,R1,2,10,tim,1\n", encoding="utf-8")
    res = lookup("ivascular angiolite 2.5 x 31", inv, load_positions(p))
    msg = to_tm_message(res)
    assert (msg["box_id"], msg["section"], msg["slot"], msg["section_boxes"]) == ("K13", "2", 2, 3)
    assert "box 2 of 3 from the section's left end" in to_text(res)
    assert section_slot("K10", load_positions(p), inv) == (1, 1)

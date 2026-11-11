import json, os
os.chdir(os.path.expanduser("~/Downloads/claude_offline_20261101/aipc_20261104/tasks/out"))
for v in ["v6s", "v5det", "v5srv", "v5en", "easy", "tess"]:
    row = [v]
    for s in ["t1_drive", "t2_drive", "t1_orig"]:
        try:
            c = json.load(open(f"ocr_{v}/{s}_score.json"))["summary"]["cells"]; row.append(f"{c.get('TP',0)}/{c.get('FP',0)}")
        except Exception as e: row.append("-")
    try:
        t = json.load(open(f"ocr_{v}/task3.json"))["summary"]; row.append(f"{t['on_shelf_green_right_box']}/{t['wrong_box_green']}/{t.get('absent_any_green')}")
    except Exception: row.append("-")
    for s in ["t1_drive", "t2_drive"]:
        try:
            d = json.load(open(f"ocrt_{v}/{s}_time.json")); row.append(f"pick {d['pick_s_median']} all {d['all_read_s_median']} box {d['picked_box']}")
        except Exception: row.append("-")
    print(" | ".join(map(str, row)))

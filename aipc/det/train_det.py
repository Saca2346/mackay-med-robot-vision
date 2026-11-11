"""AI PC box detector candidates (4 Nov 2026).
Usage: python train_det.py MODEL IMGSZ BATCH [EPOCHS] [split] [PHOTO_MULT] [FRAME_STEP]

split "hold" (default) = honest-evaluation model: trained WITHOUT the 28-Sep photo session (87 of the 196 human-reviewed
photos), which is the test set (eval_det.py). split "hold2" = second fold: trained WITHOUT the 23-Sep session, which is
its test set. split "all" = final model, all 196 photos (its number is not a test).
PHOTO_MULT (default 2, as model B) = how often each photo is listed; FRAME_STEP (default 3) = every n-th webcam frame.
Recipe = model B's (exp_outline_max_20261020/train_max.py): 196 reviewed photos x2 + static-pose webcam frames (here every
3rd frame, B used all), BadCamera augmentation, rotation +-180, AdamW 0.0005 cosine, close_mosaic 15, val off, last epoch
kept (no checkpoint chosen on test data). Start: the official pretrained OBB weights (not model B, which saw the test
photos); MODEL *-obb.yaml (YOLO12, no published OBB weights): the official detection weights, new OBB head. Reads the master project read-only; writes only under this folder. OOM on the shared GPU: resume, batch halved.
"""
import glob
import json
import os
import sys
import time

M = os.path.expanduser("~/Downloads/prelabel_size_label_full")
X = os.path.expanduser("~/Downloads/exp_outline_max_20261020")
E = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(M, "tools"))
from train_best import bad_camera_trainer  # noqa: E402

import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

# Shared GPU: at most 60% of its memory for training, so the reader tests and other users keep room. A file PAUSE next
# to this script holds the next job back (timing runs need the GPU free of training).
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("GPU_FRACTION", "0.6")), 0)
while os.path.exists(os.path.join(E, "PAUSE")):
    time.sleep(30)
model, imgsz, batch = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 100
split = sys.argv[5] if len(sys.argv) > 5 else "hold"
mult = int(sys.argv[6]) if len(sys.argv) > 6 else 2
step = int(sys.argv[7]) if len(sys.argv) > 7 else 3
photos = sorted(glob.glob(os.path.join(M, "export", "cv", "all", "images", "train", "*")))
assert len(photos) == 196
held = {"hold": "IMG_20260928", "hold2": "IMG_20260923"}.get(split)
test = [p for p in photos if held and os.path.basename(p).startswith(held)]
train_photos = [p for p in photos if p not in test]
man = json.load(open(os.path.join(X, "dataset", "manifest.json")))
frames = [os.path.join(X, "dataset", "images", m["name"] + ".jpg") for m in man if m["day"] in {"d15", "d17", "d18", "d20"}]
frames = frames[::step]
tag = "" if (mult, step) == (2, 3) else f"_p{mult}f{step}"
lst = os.path.join(E, f"train_{split}{tag}.txt")
open(lst, "w").write("\n".join(train_photos * mult + frames) + "\n")
if test:
    open(os.path.join(E, f"test_photos{'' if split == 'hold' else '_' + split}.txt"), "w").write("\n".join(test) + "\n")
data = os.path.join(E, f"data_{split}{tag}.yaml")
open(data, "w").write(f"train: {lst}\nval: {lst}\nnames:\n  0: box\n  1: size_label\n  2: expiry_label\n")
name = f"{os.path.splitext(model)[0]}_{imgsz}_{split}{tag}"
print(f"{name}: {len(train_photos)} photos x{mult} + {len(frames)} frames, test {len(test)} photos", flush=True)
args = dict(trainer=bad_camera_trainer(), data=data, imgsz=imgsz, epochs=epochs, batch=batch, device="0", workers=6,
            degrees=180.0, fliplr=0.0, optimizer="AdamW", lr0=0.0005, lrf=0.05, cos_lr=True, warmup_epochs=2.0,
            close_mosaic=15, patience=0, project=os.path.join(E, "runs"), name=name, exist_ok=True, plots=False, seed=0,
            verbose=False, val=False, save_period=10)
last = os.path.join(E, "runs", name, "weights", "last.pt")
t0 = time.time()
for attempt in range(7):
    try:
        if attempt == 0 or not os.path.exists(last):
            # an architecture without published OBB weights (YOLO12): built from its OBB yaml, the backbone and neck
            # from the official pretrained detection weights (yolo12s-obb.yaml <- yolo12s.pt); the OBB head starts new
            net = YOLO(model).load(model.replace("-obb.yaml", ".pt")) if model.endswith(".yaml") else YOLO(model)
            net.train(**args)
        else:
            YOLO(last).train(resume=True)
        open(os.path.join(E, "runs", name, "done"), "w").write(f"{time.time() - t0:.0f} s\n")
        break
    except Exception as e:                    # shared GPU: out of memory etc.
        print(f"attempt {attempt} failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
        if "out of memory" in str(e).lower() and not os.path.exists(last):
            args["batch"] = max(1, args["batch"] // 2)
        time.sleep(60)
print(f"TRAIN {name} FINISHED {time.time() - t0:.0f} s", flush=True)

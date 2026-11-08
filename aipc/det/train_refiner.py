"""Train the edge refiner (refine_lib.py) on the DEV photos only (23 Sep + 1 Oct); the 28-Sep test photos are not read.

  python train_refiner.py PRED_NAME [EPOCHS] [SOURCE] [TAG] [--all]
      PRED_NAME = cached detector outlines in pred/<PRED_NAME>/ (snap_eval.py)

Inputs per owner box: the detector outline matched to it (IoU >= 0.5) and 8 jittered copies (centre, length, width,
angle) of the owner outline (SOURCE truth, v1) or of that detector outline (SOURCE pred, v2: the detector's own bias,
sides about 5% wide, is in the inputs; plus 2 copies of the owner outline). Without --all one dev photo in 7 is the
validation set (by photo) and the epoch is chosen on it; with --all every dev photo trains and the last epoch is kept
(the set-up chosen by cv_refine.py). Writes weights/refiner_<TAG>.pt and eval/refiner_train_<TAG>.json.
"""
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import refine_lib as rl
import tightlib as tl

E = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)
torch.manual_seed(0)
JITTER = 8


def jitter(gt, k=1.0):
    c, u, v, L, W = tl.frame_of(gt)
    a = np.radians(rng.normal(0, 0.4 * k))
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    u2, v2 = R @ u, R @ v
    c2 = c + u * rng.normal(0, 0.012 * k * 2 * L) + v * rng.normal(0, 0.04 * k * 2 * W)
    L2, W2 = L * (1 + rng.normal(0, 0.012 * k)), W * (1 + rng.normal(0, 0.06 * k))
    return np.array([c2 - L2 * u2 - W2 * v2, c2 + L2 * u2 - W2 * v2, c2 + L2 * u2 + W2 * v2, c2 - L2 * u2 + W2 * v2])


def build_photo(p, pred_name, source, jit):
    """Strips and targets of one photo -> (end_x, end_y, side_x, side_y), each (2 * inputs, ...)."""
    img = cv2.imread(p)
    gts = tl.truth(p, img.shape)
    prs = list(np.load(os.path.join(E, "pred", pred_name, os.path.basename(p) + ".npy")))
    inputs = []
    if gts and prs:
        m = np.array([[tl.iou(g, q) for q in prs] for g in gts])
        pairs = [(gts[i], prs[j]) for i, j in tl.greedy(m)]
        inputs += pairs
        if source == "pred":
            inputs += [(g, jitter(q, 0.5)) for g, q in pairs for _ in range(jit)]
    for g in gts:
        inputs += [(g, jitter(g)) for _ in range(jit if source == "truth" else 2)]
    ex, ey, sx, sy = [], [], [], []
    for g, q in inputs:
        e, s, (re, rs) = rl.strips(img, q)
        off = rl.truth_offsets(q, g)
        ex.append(e)
        ey.append(off[:2] / re)
        sx.append(s)
        sy.append(off[2:] / rs)
    if not inputs:
        return None
    return (np.concatenate(ex), np.concatenate(ey), np.concatenate(sx), np.concatenate(sy))


def stack(parts):
    parts = [x for x in parts if x is not None]
    return tuple(np.concatenate([x[k] for x in parts]) for k in range(4))


def labels(y, N, sigma=1.5):
    """soft label over N bins for offsets y in [-1, 1] (fraction of the range); outside -> masked."""
    idx = (y + 1) / 2 * (N - 1)
    ok = (y > -1) & (y < 1)
    b = torch.arange(N).float()[None]
    t = torch.exp(-0.5 * ((b - torch.as_tensor(idx).float()[:, None]) / sigma) ** 2)
    return t / t.sum(1, keepdim=True), torch.as_tensor(ok)


def augment(x):
    """x: (B, P, N, 3) uint8 -> augmented copy: flip along the edge, brightness / contrast, lower resolution."""
    x = x.copy()
    B = len(x)
    flip = rng.random(B) < 0.5
    x[flip] = x[flip, ::-1]
    out = []
    for k in range(B):
        a = x[k].astype(np.float32)
        a = (a - a.mean()) * rng.uniform(0.6, 1.4) + a.mean() + rng.uniform(-30, 30)
        if rng.random() < 0.5:                        # webcam-like resolution: down 1.5-4x across the edge, back up
            f = rng.uniform(1.5, 4)
            n = a.shape[1]
            a = cv2.resize(cv2.resize(a, (max(8, int(n / f)), a.shape[0]), interpolation=cv2.INTER_AREA), (n, a.shape[0]),
                           interpolation=cv2.INTER_LINEAR)
        if rng.random() < 0.3:
            a += rng.normal(0, 6, a.shape)
        out.append(np.clip(a, 0, 255).astype(np.uint8))
    return np.stack(out)


def fit(net, X, Y, N, dev_, epochs, Xv=None, Yv=None, log=True):
    """Xv None: train only, keep the last epoch; else keep the epoch with the lowest validation error."""
    yl, ok = labels(Y, N)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * ((int(ok.sum()) + 255) // 256))
    best, best_state, hist = 1e9, None, []
    grid = np.linspace(-1, 1, N)
    for ep in range(epochs):
        net.train()
        perm = rng.permutation(np.flatnonzero(ok.numpy()))
        for b in range(0, len(perm), 256):
            i = perm[b:b + 256]
            xb = rl.to_tensor(augment(X[i])).to(dev_)
            loss = -(yl[i].to(dev_) * F.log_softmax(net(xb), 1)).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        if Xv is None:
            continue
        net.eval()
        okv = (Yv > -1) & (Yv < 1)
        with torch.no_grad():
            iv = np.flatnonzero(okv)
            pv = torch.cat([torch.softmax(net(rl.to_tensor(Xv[iv[b:b + 512]]).to(dev_)), 1).cpu()
                            for b in range(0, len(iv), 512)]).numpy()
        err = float(np.median(np.abs(pv @ grid - Yv[iv])))
        hist.append(err)
        if err < best:
            best, best_state = err, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        if log:
            print(f"ep {ep} val median |err| {err:.4f} of the range", flush=True)
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    return best_state, hist


def train_pair(data, dev_, epochs, val=None):
    """data / val = (end_x, end_y, side_x, side_y) -> (end_state, side_state, end_hist, side_hist)."""
    Ex, Ey, Sx, Sy = data
    end_net, side_net = rl.EdgeNet(rl.END_P).to(dev_), rl.EdgeNet(rl.SIDE_P).to(dev_)
    v = val or (None, None, None, None)
    end_state, eh = fit(end_net, Ex, Ey.reshape(-1), rl.END_N, dev_, epochs, v[0], None if val is None else v[1].reshape(-1))
    side_state, sh = fit(side_net, Sx, Sy.reshape(-1), rl.SIDE_N, dev_, epochs, v[2], None if val is None else v[3].reshape(-1))
    return end_state, side_state, eh, sh


def flat(d):
    Ex, Ey, Sx, Sy = d
    return Ex.reshape(-1, *Ex.shape[-3:]), Ey, Sx.reshape(-1, *Sx.shape[-3:]), Sy


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--all"]
    pred_name = args[0]
    EPOCHS = int(args[1]) if len(args) > 1 else 40
    SOURCE = args[2] if len(args) > 2 else "truth"
    TAG = args[3] if len(args) > 3 else "v1"
    ALL = "--all" in sys.argv
    t0 = time.time()
    dev = tl.photos("dev")
    val_photos = [] if ALL else dev[::7]
    tr = [p for p in dev if p not in val_photos]
    data = flat(stack([build_photo(p, pred_name, SOURCE, JITTER) for p in tr]))
    val = flat(stack([build_photo(p, pred_name, SOURCE, 2) for p in val_photos])) if val_photos else None
    print("samples end", len(data[0]), "side", len(data[2]), f"{time.time() - t0:.0f} s", flush=True)
    dev_ = "cuda" if torch.cuda.is_available() else "cpu"
    end_state, side_state, end_hist, side_hist = train_pair(data, dev_, EPOCHS, val)
    os.makedirs(os.path.join(E, "weights"), exist_ok=True)
    torch.save({"end": end_state, "side": side_state, "iters": 1, "source": SOURCE},
               os.path.join(E, "weights", f"refiner_{TAG}.pt"))
    json.dump({"pred": pred_name, "epochs": EPOCHS, "source": SOURCE, "all": ALL, "train_photos": len(tr),
               "val_photos": len(val_photos), "end_val_hist": end_hist, "side_val_hist": side_hist,
               "seconds": round(time.time() - t0)},
              open(os.path.join(E, "eval", f"refiner_train_{TAG}.json"), "w"), indent=1)
    print("REFINER_TRAINED", TAG, f"{time.time() - t0:.0f} s", flush=True)

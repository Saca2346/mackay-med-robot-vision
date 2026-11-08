"""Learned edge refiner (6 Nov 2026): the 4 edges of a detected box moved to the box's real edges.

The snap (tightlib.snap) moves each edge to the strongest gradient, and near the ends of a box that is often a label
edge or a shadow: after it, 49% of the 28-Sep boxes still end short of the box (> 1%) and 58% have a side past it
(> 5%). Here a small CNN looks at a full-resolution strip across each edge and says where the box edge is.

Strips, in the frame of the detected rectangle (u = long axis, v = short axis, half sizes L, W):
  end strip   END_P x END_N: along the end (t in +-0.8 W) x across it (d in +-END_R * 2L), outward = increasing d
  side strip  SIDE_P x SIDE_N: along the side (t in +-0.9 L) x across it (d in +-SIDE_R * 2W)
The net gives a distribution over the N positions across the edge; the edge goes to its expected position. The box
stays a rectangle (the owner outlines are exact rectangles: corner angles within 0.003 degrees on 90% of the boxes).
"""
import cv2
import numpy as np
import torch
import torch.nn as nn

import tightlib as tl

END_P, END_N, END_R = 32, 128, 0.05
SIDE_P, SIDE_N, SIDE_R = 128, 64, 0.20


def frame(poly):
    return tl.frame_of(np.asarray(poly, np.float64))


def strip(img, c, axis, perp, half, other, sign, P, N, R, extent, part=None):
    """Remapped BGR strip (P, N, 3) uint8: rows along the edge, columns across it, outward to the right. part -1 / +1:
    only one half of the edge (for a tilted edge, quad())."""
    n = axis * sign
    ts = np.linspace(-extent * other, extent * other, P) if part is None else \
        np.linspace(0.0, part * extent * other, P)
    ds = np.linspace(-R, R, N)
    pts = (c + n * half)[None, None, :] + ts[:, None, None] * perp[None, None, :] + ds[None, :, None] * n[None, None, :]
    return cv2.remap(img, pts[..., 0].astype(np.float32), pts[..., 1].astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def strips(img, poly):
    """-> (ends (2, P, N, 3), sides (2, P, N, 3), ranges (R_end_px, R_side_px)); order: +u, -u, +v, -v."""
    c, u, v, L, W = frame(poly)
    re, rs = END_R * 2 * L, SIDE_R * 2 * W
    ends = [strip(img, c, u, v, L, W, s, END_P, END_N, re, 0.8) for s in (1, -1)]
    sides = [strip(img, c, v, u, W, L, s, SIDE_P, SIDE_N, rs, 0.9) for s in (1, -1)]
    return np.stack(ends), np.stack(sides), (re, rs)


def truth_offsets(poly, gt):
    """Where the truth edges are, along each outward normal of the detected rectangle, relative to its edges (px)."""
    c, u, v, L, W = frame(poly)
    gu, gv = (gt - c) @ u, (gt - c) @ v
    k = np.argsort(gu)
    m = np.argsort(gv)
    return np.array([gu[k[2:]].mean() - L, -gu[k[:2]].mean() - L, gv[m[2:]].mean() - W, -gv[m[:2]].mean() - W])


def rect_from(poly, off):
    """Rectangle of poly with its 4 edges moved outward by off = (+u, -u, +v, -v) px (as tightlib.snap)."""
    c, u, v, L, W = frame(poly)
    a_u, b_u, a_v, b_v = -(L + off[1]), L + off[0], -(W + off[3]), W + off[2]
    return np.array([c + a_u * u + a_v * v, c + b_u * u + a_v * v, c + b_u * u + b_v * v, c + a_u * u + b_v * v])


class EdgeNet(nn.Module):
    """(B, 3, P, N) strip -> (B, N) logits over the positions across the edge."""

    def __init__(self, P, width=32):
        super().__init__()
        def blk(i, o, pool):
            return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True)] + \
                   ([nn.MaxPool2d((2, 1))] if pool else [])
        layers, ch, p = [], 3, P
        for k in range(6):
            pool = p > 2
            layers += blk(ch, width if k < 3 else 2 * width, pool)
            ch = width if k < 3 else 2 * width
            p = p // 2 if pool else p
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Conv1d(ch, ch, 5, padding=2), nn.ReLU(inplace=True), nn.Conv1d(ch, 1, 1))

    def forward(self, x):
        f = self.body(x).mean(2)
        return self.head(f)[:, 0]


def to_tensor(a):
    """(..., P, N, 3) uint8 BGR -> (B, 3, P, N) float, per-strip normalised."""
    t = torch.as_tensor(np.ascontiguousarray(a)).float().permute(0, 3, 1, 2) / 255.0
    m = t.mean((1, 2, 3), keepdim=True)
    s = t.std((1, 2, 3), keepdim=True) + 0.05
    return (t - m) / s


class Refiner:
    def __init__(self, path, device="cpu"):
        ck = torch.load(path, map_location=device, weights_only=False)
        self.end, self.side = EdgeNet(END_P), EdgeNet(SIDE_P)
        self.end.load_state_dict(ck["end"])
        self.side.load_state_dict(ck["side"])
        self.end.eval().to(device)
        self.side.eval().to(device)
        self.device, self.iters, self.side_bias = device, ck.get("iters", 1), ck.get("side_bias")
        self.end_bias = ck.get("end_bias", 0.0)

    @torch.no_grad()
    def offsets(self, img, poly):
        e, s, (re, rs) = strips(img, poly)
        pe = torch.softmax(self.end(to_tensor(e).to(self.device)), 1).cpu().numpy()
        ps = torch.softmax(self.side(to_tensor(s).to(self.device)), 1).cpu().numpy()
        de = pe @ np.linspace(-re, re, END_N)
        ds = ps @ np.linspace(-rs, rs, SIDE_N)
        return np.array([de[0], de[1], ds[0], ds[1]])

    @torch.no_grad()
    def quad(self, img, poly, sides=False):
        """The refined rectangle, then each end (and with sides=True each side, by the side net) placed separately on its
        two halves, so the edge may tilt: a photo taken close and at an angle shows a box as a trapezoid, which no
        rectangle fits (Task 1 Drive photo, 6 Nov). Corners = where neighbouring edge lines cross."""
        p = self(img, poly)
        c, u, v, L, W = frame(p)
        lines = []                                    # (point, direction) per edge: +u, +v, -u, -v (corner order)
        for axis, perp, half, other, P, N, Rf, net, ext, learn in (
                (u, v, L, W, END_P, END_N, END_R, self.end, 0.8, True), (v, u, W, L, SIDE_P, SIDE_N, SIDE_R, self.side, 0.9, sides),
                (-u, v, L, W, END_P, END_N, END_R, self.end, 0.8, True), (-v, u, W, L, SIDE_P, SIDE_N, SIDE_R, self.side, 0.9, sides)):
            base = c + axis * half
            if not learn:
                lines.append((base, perp))
                continue
            R = Rf * 2 * (L if net is self.end else W)
            pts = []
            for part in (-1, 1):
                s = strip(img, c, axis, perp, half, other, 1, P, N, R, ext, part)
                pr = torch.softmax(net(to_tensor(s[None]).to(self.device)), 1).cpu().numpy()[0]
                d = float(pr @ np.linspace(-R, R, N)) + (self.end_bias * 2 * L if net is self.end else 0.0)
                pts.append(base + part * 0.5 * ext * other * perp + d * axis)
            lines.append((pts[0], pts[1] - pts[0]))

        def cross(a, b):
            (p1, d1), (p2, d2) = a, b
            A = np.array([d1, -d2]).T
            if abs(np.linalg.det(A)) < 1e-9:
                return None
            s = np.linalg.solve(A, p2 - p1)
            return p1 + s[0] * d1
        q = [cross(lines[k], lines[(k + 1) % 4]) for k in range(4)]
        if any(x is None for x in q) or tl.iou(np.array(q), p) < 0.8:
            return p
        return np.array(q)

    def __call__(self, img, poly, iters=None, side_bias="ck", end_bias="ck"):
        """side_bias None: the side net places the sides; a number b: the ends by the net, each side moved by b x the
        box width from the detector's (a constant, chosen on the dev photos); end_bias: each end moved out by this x the
        box length after the net; "ck": as stored with the weights."""
        b = self.side_bias if side_bias == "ck" else side_bias
        eb = self.end_bias if end_bias == "ck" else end_bias
        p = np.asarray(poly, np.float64)
        for _ in range(iters or self.iters):
            off = self.offsets(img, p)
            if b is not None:
                off[2:] = 0.0
            p = rect_from(p, off)
        if b is not None:
            W = frame(p)[4]
            p = rect_from(p, [0.0, 0.0, b * 2 * W, b * 2 * W])
        if eb:
            L = frame(p)[3]
            p = rect_from(p, [eb * 2 * L, eb * 2 * L, 0.0, 0.0])
        return p

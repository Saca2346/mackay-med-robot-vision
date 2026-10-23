"""
Grasp data of the PICKed box for the arm team (pure geometry; the camera and depth come from src/box_camera.py).

Input: the rotated box of the target spine in the colour image (4 corners, px), the depth image aligned to that colour
image (metres, 0 = no depth), the colour camera intrinsics, the rotated boxes of the other spines and the label polygons
of the target (size / expiry label = no-grip zone). Output: one dict, JSON-ready:

  center_px, angle_deg (spine axis vs image vertical), spine_width_px / spine_length_px,
  xyz_m          centre of the visible spine in the CAMERA frame (x right, y down, z forward), from the median depth
  normal         unit vector of the spine face (plane fit), pointing towards the camera; approach = -normal
  spine_width_mm / spine_length_mm (visible part)
  gap_left_mm / gap_right_mm   free space to the nearest neighbour spine on each side (image left / right)
  no_grip_px     label polygons (do not grip there; the barcode / label must stay readable)
  depth_valid    share of the spine with a valid depth reading (low = glare / too close: do not trust xyz)

Without depth (video replay, webcam) the pixel fields are filled and the metric fields are None. Transforming the camera
frame into the robot frame (hand-eye calibration) is the arm team's step.
"""
from __future__ import annotations

import math

import numpy as np

Z_MIN, Z_MAX = 0.15, 3.0          # D455 usable range for this use (m)
INNER = 0.7                        # depth is read in the inner 70 % of the spine: edges mix in the neighbours / shelf


def _axes(p: np.ndarray):
    """Rotated box -> (centre, unit long axis, unit short axis, long px, short px)."""
    c = p.mean(axis=0)
    e1, e2 = p[1] - p[0], p[2] - p[1]
    l1, l2 = float(np.linalg.norm(e1)), float(np.linalg.norm(e2))
    long_v, long_l, short_l = (e1, l1, l2) if l1 >= l2 else (e2, l2, l1)
    u = long_v / max(np.linalg.norm(long_v), 1e-9)
    if u[1] < 0:                                   # long axis pointing down the image
        u = -u
    v = np.array([u[1], -u[0]])                    # short axis
    if v[0] < 0:                                   # pointing to the image right
        v = -v
    return c, u, v, long_l, short_l


def _inside_mask(p: np.ndarray, shape, scale: float = 1.0) -> np.ndarray:
    import cv2
    c = p.mean(axis=0)
    q = (c + (p - c) * scale).astype(np.int32)
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(m, q, 1)
    return m.astype(bool)


def _gaps(c, u, v, long_l, short_l, neighbours):
    """Free space (px) to the nearest neighbour on the left / right of the spine, measured along the short axis between
    the spine edge and the neighbour's nearest corner, for neighbours that overlap the spine along its length."""
    left = right = None
    for nb in neighbours:
        q = np.asarray(nb, float).reshape(4, 2) - c
        along, across = q @ u, q @ v
        if along.max() < -long_l / 2 or along.min() > long_l / 2:
            continue                                   # above / below the spine, not beside it
        if across.min() > 0:
            g = max(0.0, float(across.min()) - short_l / 2)
            right = g if right is None else min(right, g)
        elif across.max() < 0:
            g = max(0.0, float(-across.max()) - short_l / 2)
            left = g if left is None else min(left, g)
        else:
            if across.mean() > 0:
                right = 0.0
            else:
                left = 0.0
    return left, right


def grasp_info(obb_px, depth_m=None, intr=None, neighbours=(), no_grip=()) -> dict:
    p = np.asarray(obb_px, float).reshape(4, 2)
    c, u, v, long_l, short_l = _axes(p)
    gl, gr = _gaps(c, u, v, long_l, short_l, neighbours)
    out = {
        "center_px": [round(float(c[0]), 1), round(float(c[1]), 1)],
        "obb_px": [[round(float(x), 1), round(float(y), 1)] for x, y in p],
        "angle_deg": round(math.degrees(math.atan2(u[0], u[1])), 1),
        "spine_width_px": round(short_l, 1),
        "spine_length_px": round(long_l, 1),
        "gap_left_px": None if gl is None else round(gl, 1),
        "gap_right_px": None if gr is None else round(gr, 1),
        "no_grip_px": [[[round(float(x), 1), round(float(y), 1)] for x, y in np.asarray(q, float).reshape(-1, 2)]
                       for q in no_grip],
        "xyz_m": None, "normal": None, "approach": None, "spine_width_mm": None, "spine_length_mm": None,
        "gap_left_mm": None, "gap_right_mm": None, "depth_valid": 0.0,
    }
    if depth_m is None or intr is None:
        return out
    d = np.asarray(depth_m, np.float32)
    m = _inside_mask(p, d.shape, INNER)
    z = d[m]
    ok = (z > Z_MIN) & (z < Z_MAX)
    out["depth_valid"] = round(float(ok.mean()) if z.size else 0.0, 2)
    if ok.sum() < 20:
        return out
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    zc = float(np.median(z[ok]))
    xyz = [(c[0] - cx) * zc / fx, (c[1] - cy) * zc / fy, zc]
    out["xyz_m"] = [round(float(a), 4) for a in xyz]
    # Projected lengths at the centre depth: each rotated image axis has its own scale when fx != fy.
    short_scale = zc * math.hypot(v[0] / fx, v[1] / fy)
    long_scale = zc * math.hypot(u[0] / fx, u[1] / fy)
    out["spine_width_mm"] = round(short_l * short_scale * 1000, 1)
    out["spine_length_mm"] = round(long_l * long_scale * 1000, 1)
    out["gap_left_mm"] = None if gl is None else round(gl * short_scale * 1000, 1)
    out["gap_right_mm"] = None if gr is None else round(gr * short_scale * 1000, 1)
    ys, xs = np.nonzero(m)
    zz = d[ys, xs]
    sel = (zz > Z_MIN) & (zz < Z_MAX) & (np.abs(zz - zc) < 0.05)       # spine face only (5 cm band)
    if sel.sum() >= 30:
        pts = np.stack([(xs[sel] - cx) * zz[sel] / fx, (ys[sel] - cy) * zz[sel] / fy, zz[sel]], axis=1)
        if len(pts) > 4000:
            pts = pts[:: len(pts) // 4000]
        n = np.linalg.svd(pts - pts.mean(axis=0), full_matrices=False)[2][-1]
        if n[2] > 0:                                   # towards the camera = negative z
            n = -n
        out["normal"] = [round(float(a), 3) for a in n]
        out["approach"] = [round(float(-a), 3) for a in n]
    return out


# ------------------------------------------------------------------ hand-off to the grasp planner (GraspGen-X)
# GraspGen / GraspGen-X do NOT take a bounding box: they take a segmented partial point cloud of the object (or a scene
# point cloud with per-object labels) and a gripper name / config, and return ranked grasp poses. The rotated box of the
# spine is therefore only the MASK that selects the depth pixels; what Leon's module receives is the point cloud.
OBJECT_POINTS = 2048               # point count GraspGen's model card quotes for one object
POINTCLOUD_FRAME = "camera_color_m"  # x right, y down, z forward, metres, colour camera of the D455 (depth aligned to colour)


def _deproject(xs, ys, zz, intr) -> np.ndarray:
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    return np.stack([(xs - cx) * zz / fx, (ys - cy) * zz / fy, zz], axis=1).astype(np.float32)


def object_point_cloud(obb_px, depth_m, intr, n_points: int = OBJECT_POINTS, scale: float = 0.95,
                       band_m: float = 0.08, seed: int = 0):
    """Partial point cloud of the target box -> (N, 3) float32, metres, colour camera frame; None without usable depth.

    The pixels inside the rotated box (shrunk to `scale` so neighbours and the shelf do not leak in) are de-projected with
    the intrinsics. Pixels more than `band_m` from the median depth of the box are dropped (shelf / hand behind or in
    front). At most `n_points` points (uniform random, fixed seed); fewer when fewer are valid: nothing is duplicated.
    Fewer than 100 valid points = no cloud (glare, too close): the robot must not plan from it."""
    if depth_m is None or intr is None:
        return None
    d = np.asarray(depth_m, np.float32)
    m = _inside_mask(np.asarray(obb_px, float).reshape(4, 2), d.shape, scale)
    ys, xs = np.nonzero(m)
    zz = d[ys, xs]
    ok = (zz > Z_MIN) & (zz < Z_MAX)
    if ok.sum() < 100:
        return None
    zc = float(np.median(zz[ok]))
    ok &= np.abs(zz - zc) < band_m
    if ok.sum() < 100:
        return None
    pts = _deproject(xs[ok], ys[ok], zz[ok], intr)
    if len(pts) > n_points:
        pts = pts[np.random.default_rng(seed).choice(len(pts), n_points, replace=False)]
    return pts


def scene_point_cloud(depth_m, intr, target_obb, neighbours=(), stride: int = 4):
    """Whole scene (down-sampled depth) with one label per point -> (points (M, 3) float32, labels (M,) int16).
    Label 0 = background / shelf, 1 = the target box, 2.. = the other detected boxes (in the order given). The planner can
    reject grasps that collide with the labelled rest of the scene. Labels come from the detected rotated boxes only."""
    if depth_m is None or intr is None:
        return None
    d = np.asarray(depth_m, np.float32)
    import cv2
    label = np.zeros(d.shape[:2], np.int16)
    for i, nb in enumerate(neighbours):
        cv2.fillConvexPoly(label, np.asarray(nb, float).reshape(4, 2).astype(np.int32), 2 + i)
    cv2.fillConvexPoly(label, np.asarray(target_obb, float).reshape(4, 2).astype(np.int32), 1)
    ys, xs = np.mgrid[0:d.shape[0]:stride, 0:d.shape[1]:stride]
    ys, xs = ys.ravel(), xs.ravel()
    zz = d[ys, xs]
    ok = (zz > Z_MIN) & (zz < Z_MAX)
    return _deproject(xs[ok], ys[ok], zz[ok], intr), label[ys[ok], xs[ok]]


def save_pointcloud_npz(path, obb_px, depth_m, intr, neighbours=(), frame_size=None):
    """Write the GraspGen-X hand-off arrays of one PICK; returns {file, n_object_points, frame} for grasp.jsonl, or None.
    Arrays: object_points (N,3), scene_points (M,3), scene_labels (M,), intrinsics [fx, fy, cx, cy], obb_px (4,2),
    frame_size [w, h]. Units metres, frame POINTCLOUD_FRAME. The camera -> robot base transform is the arm side's."""
    obj = object_point_cloud(obb_px, depth_m, intr)
    if obj is None:
        return None
    scene, labels = scene_point_cloud(depth_m, intr, obb_px, neighbours)
    np.savez_compressed(path, object_points=obj, scene_points=scene, scene_labels=labels,
                        intrinsics=np.float32([intr["fx"], intr["fy"], intr["cx"], intr["cy"]]),
                        obb_px=np.asarray(obb_px, np.float32).reshape(4, 2),
                        frame_size=np.int32(frame_size if frame_size is not None else np.asarray(depth_m).shape[::-1]))
    return {"file": str(getattr(path, "name", path)), "n_object_points": int(len(obj)), "frame": POINTCLOUD_FRAME}

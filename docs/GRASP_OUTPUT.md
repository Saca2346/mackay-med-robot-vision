# Grasp output for the arm team

On every **PICK** (the box is found and verified by two independent sources), the vision side writes one JSON line to
`grasp.jsonl`, next to `verify_log.csv` (`results\uji_realsense\` for `TEST_REALSENSE.bat` menu **G**). It also prints a
`[grasp]` line in the terminal. The code is in `src/box_grasp.py`, and the depth comes from `src/box_camera.py`.

Run it with `TEST_REALSENSE.bat`, menu **G**, or with
`scripts\test_webcam_box.py --config ... --depth --request "..."`.
Without `--depth` (a webcam or video replay), only the pixel fields are filled and the metric fields are `null`.

## Camera and frames

- Intel RealSense D455. Colour is 1280x800. Depth is z16 1280x720, **aligned to the colour image** (`rs.align`), so a
  colour pixel and the depth pixel at the same position are the same point.
- `xyz_m` is in the **colour camera frame**: x to the right, y down, z forward, in metres. `intrinsics` (fx, fy, cx, cy
  of the colour stream) is written in each line.
- Depth and intrinsics are retained with the exact colour frame handed to the pipeline, including while OCR and
  crop generation run. If that frame has no matching depth, metric output stays `null`; a later depth frame is never
  substituted for it.
- Converting camera coordinates into the robot base frame (hand-eye calibration) is done on the arm side.

## Fields (one line = one PICK)

| field | unit | meaning |
|---|---|---|
| `time`, `request`, `track`, `status` | | when, what was asked, track id of the box, always `PICK` |
| `frame_size`, `intrinsics` | px | colour image size and its intrinsics (`null` without depth) |
| `center_px`, `obb_px` | px | centre and 4 corners of the rotated box of the target spine |
| `angle_deg` | deg | spine long axis vs the image vertical (0 = upright) |
| `spine_width_px` / `spine_length_px` | px | visible spine size |
| `gap_left_px` / `gap_right_px` | px | free space to the nearest neighbour spine (0 = touching, `null` = none detected) |
| `no_grip_px` | px | size / expiry label polygons and the located or decoded barcode region that must stay readable |
| `xyz_m` | m | spine centre in the camera frame, from the median depth of the inner 70 % of the spine |
| `normal` / `approach` | unit vector | spine face normal (plane fit, points to the camera) / approach direction = -normal |
| `spine_width_mm` / `spine_length_mm` | mm | projected visible spine size at the centre depth, using fx and fy along each rotated axis |
| `gap_left_mm` / `gap_right_mm` | mm | projected free space beside the spine at the centre depth, using fx and fy along its short axis |
| `depth_valid` | 0..1 | share of the spine with a valid depth reading; low (glare, too close) = **do not trust xyz** |
| `pointcloud` | | `{file, n_object_points, frame}` of the point-cloud file written next to this line (GraspGen-X hand-off below); `null` when the depth was not usable |

`xyz_m` is `null` when fewer than 20 depth pixels are valid. In that case the robot should not guess the position; it
should move or wait for the next PICK.

Metric dimensions and gaps use the centre-depth projection. They are not corrected for face tilt or a neighbouring
box at a different depth, and do not establish a collision-free gripper path. Physical depth/grasp validation with the
arm team remains open.

## Hand-off to GraspGen-X (Leon's module)

**A bounding box is not the input.** GraspGen and GraspGen-X (NVIDIA, CVPR 2026) take a *segmented partial point cloud of the
object* (or a scene point cloud with one label per object) plus a gripper name or config, and return ranked grasp poses. The
only "bounding boxes" in the GraspGen-X README are the swept-volume boxes of the gripper. So the vision side sends the point
cloud of the verified box; the rotated box (`obb_px`) only selects the depth pixels and is kept as metadata.

Per PICK with valid depth, `grasp.jsonl` gets the usual line and `grasp_<track>_<YYYYmmdd_HHMMSS>.npz` is written beside it
(`save_pointcloud_npz` in `src/box_grasp.py`). The line's `pointcloud` field names the file.

| array in the `.npz` | shape | meaning |
|---|---|---|
| `object_points` | (N, 3) float32, N up to 2048 | partial cloud of the target spine: depth pixels inside the rotated box (shrunk to 95 %), within 8 cm of its median depth, de-projected with the intrinsics. Fewer than 100 valid points = no file. Nothing is duplicated to reach N |
| `scene_points`, `scene_labels` | (M, 3) float32, (M,) int16 | the whole depth image at stride 4, label 0 = background / shelf, 1 = target box, 2.. = the other detected boxes, so the planner can drop grasps that collide with them |
| `intrinsics` | (4,) | fx, fy, cx, cy of the colour stream |
| `obb_px`, `frame_size` | (4, 2), (2,) | rotated box of the target (pixels) and the image size (w, h) |

Units are metres; frame is `camera_color_m` (x right, y down, z forward, D455 colour camera, depth aligned to colour).

```python
import numpy as np
z = np.load("grasp_7_20261023_131530.npz")
object_pc = z["object_points"]                       # (N, 3), metres, camera frame -> GraspGen(-X) object point cloud
scene_pc, labels = z["scene_points"], z["scene_labels"]   # scene demo with per-object segmentation (label 1 = target)
```

On Leon's side (to confirm; vision does not do these): the gripper (`--gripper_name`, e.g. `robotiq_2f_85`, or a URDF run through the
gripper wizard), the camera -> robot base transform (hand-eye calibration) applied to the returned grasp poses, collision-free
planning, and the choice among the ranked grasps. Keep `no_grip_px` (label and barcode zones) out of the finger contact if the
gripper allows it.

Limits and open points:

- The README of GraspGen does not state the coordinate frame, the units or the pose format of its API; metres and 4 x 4
  homogeneous poses in the frame of the input cloud are the usual reading and must be confirmed by running it once.
- The cloud is the visible spine face only (a partial view). The mask is the detected rotated box, not an instance mask; at
  the box ends and between touching boxes a few neighbour pixels can remain inside it.
- Replays and the webcam have no depth: no `.npz`, `pointcloud` is `null`, and nothing can be planned.
- **Measured 23-10 on the D455 (shelf of 11 boxes, about 0.9 m, window blinds behind):** only about 5 % of the pixels inside a
  spine have a depth reading (median over the 11 boxes; scene 4-16 % depending on the preset), with the device preset as found
  (Custom), Default, High Density and Medium Density alike; merging 90 static frames adds nothing (the same pixels stay invalid;
  the valid ones repeat to 0.4 mm), and hole filling only interpolates. The example file therefore holds 297 points, not 2048. The
  depth of plain white / teal spines is the weak point, not the point-cloud code: a closer camera (the Gemini 305 works at 4-100 cm),
  less ambient IR or another depth source is needed before GraspGen-X can be fed reliably. Scripts and the valid-pixel picture:
  `results/weekly_20261023/grasp_handoff`.
- **Depth exposure sweep (same shelf):** manual depth exposure helps only a little: median valid share of a box in one frame 6 % (auto)
  -> 9 % at 500 us, 8 % at 1-2 ms, 7 % at 4 ms (gain 16); the share of a box valid in at least one of 10 frames 12 % -> 33 % at 500 us.
  Still far from dense. The D455 then dropped off the USB bus ("device not functioning"; status Unknown) during the sweep, so the larger
  exposures, gain 64 and `emitter_always_on` are untested. Next physical steps: replug on a USB 3 port, close the window blinds (ambient IR),
  repeat the sweep once, and compare with the Gemini 305 at 10-30 cm. Not enabled in `src/box_camera.py`.
- Nothing here is validated on the arm (no measured 3D error, no hand-eye calibration, no physical grasp).
- A grasp is only ever requested for a PICK (two independent sources). The point cloud never changes the identity decision.

## Example (video replay, so no depth)

```json
{"time": "2026-10-15T07:56:40", "request": "accuforce 2.5 x 12 mm", "track": 1, "status": "PICK",
 "frame_size": [1280, 800], "intrinsics": null, "center_px": [789.7, 405.9], "angle_deg": -2.3,
 "spine_width_px": 47.6, "spine_length_px": 670.2, "gap_left_px": 1.7, "gap_right_px": 0.0,
 "no_grip_px": [[[754.8, 735.8], ...]], "xyz_m": null, "normal": null, "spine_width_mm": null, "depth_valid": 0.0}
```

## Open questions for the arm team

1. Should the target be the camera frame (current) or the robot base frame? If it is the base frame, who provides the
   hand-eye calibration?
2. Transport: a file (`grasp.jsonl`, current), a ROS 2 topic, or a socket?
3. Gripper: maximum opening and finger thickness, so that a PICK can be flagged when `gap_*_mm` is too small.
4. Grip point: the spine centre (current), or a point outside the `no_grip_px` labels?

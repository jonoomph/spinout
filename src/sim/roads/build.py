"""Helpers to convert a road plan into geometry and terrain updates."""


from __future__ import annotations
import math
from typing import List, Tuple

import numpy as np
from noise import pnoise2
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.interpolate import RectBivariateSpline

from . import plan as cfg

_vec2 = np.ndarray
def _tangent(path: List[_vec2], i: int) -> _vec2:
    if i == 0:
        v = path[1] - path[0]
    elif i == len(path) - 1:
        v = path[-1] - path[-2]
    else:
        v = path[i + 1] - path[i - 1]
    n = float(np.linalg.norm(v))
    return (v / n) if n > 0 else np.array([0.0, 1.0])


def _get_height(
    heights: np.ndarray,
    cell_x: float,
    cell_z: float,
    width: float,
    height: float,
    x: float,
    z: float,
) -> float:
    if not (0 <= x <= width and 0 <= z <= height):
        return 0.0
    res_x, res_z = heights.shape
    ix = min(max(0, int(x / cell_x)), res_x - 2)
    iz = min(max(0, int(z / cell_z)), res_z - 2)
    fx = (x - ix * cell_x) / cell_x
    fz = (z - iz * cell_z) / cell_z
    h00 = heights[ix, iz]
    h10 = heights[ix + 1, iz]
    h01 = heights[ix, iz + 1]
    h11 = heights[ix + 1, iz + 1]
    return (
        (1 - fx) * (1 - fz) * h00
        + fx * (1 - fz) * h10
        + (1 - fx) * fz * h01
        + fx * fz * h11
    )


# ---------------------------------------------------------------------------
# Stamping into the heightmap (deck + *real* ditch)


def _stamp_road(
    terrain,
    path: List[_vec2],
    lane_width: float,
    lanes: int,
    shoulder: float,
    road_height: float,
    cross_pitch_rad: float,
    ditch_width: float,
    ditch_depth: float,
    bump_amp: float,
    hole_amp: float,
    noise_f: float,
    rng: np.random.Generator,
    road_friction: float,
) -> None:
    half_road = 0.5 * lane_width * lanes
    half_plus_sh = half_road + shoulder
    total_half = half_plus_sh + ditch_width

    cell_x = terrain.cell_size_x
    cell_z = terrain.cell_size_z
    width = terrain.width
    height = terrain.height
    res_x, res_z = terrain.heights.shape
    noise_dx = rng.uniform(0, 1000)
    noise_dz = rng.uniform(0, 1000)

    heights_orig = terrain.heights.copy()

    # Smooth base under centerline, higher sigma for more lanes
    long_smooth_sigma = np.interp(lanes, [cfg.LANE_COUNT_MIN, cfg.LANE_COUNT_MAX], [cfg.LONG_SMOOTH_SIGMA_MIN, cfg.LONG_SMOOTH_SIGMA_MAX])
    base_under = np.array(
        [
            _get_height(
                heights_orig, cell_x, cell_z, width, height, p[0], p[1]
            )
            for p in path
        ]
    )
    base_under_sm = gaussian_filter1d(base_under, long_smooth_sigma)

    for i, c in enumerate(path):
        tdir = _tangent(path, i)
        nrm = np.array([-tdir[1], tdir[0]], dtype=float)

        # detect uphill/downhill side by sampling original terrain
        sample_off = half_plus_sh + 0.3 * ditch_width
        left_x = c[0] - nrm[0] * sample_off
        left_z = c[1] - nrm[1] * sample_off
        right_x = c[0] + nrm[0] * sample_off
        right_z = c[1] + nrm[1] * sample_off
        h_left = _get_height(
            heights_orig, cell_x, cell_z, width, height, left_x, left_z
        )
        h_right = _get_height(
            heights_orig, cell_x, cell_z, width, height, right_x, right_z
        )
        # lower side gets deeper ditch
        depth_left = ditch_depth * (1.3 if h_left < h_right else 0.7)
        depth_right = ditch_depth * (1.3 if h_right < h_left else 0.7)

        # local base
        base = base_under_sm[i] + road_height

        # sweep across
        across = np.arange(-total_half, total_half + cfg.ACROSS_STEP * 0.5, cfg.ACROSS_STEP)
        for d in across:
            x = c[0] + nrm[0] * d
            z = c[1] + nrm[1] * d
            if not (0 <= x <= width and 0 <= z <= height):
                continue
            ix = int(x / cell_x)
            iz = int(z / cell_z)
            if ix < 0 or ix >= res_x or iz < 0 or iz >= res_z:
                continue
            dist = abs(d)
            orig = _get_height(
                heights_orig, cell_x, cell_z, width, height, x, z
            )
            if dist <= half_plus_sh:
                ht = base - math.tan(cross_pitch_rad) * dist
                n = pnoise2(x * noise_f + noise_dx, z * noise_f + noise_dz)
                ht += (n * bump_amp) if n >= 0 else (n * hole_amp)
                terrain.heights[ix, iz] = ht
                # mark road friction separately so terrain friction doesn't bleed through
                terrain.road_friction[ix, iz] = road_friction
            else:
                t = (dist - half_plus_sh) / max(ditch_width, 1e-6)
                t = min(max(t, 0.0), 1.0)
                # cosine bowl (smooth in/out)
                bowl = 0.5 * (1.0 - math.cos(math.pi * t))  # 0..1
                side_depth = depth_right if d > 0 else depth_left
                edge = base - math.tan(cross_pitch_rad) * half_plus_sh
                ht = (1 - t) * edge + t * orig
                ht -= side_depth * bowl
                terrain.heights[ix, iz] = min(orig, ht)

    # Corridor-only smoothing to tidy edges
    mask = np.zeros_like(terrain.heights, dtype=float)
    min_cell = min(cell_x, cell_z)
    pad = int((total_half + 2.0) / min_cell) + 2
    for p in path:
        cx = int(p[0] / cell_x)
        cz = int(p[1] / cell_z)
        x0 = max(0, cx - pad); x1 = min(res_x, cx + pad + 1)
        z0 = max(0, cz - pad); z1 = min(res_z, cz + pad + 1)
        mask[x0:x1, z0:z1] = 1.0
    smoothed = gaussian_filter(terrain.heights, sigma=5.0, mode="nearest")
    terrain.heights = smoothed * mask + terrain.heights * (1.0 - mask)


# ---------------------------------------------------------------------------
# Public API


def build_road_vertices(
    terrain,
    path: List[Tuple[float, float]],
    lane_width: float,
    lanes: int,
    shoulder: float,
    road_height: float,
    cross_pitch: float,
    ditch_width: float | None = None,
    road_color=cfg.ROAD_COL,
    drive_line: List[Tuple[float, float]] | None = None,
    **_: dict,
) -> np.ndarray:
    if ditch_width is None:
        ditch_width = cfg.DITCH_WIDTH_MAX

    # helper to sample terrain height
    def base_height(x: float, z: float) -> float:
        return float(terrain.get_height(x, z))

    cell = min(terrain.cell_size_x, terrain.cell_size_z)
    along_step = max(cell * cfg.ALONG_STEP_FACTOR, 0.35)
    half_road = 0.5 * lane_width * lanes
    half_plus_sh = half_road + shoulder

    # offsets for road deck
    gray_offsets = np.linspace(-half_plus_sh, half_plus_sh, max(6, lanes * 3 + 3))

    # prepare lane-marking definitions
    lines_list: list[tuple[float, list[float], bool]] = []
    left_edge_off = -half_plus_sh
    right_edge_off = half_plus_sh
    left_edge_col = cfg.WHITE_COL
    right_edge_col = cfg.YELLOW_COL if lanes == 1 else cfg.WHITE_COL
    lines_list.append((left_edge_off, left_edge_col, False))
    for k in range(1, lanes):
        off = -half_road + k * lane_width
        is_yellow = False
        if lanes % 2 == 0:
            if abs(off) < 1e-3:
                is_yellow = True
        else:
            mid_left = -half_road + (lanes // 2) * lane_width
            mid_right = mid_left + lane_width
            if abs(off - mid_left) < 1e-3 or abs(off - mid_right) < 1e-3:
                is_yellow = True
        col = cfg.YELLOW_COL if is_yellow else cfg.WHITE_COL
        dotted = not is_yellow
        lines_list.append((off, col, dotted))
    lines_list.append((right_edge_off, right_edge_col, False))

    # dash and line width
    period = cfg.DASH_LENGTH + cfg.GAP_LENGTH
    line_half = cfg.LINE_WIDTH * 0.5

    # convert path and s-path
    path_np = [np.array(p, float) for p in path]
    s_path = [0.0]
    for p0, p1 in zip(path_np[:-1], path_np[1:]):
        s_path.append(s_path[-1] + np.linalg.norm(p1 - p0))

    # sample centerline into (pos, normal, s)
    samples: list[tuple[np.ndarray, np.ndarray, float]] = []
    for ii, (p0, p1) in enumerate(zip(path_np[:-1], path_np[1:])):
        seg = p1 - p0
        L = np.linalg.norm(seg)
        if L < 1e-6:
            continue
        dir2 = seg / L
        nrm2 = np.array([-dir2[1], dir2[0]], dtype=float)
        nsteps = max(2, int(L / along_step))
        for k in range(nsteps):
            frac = k / nsteps
            c = p0 + dir2 * (L * frac)
            s_val = s_path[ii] + L * frac
            samples.append((c, nrm2, s_val))
    # include final point
    if len(path_np) >= 2:
        last = path_np[-1]
        prev = path_np[-2]
        dir_last = last - prev
        if np.linalg.norm(dir_last) > 1e-6:
            dir_last /= np.linalg.norm(dir_last)
        nrm_last = np.array([-dir_last[1], dir_last[0]], dtype=float)
        samples.append((last, nrm_last, s_path[-1]))

    # build deck rings
    rings: list[list[list[float]]] = []
    for (c, nrm2, _) in samples:
        ring = []
        for off in gray_offsets:
            x = c[0] + nrm2[0] * off
            z = c[1] + nrm2[1] * off
            y = base_height(x, z) + cfg.ROAD_EPS
            ring.append([x, y, z])
        rings.append(ring)

    verts: list[float] = []
    def emit_quad(a, b, c, d, col):
        col_list = list(col)
        verts.extend(a + col_list); verts.extend(b + col_list); verts.extend(c + col_list)
        verts.extend(c + col_list); verts.extend(b + col_list); verts.extend(d + col_list)

    # connect deck
    for i in range(len(rings) - 1):
        r0, r1 = rings[i], rings[i + 1]
        for j in range(len(r0) - 1):
            emit_quad(r0[j], r0[j + 1], r1[j], r1[j + 1], road_color)

    # lane markings extrusion
    for off, col, dotted in lines_list:
        prev_pair = None
        prev_emit = False
        for (c, nrm2, s_val) in samples:
            emit = True
            if dotted:
                emit = (s_val % period) < cfg.DASH_LENGTH
            left_off = off - line_half
            right_off = off + line_half
            a = np.array([
                c[0] + nrm2[0] * left_off,
                base_height(c[0] + nrm2[0] * left_off, c[1] + nrm2[1] * left_off) + cfg.LINE_EPS,
                c[1] + nrm2[1] * left_off
            ]).tolist()
            b = np.array([
                c[0] + nrm2[0] * right_off,
                base_height(c[0] + nrm2[0] * right_off, c[1] + nrm2[1] * right_off) + cfg.LINE_EPS,
                c[1] + nrm2[1] * right_off
            ]).tolist()
            this_pair = [a, b]
            if prev_pair is not None and emit and prev_emit:
                emit_quad(prev_pair[0], prev_pair[1], this_pair[0], this_pair[1], col)
            prev_pair = this_pair
            prev_emit = emit

    # green driveline extrusion
    if drive_line:
        half_dl = cfg.DRIVE_LINE_WIDTH * 0.5
        green = [0.0, 1.0, 0.0, 1.0]
        prev_pair = None
        for p0, p1 in zip(drive_line[:-1], drive_line[1:]):
            a0 = np.array([p0[0], base_height(p0[0], p0[1]) + cfg.DRIVE_LINE_HEIGHT, p0[1]])
            a1 = np.array([p1[0], base_height(p1[0], p1[1]) + cfg.DRIVE_LINE_HEIGHT, p1[1]])
            if not (np.isfinite(a0).all() and np.isfinite(a1).all()):
                continue
            seg = a1 - a0
            dir2 = np.array([seg[0], seg[2]], dtype=float)
            L = np.linalg.norm(dir2)
            if L < 1e-6:
                continue
            dir2 /= L
            nrm2 = np.array([-dir2[1], dir2[0]], dtype=float)
            left = (a0 + np.array([nrm2[0]*half_dl, 0, nrm2[1]*half_dl])).tolist()
            right= (a0 - np.array([nrm2[0]*half_dl, 0, nrm2[1]*half_dl])).tolist()
            this_pair = [left, right]
            if prev_pair is not None:
                emit_quad(prev_pair[0], prev_pair[1], this_pair[0], this_pair[1], green)
            prev_pair = this_pair

    return np.array(verts, dtype="f4")


def build_speed_sign_vertices(
    terrain,
    path: List[Tuple[float, float]],
    lane_width: float,
    lanes: int,
    shoulder: float,
    speed_limits: List[dict] | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Return geometry for sign posts and textured sign billboards.

    The function returns a tuple ``(posts, billboards)`` where ``posts`` is a
    NumPy array of colored vertices for the vertical sign posts and
    ``billboards`` is a list of dictionaries with keys ``"speed"`` and
    ``"verts"`` describing the textured quads that display the generated sign
    images.
    """
    if not speed_limits:
        return np.zeros((0, 7), dtype="f4"), []

    path_np = [np.array(p, float) for p in path]
    half_road = 0.5 * lane_width * lanes
    half_plus_sh = half_road + shoulder

    post_verts: list[float] = []
    billboards: list[dict] = []

    def emit_rect(center, right, up, width, height, color):
        hw = width * 0.5
        hh = height * 0.5
        a = (center - right * hw - up * hh).tolist()
        b = (center + right * hw - up * hh).tolist()
        c = (center - right * hw + up * hh).tolist()
        d = (center + right * hw + up * hh).tolist()
        post_verts.extend(a + color)
        post_verts.extend(b + color)
        post_verts.extend(c + color)
        post_verts.extend(c + color)
        post_verts.extend(b + color)
        post_verts.extend(d + color)

    gray = [0.6, 0.6, 0.6, 1.0]

    for seg in speed_limits:
        idx = seg.get("sign_idx")
        if idx is None or idx < 0 or idx >= len(path_np):
            continue
        c = path_np[idx]
        if idx < len(path_np) - 1:
            tvec = path_np[idx + 1] - path_np[idx]
        else:
            tvec = path_np[idx] - path_np[idx - 1]
        L = float(np.linalg.norm(tvec))
        if L <= 1e-6:
            continue

        dir2 = tvec / L
        forward = np.array([-dir2[0], 0.0, -dir2[1]])
        right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
        right /= np.linalg.norm(right) + 1e-8
        up = np.array([0.0, 1.0, 0.0])

        sign_off = half_plus_sh + 2.0
        sx = c[0] + right[0] * sign_off
        sz = c[1] + right[2] * sign_off
        ground_y = terrain.get_height(sx, sz)
        sy = ground_y + 2.0
        center = np.array([sx, sy, sz])

        # Sign dimensions (in meters)
        sign_w = 1.2
        sign_h = 1.6

        # Post behind the sign
        post_width = 0.1
        post_height = (sy - sign_h * 0.5) - ground_y
        post_center = np.array([sx, ground_y + post_height * 0.5, sz]) - forward * 0.02
        emit_rect(post_center, right, up, post_width, post_height, gray)

        # Textured quad for the sign face
        hw = sign_w * 0.5
        hh = sign_h * 0.5
        a = (center - right * hw - up * hh).tolist()
        b = (center + right * hw - up * hh).tolist()
        c2 = (center - right * hw + up * hh).tolist()
        d = (center + right * hw + up * hh).tolist()
        billboard = {
            "speed": seg["speed_mph"],
            "verts": np.array(
                [
                    *a, 0.0, 1.0,
                    *b, 1.0, 1.0,
                    *c2, 0.0, 0.0,
                    *b, 1.0, 1.0,
                    *d, 1.0, 0.0,
                    *c2, 0.0, 0.0,
                ],
                dtype="f4",
            ),
        }
        billboards.append(billboard)

    return np.array(post_verts, dtype="f4"), billboards


def apply_plan(terrain, path: List[Tuple[float, float]], params: dict, rng: np.random.Generator | None = None) -> None:
    """Stamp the planned road into *terrain* using the stored parameters."""
    if rng is None:
        rng = np.random.default_rng()
    if cfg.UPSAMPLE_FACTOR > 1:
        old_res_x = terrain.res_x
        old_res_z = terrain.res_z
        old_width = terrain.width
        old_height = terrain.height
        new_res_x = (old_res_x - 1) * cfg.UPSAMPLE_FACTOR + 1
        new_res_z = (old_res_z - 1) * cfg.UPSAMPLE_FACTOR + 1
        new_cell_x = old_width / (new_res_x - 1)
        new_cell_z = old_height / (new_res_z - 1)
        x_old = np.linspace(0, old_width, old_res_x)
        z_old = np.linspace(0, old_height, old_res_z)
        x_new = np.linspace(0, old_width, new_res_x)
        z_new = np.linspace(0, old_height, new_res_z)
        spline = RectBivariateSpline(x_old, z_old, terrain.heights, kx=3, ky=3)
        terrain.heights = spline(x_new, z_new)
        spline_fric = RectBivariateSpline(x_old, z_old, terrain.surface_friction, kx=1, ky=1)
        terrain.surface_friction = spline_fric(x_new, z_new)
        spline_road = RectBivariateSpline(x_old, z_old, terrain.road_friction, kx=1, ky=1)
        terrain.road_friction = spline_road(x_new, z_new)
        terrain.res_x = new_res_x
        terrain.res_z = new_res_z
        terrain.cell_size_x = new_cell_x
        terrain.cell_size_z = new_cell_z
    _stamp_road(
        terrain,
        [np.array(p) for p in path],
        params["lane_width"],
        params["lanes"],
        params["shoulder"],
        params["road_height"],
        params["cross_pitch"],
        params["ditch_width"],
        params["ditch_depth"],
        params["bump_amp"],
        params["hole_amp"],
        params["noise_f"],
        rng,
        params.get("road_friction", 1.0),
    )

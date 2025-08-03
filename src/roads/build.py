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


def _get_height(heights: np.ndarray, res: int, cell: float, size: float, x: float, z: float) -> float:
    if not (0 <= x <= size and 0 <= z <= size):
        return 0.0
    ix = min(max(0, int(x / cell)), res - 2)
    iz = min(max(0, int(z / cell)), res - 2)
    fx = (x - ix * cell) / cell
    fz = (z - iz * cell) / cell
    h00 = heights[ix, iz]
    h10 = heights[ix + 1, iz]
    h01 = heights[ix, iz + 1]
    h11 = heights[ix + 1, iz + 1]
    return (1 - fx) * (1 - fz) * h00 + fx * (1 - fz) * h10 + (1 - fx) * fz * h01 + fx * fz * h11


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

    cell = terrain.cell_size
    size = terrain.size
    res = terrain.res
    noise_dx = rng.uniform(0, 1000)
    noise_dz = rng.uniform(0, 1000)

    heights_orig = terrain.heights.copy()

    # Smooth base under centerline, higher sigma for more lanes
    long_smooth_sigma = np.interp(lanes, [cfg.LANE_COUNT_MIN, cfg.LANE_COUNT_MAX], [cfg.LONG_SMOOTH_SIGMA_MIN, cfg.LONG_SMOOTH_SIGMA_MAX])
    base_under = np.array([_get_height(heights_orig, res, cell, size, p[0], p[1]) for p in path])
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
        h_left = _get_height(heights_orig, res, cell, size, left_x, left_z)
        h_right = _get_height(heights_orig, res, cell, size, right_x, right_z)
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
            if not (0 <= x <= size and 0 <= z <= size):
                continue
            ix = int(x / cell); iz = int(z / cell)
            if ix < 0 or ix >= res or iz < 0 or iz >= res:
                continue
            dist = abs(d)
            orig = _get_height(heights_orig, res, cell, size, x, z)
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
    pad = int((total_half + 2.0) / cell) + 2
    for p in path:
        cx = int(p[0] / cell)
        cz = int(p[1] / cell)
        x0 = max(0, cx - pad); x1 = min(res, cx + pad + 1)
        z0 = max(0, cz - pad); z1 = min(res, cz + pad + 1)
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
    skirt_color=cfg.SKIRT_COL,
    **_: dict,
) -> np.ndarray:
    if ditch_width is None:
        ditch_width = cfg.DITCH_WIDTH_MAX

    def base_height(x: float, z: float) -> float:
        return float(terrain.get_height(x, z))

    cell = terrain.cell_size
    along_step = max(cell * cfg.ALONG_STEP_FACTOR, 0.35)

    half_road = 0.5 * lane_width * lanes
    half_plus_sh = half_road + shoulder

    gray_offsets = np.linspace(-half_plus_sh, half_plus_sh, max(6, lanes * 3 + 3))
    skirt_offsets = [half_plus_sh + (j + 1) * (ditch_width / cfg.SKIRT_RINGS) for j in range(cfg.SKIRT_RINGS)]

    # Compute cumulative distance along path
    s = [0.0]
    for p_prev, p_curr in zip(path[:-1], path[1:]):
        dist = math.sqrt((p_curr[0] - p_prev[0])**2 + (p_curr[1] - p_prev[1])**2)
        s.append(s[-1] + dist)

    # Define lane marking lines: (offset, color, dotted)
    lines_list = []
    left_edge_off = -half_plus_sh
    right_edge_off = half_plus_sh
    left_edge_col = cfg.YELLOW_COL if lanes == 1 else cfg.WHITE_COL
    lines_list.append((left_edge_off, left_edge_col, False))
    for k in range(1, lanes):
        off = -half_road + k * lane_width
        is_yellow = False
        if lanes % 2 == 0:
            if abs(off) < 1e-3:  # center
                is_yellow = True
        else:
            if lanes > 1:
                mid_left = -half_road + (lanes // 2) * lane_width
                mid_right = mid_left + lane_width
                if abs(off - mid_left) < 1e-3 or abs(off - mid_right) < 1e-3:
                    is_yellow = True
        col = cfg.YELLOW_COL if is_yellow else cfg.WHITE_COL
        dotted = not is_yellow
        lines_list.append((off, col, dotted))
    lines_list.append((right_edge_off, cfg.WHITE_COL, False))

    verts: list[float] = []

    def emit_quad(a, b, c, d, col):
        verts.extend(a + col); verts.extend(b + col); verts.extend(c + col)
        verts.extend(c + col); verts.extend(b + col); verts.extend(d + col)

    path_np = [np.array(p, float) for p in path]
    period = cfg.DASH_LENGTH + cfg.GAP_LENGTH
    line_half = cfg.LINE_WIDTH / 2.0

    for ii, (p0, p1) in enumerate(zip(path_np[:-1], path_np[1:])):
        seg = p1 - p0
        L = float(np.linalg.norm(seg))
        if L <= 1e-6:
            continue
        dir2 = seg / L
        nrm2 = np.array([-dir2[1], dir2[0]], dtype=float)

        nsteps = max(2, int(L / along_step))
        for k in range(nsteps):
            frac0 = k / nsteps
            frac1 = (k + 1) / nsteps
            c0 = p0 + dir2 * (L * frac0)
            c1 = p0 + dir2 * (L * frac1)

            deck0 = []
            deck1 = []
            for off in gray_offsets:
                x0 = c0[0] + nrm2[0] * off
                z0 = c0[1] + nrm2[1] * off
                y0 = base_height(x0, z0) + cfg.ROAD_EPS
                deck0.append([x0, y0, z0])

                x1 = c1[0] + nrm2[0] * off
                z1 = c1[1] + nrm2[1] * off
                y1 = base_height(x1, z1) + cfg.ROAD_EPS
                deck1.append([x1, y1, z1])
            for j in range(len(gray_offsets) - 1):
                emit_quad(deck0[j], deck0[j + 1], deck1[j], deck1[j + 1], road_color)

            # skirts on each side
            for side in (-1.0, +1.0):
                edge_idx = -1 if side > 0 else 0
                ring_prev0 = deck0[edge_idx]
                ring_prev1 = deck1[edge_idx]
                edge_y0 = ring_prev0[1] - cfg.ROAD_EPS  # subtract eps to compute blend, then add back
                edge_y1 = ring_prev1[1] - cfg.ROAD_EPS
                for j in range(cfg.SKIRT_RINGS):
                    off = skirt_offsets[j]
                    d = off * side
                    x0 = c0[0] + nrm2[0] * d; z0 = c0[1] + nrm2[1] * d
                    x1 = c1[0] + nrm2[0] * d; z1 = c1[1] + nrm2[1] * d
                    t = (off - half_plus_sh) / max(ditch_width, 1e-6)
                    t = min(max(t, 0.0), 1.0)
                    y0 = (1 - t) * edge_y0 + t * float(base_height(x0, z0)) + cfg.ROAD_EPS
                    y1 = (1 - t) * edge_y1 + t * float(base_height(x1, z1)) + cfg.ROAD_EPS
                    ring0 = [x0, y0, z0]
                    ring1 = [x1, y1, z1]
                    emit_quad(ring_prev0, ring0, ring_prev1, ring1, skirt_color)
                    ring_prev0, ring_prev1 = ring0, ring1

            # lane markings (per sub-segment)
            sub_s0 = s[ii] + frac0 * L
            sub_s1 = s[ii] + frac1 * L
            mid_sub_s = (sub_s0 + sub_s1) / 2.0
            for off, col, dotted in lines_list:
                should_emit = True
                if dotted:
                    mod = mid_sub_s % period
                    should_emit = mod < cfg.DASH_LENGTH
                if should_emit:
                    left_off = off - line_half
                    right_off = off + line_half

                    xl0 = c0[0] + nrm2[0] * left_off
                    zl0 = c0[1] + nrm2[1] * left_off
                    yl0 = base_height(xl0, zl0) + cfg.ROAD_EPS
                    vert_l0 = [xl0, yl0, zl0]

                    xr0 = c0[0] + nrm2[0] * right_off
                    zr0 = c0[1] + nrm2[1] * right_off
                    yr0 = base_height(xr0, zr0) + cfg.ROAD_EPS
                    vert_r0 = [xr0, yr0, zr0]

                    xl1 = c1[0] + nrm2[0] * left_off
                    zl1 = c1[1] + nrm2[1] * left_off
                    yl1 = base_height(xl1, zl1) + cfg.ROAD_EPS
                    vert_l1 = [xl1, yl1, zl1]

                    xr1 = c1[0] + nrm2[0] * right_off
                    zr1 = c1[1] + nrm2[1] * right_off
                    yr1 = base_height(xr1, zr1) + cfg.ROAD_EPS
                    vert_r1 = [xr1, yr1, zr1]

                    emit_quad(vert_l0, vert_r0, vert_l1, vert_r1, col)

    return np.array(verts, dtype="f4")


# ---------------------------------------------------------------------------
# Mini-map generation


def generate_mini_map(terrain, path: List[Tuple[float, float]], params: dict, car_pos: np.ndarray) -> np.ndarray:
    """Generate a 250x250 RGB mini-map image using pure NumPy."""
    size = terrain.size
    mini_size = 250
    x_mini = np.linspace(0, size, mini_size)
    xx, zz = np.meshgrid(x_mini, x_mini)

    # Terrain heights downsampled
    x_orig = np.linspace(0, size, terrain.res)
    spline = RectBivariateSpline(x_orig, x_orig, terrain.heights, kx=1, ky=1)
    heights_low = spline(xx, zz)

    h_min = np.min(heights_low)
    h_max = np.max(heights_low)
    if h_max > h_min:
        green = 50 + 150 * (heights_low - h_min) / (h_max - h_min)
    else:
        green = np.full_like(heights_low, 100)
    img = np.zeros((mini_size, mini_size, 3), dtype=np.uint8)
    img[:, :, 1] = green.astype(np.uint8)

    # Road: gray where distance to path <= half road width
    full_width = params['lanes'] * params['lane_width'] + 2 * params['shoulder']
    half_width = full_width / 2.0
    path_arr = np.array(path)

    min_d = np.full((mini_size, mini_size), np.inf)
    for k in range(len(path_arr) - 1):
        ax, az = path_arr[k]
        bx, bz = path_arr[k + 1]
        dx = bx - ax
        dz = bz - az
        len2 = dx**2 + dz**2
        if len2 < 1e-6:
            continue
        px = xx - ax
        pz = zz - az
        t = (px * dx + pz * dz) / len2
        t = np.clip(t, 0, 1)
        closest_x = ax + t * dx
        closest_z = az + t * dz
        dist = np.sqrt((xx - closest_x)**2 + (zz - closest_z)**2)
        min_d = np.minimum(min_d, dist)

    road_mask = min_d <= half_width
    img[road_mask] = [100, 100, 100]  # gray

    # Car: yellow circle
    car_x, car_z = car_pos[0], car_pos[2]
    ix = int((car_x / size) * (mini_size - 1))
    iz = int((car_z / size) * (mini_size - 1))
    radius = 2  # pixels
    for di in range(-radius, radius + 1):
        for dz in range(-radius, radius + 1):
            if di**2 + dz**2 <= radius**2:
                cx = ix + di
                cz = iz + dz
                if 0 <= cx < mini_size and 0 <= cz < mini_size:
                    img[cz, cx] = [255, 255, 0]  # yellow

    return img

def apply_plan(terrain, path: List[Tuple[float, float]], params: dict, rng: np.random.Generator | None = None) -> None:
    """Stamp the planned road into *terrain* using the stored parameters."""
    if rng is None:
        rng = np.random.default_rng()
    if cfg.UPSAMPLE_FACTOR > 1:
        old_res = terrain.res
        old_cell = terrain.cell_size
        old_size = terrain.size
        new_res = (old_res - 1) * cfg.UPSAMPLE_FACTOR + 1
        new_cell = old_size / (new_res - 1)
        x_old = np.linspace(0, old_size, old_res)
        z_old = x_old
        x_new = np.linspace(0, old_size, new_res)
        z_new = x_new
        spline = RectBivariateSpline(x_old, z_old, terrain.heights.T, kx=3, ky=3)
        terrain.heights = spline(x_new, z_new).T
        spline_fric = RectBivariateSpline(x_old, z_old, terrain.surface_friction.T, kx=1, ky=1)
        terrain.surface_friction = spline_fric(x_new, z_new).T
        spline_road = RectBivariateSpline(x_old, z_old, terrain.road_friction.T, kx=1, ky=1)
        terrain.road_friction = spline_road(x_new, z_new).T
        terrain.res = new_res
        terrain.cell_size = new_cell
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

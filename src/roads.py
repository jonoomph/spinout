# roads.py
"""Generate realistic road geometry over a heightmap terrain.

Key additions in this revision
------------------------------
- **Bottom→Top routing**: the road starts near the *bottom edge* and snakes
  generally upward, exiting near the top.
- **Piecewise road grammar**: straight, arc (C), S-curve, and hard 90° turns
  chosen under lane-aware curvature limits (wider roads curve less).
- **Physics ditch**: ditches are cut into the heightmap (not only visual).
  The downhill side digs deeper; the uphill side is shallower. Profile is
  smooth and clamps to only *lower* the terrain.
- **Car placement helper**: see note at bottom for orienting the car to the
  first segment direction (yaw quaternion snippet for `game.py`).

The rest of the module still:
- stamps the road deck into the heightmap (collisions work),
- emits a dense gray deck plus multiple green skirt rings that blend back to
  terrain, and
- keeps all tunables at the top.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
from noise import pnoise2
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import CubicSpline
from src.physics import Quaternion

# ---------------- Tunables / randomness (all in one place) -----------------
# Lanes & shoulders
LANE_WIDTH_MIN = 2.8  # m
LANE_WIDTH_MAX = 3.6
LANE_COUNT_MIN = 1
LANE_COUNT_MAX = 6
SHOULDER_MIN = 0.0    # m per side
SHOULDER_MAX = 1.8

# Cross slope (runoff camber)
CROSS_PITCH_MIN_DEG = 1.0
CROSS_PITCH_MAX_DEG = 4.0

# Embankment/ditch blending width beyond shoulder
DITCH_WIDTH_MIN = 1.5  # m
DITCH_WIDTH_MAX = 4.0
DITCH_DEPTH_MIN = 0.25  # m (extra cut below original terrain)
DITCH_DEPTH_MAX = 1.25

# Road crown above terrain (prevents z-fighting)
ROAD_HEIGHT_MIN = 0.6  # m
ROAD_HEIGHT_MAX = 1.8

# Centerline generation (bottom → top bias)
CTRL_SEG_LEN = 40.0          # meters (coarse move when laying pieces)
RESAMPLE_STEP = 1.0          # dense resampling step for stamping
MAX_CTRL_POINTS = 15         # max number of control points (fewer for more lanes)
MIN_CTRL_POINTS = 5
CTRL_DEVIATION_SCALE = 0.15   # deviation factor, smaller for more lanes
SELF_HIT_MARGIN_FACTOR = 1.5 # margin as factor of full road width
BOTTOM_MARGIN = 0.0         # start z at bottom
SIDE_MARGIN = 0.10           # keep x inside 10% margins
TOP_EXIT_MARGIN = 0.0       # stop once z > 95% of map size
GEN_MAX_TRIES = 20           # max tries to generate non-overlapping path

# Curvature is lane-width aware. Use min radius per lane count.
# (crude but effective for shaping)
MIN_RADIUS_BY_LANES = {  # meters
    1: 18.0,
    2: 35.0,
    3: 65.0,
    4: 110.0,
    5: 160.0,
    6: 220.0,
}
MIN_RADIUS_SCALE_WITH_LANEWIDTH = 3.2  # scale vs 3.2 m nominal lane

# Sampling density around the road for stamping & rendering
ALONG_STEP_FACTOR = 0.1   # step = cell_size * factor (>= 0.35 m)
ACROSS_STEP = 0.5          # meters between cross samples for stamping
SKIRT_RINGS = 7            # number of green rings to blend back to terrain

# Surface irregularities (pick a personality once per road)
NOISE_FREQ_MIN = 0.01
NOISE_FREQ_MAX = 0.03
BUMP_MAX = 0.05  # m
HOLE_MAX = 0.05  # m depth (negative)

# Spline smoothing for resampling
PATH_SMOOTH_SIGMA_MIN = 5.0
PATH_SMOOTH_SIGMA_MAX = 20.0  # higher for more lanes (in sparse points units, will scale)

# Longitudinal smoothing for road base height
LONG_SMOOTH_SIGMA_MIN = 30.0
LONG_SMOOTH_SIGMA_MAX = 150.0  # higher for smoother longitudinal (in dense points units)

# Heightmap upsampling for higher resolution near roads
UPSAMPLE_FACTOR = 4  # Increase resolution by this factor for better detail

# Car placement
CAR_HEIGHT_ABOVE_ROAD = 1.2  # meters

# ---------------------------------------------------------------------------
# Utilities

_vec2 = np.ndarray


def _seg_intersect(a1: _vec2, a2: _vec2, b1: _vec2, b2: _vec2) -> bool:
    def orient(p, q, r):
        return (q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0])
    o1 = orient(a1, a2, b1)
    o2 = orient(a1, a2, b2)
    o3 = orient(b1, b2, a1)
    o4 = orient(b1, b2, a2)
    return (o1 * o2 < 0) and (o3 * o4 < 0)


def _too_close(p: _vec2, a: _vec2, b: _vec2, margin: float) -> bool:
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-9), 0.0, 1.0)
    closest = a + t * ab
    return float(np.linalg.norm(p - closest)) < margin


def _estimate_min_radius(path: List[_vec2], min_rad_required: float) -> bool:
    if len(path) < 3:
        return True
    xs = np.array([p[0] for p in path])
    zs = np.array([p[1] for p in path])
    dx = np.diff(xs)
    dz = np.diff(zs)
    ddx = np.diff(dx)
    ddz = np.diff(dz)
    kappa = np.abs(dx[:-1] * ddz - dz[:-1] * ddx) / (dx[:-1]**2 + dz[:-1]**2)**1.5 + 1e-6
    max_kappa = np.max(kappa)
    min_rad = 1 / max_kappa if max_kappa > 0 else np.inf
    return min_rad >= min_rad_required


def _generate_bottom_to_top(terrain, rng: np.random.Generator, lanes: int, lane_width: float, shoulder: float) -> List[_vec2]:
    size = terrain.size
    x = rng.uniform(SIDE_MARGIN * size, (1.0 - SIDE_MARGIN) * size)
    z = 0.0

    # Lane-aware parameters
    min_rad = MIN_RADIUS_BY_LANES.get(lanes, 65.0)
    min_rad *= (lane_width / MIN_RADIUS_SCALE_WITH_LANEWIDTH)
    min_rad = max(min_rad, 12.0)

    # Self-hit margin based on road width
    full_width = lanes * lane_width + 2 * shoulder
    self_hit_margin = full_width * SELF_HIT_MARGIN_FACTOR

    # Number of control points: fewer for more lanes
    num_ctrl = int(np.interp(lanes, [LANE_COUNT_MIN, LANE_COUNT_MAX], [MAX_CTRL_POINTS, MIN_CTRL_POINTS]))

    # Deviation scale: smaller for more lanes
    dev_scale = CTRL_DEVIATION_SCALE * (1 - (lanes - 1) / (LANE_COUNT_MAX - 1))

    path = None
    for try_i in range(GEN_MAX_TRIES):
        # Generate random control points
        ctrl_points = [np.array([x, z])]
        prev_x = x
        for i in range(1, num_ctrl + 1):
            z_next = i * size / num_ctrl
            x_dev = rng.uniform(-dev_scale * size, dev_scale * size)
            x_next = np.clip(prev_x + x_dev, SIDE_MARGIN * size, (1 - SIDE_MARGIN) * size)
            ctrl_points.append(np.array([x_next, z_next]))
            prev_x = x_next

        # Spline the control points for smooth path
        xs_ctrl = [p[0] for p in ctrl_points]
        zs_ctrl = [p[1] for p in ctrl_points]
        cs = CubicSpline(zs_ctrl, xs_ctrl, bc_type='natural')

        # Resample densely along z
        z_sample = np.linspace(0, size, int(size / RESAMPLE_STEP) + 1)
        x_sample = cs(z_sample)
        candidate_path = [np.array([x, z]) for x, z in zip(x_sample, z_sample)]

        # Smooth the x (horizontal) for better curves
        density_factor = CTRL_SEG_LEN / RESAMPLE_STEP
        path_smooth_sigma_dense = PATH_SMOOTH_SIGMA_MIN + (PATH_SMOOTH_SIGMA_MAX - PATH_SMOOTH_SIGMA_MIN) * ((lanes - LANE_COUNT_MIN) / (LANE_COUNT_MAX - LANE_COUNT_MIN))
        path_smooth_sigma_dense *= density_factor
        xs_smooth = gaussian_filter1d(x_sample, path_smooth_sigma_dense)
        candidate_path = [np.array([x, z]) for x, z in zip(xs_smooth, z_sample)]

        # Check for self-overlap on resampled path (check every 10th to speed up)
        check_path = candidate_path[::10]
        ok = True
        for i in range(len(check_path) - 1):
            for j in range(i + 2, len(check_path) - 1):
                if _seg_intersect(check_path[i], check_path[i+1], check_path[j], check_path[j+1]) or _too_close(check_path[i+1], check_path[j], check_path[j+1], self_hit_margin):
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue

        # Check minimum radius on subsample
        if not _estimate_min_radius(check_path, min_rad):
            continue

        path = candidate_path
        break

    if path is None:
        # Fall back to straight path
        z_sample = np.linspace(0, size, int(size / RESAMPLE_STEP) + 1)
        x_sample = np.full_like(z_sample, x)
        path = [np.array([x_val, z]) for x_val, z in zip(x_sample, z_sample)]

    return path


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
    long_smooth_sigma = np.interp(lanes, [LANE_COUNT_MIN, LANE_COUNT_MAX], [LONG_SMOOTH_SIGMA_MIN, LONG_SMOOTH_SIGMA_MAX])
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
        across = np.arange(-total_half, total_half + ACROSS_STEP * 0.5, ACROSS_STEP)
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


def add_random_road(
    terrain,
    rng: np.random.Generator | None = None,
) -> Tuple[List[Tuple[float, float]], dict]:
    if rng is None:
        rng = np.random.default_rng()

    # Upsample the heightmap for higher resolution
    if UPSAMPLE_FACTOR > 1:
        old_res = terrain.res
        old_cell = terrain.cell_size
        old_size = terrain.size
        new_res = (old_res - 1) * UPSAMPLE_FACTOR + 1
        new_cell = old_size / (new_res - 1)
        x_old = np.linspace(0, old_size, old_res)
        z_old = x_old
        x_new = np.linspace(0, old_size, new_res)
        z_new = x_new
        spline = RectBivariateSpline(x_old, z_old, terrain.heights.T, kx=3, ky=3)
        terrain.heights = spline(x_new, z_new).T
        terrain.res = new_res
        terrain.cell_size = new_cell

    lane_width = float(rng.uniform(LANE_WIDTH_MIN, LANE_WIDTH_MAX))
    lanes = int(rng.integers(LANE_COUNT_MIN, LANE_COUNT_MAX + 1))
    shoulder = float(rng.uniform(SHOULDER_MIN, SHOULDER_MAX))
    ditch_width = float(rng.uniform(DITCH_WIDTH_MIN, DITCH_WIDTH_MAX))
    ditch_depth = float(rng.uniform(DITCH_DEPTH_MIN, DITCH_DEPTH_MAX))
    road_height = float(rng.uniform(ROAD_HEIGHT_MIN, ROAD_HEIGHT_MAX))
    cross_pitch = math.radians(float(rng.uniform(CROSS_PITCH_MIN_DEG, CROSS_PITCH_MAX_DEG)))

    noise_f = float(rng.uniform(NOISE_FREQ_MIN, NOISE_FREQ_MAX))
    bump_amp = float(rng.uniform(0.0, BUMP_MAX))
    hole_amp = float(rng.uniform(0.0, HOLE_MAX))
    # personality tweak
    choice = rng.random()
    if choice < 0.33:
        hole_amp *= 0.25  # bumpy, few holes
    elif choice > 0.66:
        bump_amp *= 0.25  # holey, few bumps

    path = _generate_bottom_to_top(terrain, rng, lanes, lane_width, shoulder)

    _stamp_road(
        terrain,
        path,
        lane_width,
        lanes,
        shoulder,
        road_height,
        cross_pitch,
        ditch_width,
        ditch_depth,
        bump_amp,
        hole_amp,
        noise_f,
        rng,
    )

    params = {
        "lane_width": lane_width,
        "lanes": lanes,
        "shoulder": shoulder,
        "road_height": road_height,
        "cross_pitch": cross_pitch,
        "ditch_width": ditch_width,
    }
    return [(float(p[0]), float(p[1])) for p in path], params


def get_safe_start_position_and_rot(terrain, road_points: List[Tuple[float, float]], distance_meters: float = 5.0):
    distance = distance_meters
    if len(road_points) < 2:
        raise ValueError("Not enough road points to compute start position.")
    xs = np.array([p[0] for p in road_points])
    zs = np.array([p[1] for p in road_points])
    dx = np.diff(xs)
    dz = np.diff(zs)
    ds = np.sqrt(dx**2 + dz**2)
    s = np.cumsum(np.concatenate(([0.0], ds)))
    if distance >= s[-1]:
        raise ValueError("Requested distance exceeds road length.")
    idx = np.searchsorted(s, distance)
    if idx == 0:
        idx = 1
    frac = (distance - s[idx - 1]) / (s[idx] - s[idx - 1])
    pos_x = xs[idx - 1] + frac * dx[idx - 1]
    pos_z = zs[idx - 1] + frac * dz[idx - 1]
    height = terrain.get_height(pos_x, pos_z) + CAR_HEIGHT_ABOVE_ROAD
    car_pos = np.array([pos_x, height, pos_z])
    dir_x = dx[idx - 1]
    dir_z = dz[idx - 1]
    norm = np.sqrt(dir_x**2 + dir_z**2)
    if norm > 0:
        dir_x /= norm
        dir_z /= norm
    yaw = math.atan2(dir_x, dir_z)
    def quat_from_yaw(theta):
        return Quaternion(math.cos(theta/2), 0.0, math.sin(theta/2), 0.0)
    car_rot = quat_from_yaw(yaw)
    return car_pos, car_rot


# ---------------------------------------------------------------------------
# Rendering mesh (gray surface + green skirts)


def build_road_vertices(
    terrain,
    path: List[Tuple[float, float]],
    lane_width: float,
    lanes: int,
    shoulder: float,
    road_height: float,
    cross_pitch: float,
    ditch_width: float | None = None,
    **_: dict,
) -> np.ndarray:
    if ditch_width is None:
        ditch_width = DITCH_WIDTH_MAX

    ROAD_COL = [0.55, 0.55, 0.55, 1.0]
    SKIRT_COL = [34 / 255, 139 / 255, 34 / 255, 1.0]

    def base_height(x: float, z: float) -> float:
        return float(terrain.get_height(x, z))

    cell = terrain.cell_size
    along_step = max(cell * ALONG_STEP_FACTOR, 0.35)

    half_road = 0.5 * lane_width * lanes
    half_plus_sh = half_road + shoulder

    gray_offsets = np.linspace(-half_plus_sh, half_plus_sh, max(6, lanes * 3 + 3))
    skirt_offsets = [half_plus_sh + (j + 1) * (ditch_width / SKIRT_RINGS) for j in range(SKIRT_RINGS)]

    verts: list[float] = []

    def emit_quad(a, b, c, d, col):
        verts.extend(a + col); verts.extend(b + col); verts.extend(c + col)
        verts.extend(c + col); verts.extend(b + col); verts.extend(d + col)

    for (p0, p1) in zip(path[:-1], path[1:]):
        p0 = np.array(p0, float); p1 = np.array(p1, float)
        seg = p1 - p0
        L = float(np.linalg.norm(seg))
        if L <= 1e-6:
            continue
        dir2 = seg / L
        nrm2 = np.array([-dir2[1], dir2[0]], dtype=float)

        nsteps = max(2, int(L / along_step))
        for k in range(nsteps):
            c0 = p0 + dir2 * (L * (k / nsteps))
            c1 = p0 + dir2 * (L * ((k + 1) / nsteps))
            deck0 = []
            deck1 = []
            h0c = base_height(c0[0], c0[1])
            h1c = base_height(c1[0], c1[1])
            for off in gray_offsets:
                y0 = h0c - math.tan(cross_pitch) * abs(off)
                y1 = h1c - math.tan(cross_pitch) * abs(off)
                deck0.append([c0[0] + nrm2[0] * off, y0, c0[1] + nrm2[1] * off])
                deck1.append([c1[0] + nrm2[0] * off, y1, c1[1] + nrm2[1] * off])
            for j in range(len(gray_offsets) - 1):
                emit_quad(deck0[j], deck0[j + 1], deck1[j], deck1[j + 1], ROAD_COL)

            # skirts on each side
            for side in (-1.0, +1.0):
                edge_idx = -1 if side > 0 else 0
                ring_prev0 = deck0[edge_idx]
                ring_prev1 = deck1[edge_idx]
                edge_y0 = ring_prev0[1]
                edge_y1 = ring_prev1[1]
                for j, off in enumerate(skirt_offsets):
                    d = off * side
                    x0 = c0[0] + nrm2[0] * d; z0 = c0[1] + nrm2[1] * d
                    x1 = c1[0] + nrm2[0] * d; z1 = c1[1] + nrm2[1] * d
                    t = (off - half_plus_sh) / max(ditch_width, 1e-6)
                    t = min(max(t, 0.0), 1.0)
                    y0 = (1 - t) * edge_y0 + t * float(terrain.get_height(x0, z0))
                    y1 = (1 - t) * edge_y1 + t * float(terrain.get_height(x1, z1))
                    ring0 = [x0, y0, z0]
                    ring1 = [x1, y1, z1]
                    emit_quad(ring_prev0, ring0, ring_prev1, ring1, SKIRT_COL)
                    ring_prev0, ring_prev1 = ring0, ring1

    return np.array(verts, dtype="f4")

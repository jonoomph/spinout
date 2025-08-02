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

Improvements in this version:
- Increased deviation scales and allowed more backtracking for snakier paths on narrower roads.
- Made road height lane-dependent: higher for more lanes to keep edges above terrain.
- Reduced resampling step for fewer dense points while maintaining smoothness.
- Added sinuosity and terrain height range to the road plan printout.
- Fixed mini-map downsampling and added pure NumPy implementation.
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

# Road crown above terrain (prevents z-fighting), lane-dependent
ROAD_HEIGHT_MIN_BY_LANES = {1: 0.5, 2: 0.8, 3: 1.2, 4: 1.6, 5: 2.0, 6: 2.5}  # m min
ROAD_HEIGHT_MAX_BY_LANES = {1: 1.5, 2: 2.0, 3: 2.5, 4: 3.0, 5: 3.5, 6: 4.0}  # m max

# Centerline generation (bottom → top bias)
RESAMPLE_STEP = 5.0          # dense resampling step for stamping (increased for fewer points)
MAX_CTRL_POINTS = 25         # max number of control points (fewer for more lanes)
MIN_CTRL_POINTS = 4
CTRL_DEVIATION_SCALE_X = 0.5 # deviation factor for x, smaller for more lanes
CTRL_DEVIATION_SCALE_Z = 0.3 # deviation factor for z, smaller for more lanes
SELF_HIT_MARGIN_FACTOR = 1.5 # margin as factor of full road width
BOTTOM_MARGIN = 0.0         # start z at bottom
SIDE_MARGIN = 0.10           # keep x inside 10% margins
TOP_EXIT_MARGIN = 0.0       # stop once z > 95% of map size
GEN_MAX_TRIES = 50           # max tries to generate non-overlapping path
MAX_CTRL_FACTOR = 3          # max control points = target * factor
BACKTRACK_FACTOR = 0.3       # max backtrack as fraction of z_advance

# Curvature is lane-width aware. Use min radius per lane count.
# (crude but effective for shaping)
MIN_RADIUS_BY_LANES = {  # meters
    1: 15.0,
    2: 25.0,
    3: 45.0,
    4: 80.0,
    5: 120.0,
    6: 160.0,
}
MIN_RADIUS_SCALE_WITH_LANEWIDTH = 3.2  # scale vs 3.2 m nominal lane
MIN_DEV_SCALE = 0.05  # minimum deviation scale for large roads

# Sampling density around the road for stamping & rendering
ALONG_STEP_FACTOR = 0.1   # step = cell_size * factor (>= 0.35 m)
ACROSS_STEP = 0.5          # meters between cross samples for stamping
SKIRT_RINGS = 7            # number of green rings to blend back to terrain

# Surface irregularities (pick a personality once per road)
NOISE_FREQ_MIN = 0.01
NOISE_FREQ_MAX = 0.03
BUMP_MAX = 0.05  # m
HOLE_MAX = 0.05  # m depth (negative)

# Spline smoothing for resampling (in dense points units)
PATH_SMOOTH_SIGMA_MIN = 1.0
PATH_SMOOTH_SIGMA_MAX = 20.0  # higher for more lanes

# Longitudinal smoothing for road base height
LONG_SMOOTH_SIGMA_MIN = 100.0
LONG_SMOOTH_SIGMA_MAX = 300.0  # higher for smoother longitudinal (in dense points units)

# Heightmap upsampling for higher resolution near roads
UPSAMPLE_FACTOR = 4  # Increase resolution by this factor for better detail

# Car placement
CAR_HEIGHT_ABOVE_ROAD = 1.2  # meters

# Lane markings
LINE_WIDTH = 0.15  # m
DASH_LENGTH = 3.0  # m
GAP_LENGTH = 9.0   # m
ROAD_EPS = 0.01    # small offset to prevent z-fighting and ensure mesh above terrain

# Colors
ROAD_COL = [0.55, 0.55, 0.55, 1.0]
SKIRT_COL = [34 / 255, 139 / 255, 34 / 255, 1.0]
YELLOW_COL = [1.0, 1.0, 0.0, 1.0]
WHITE_COL = [1.0, 1.0, 1.0, 1.0]

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
    ctrl_points = [np.array([x, 0.0])]

    # Lane-aware parameters
    min_rad = MIN_RADIUS_BY_LANES.get(lanes, 65.0)
    min_rad *= (lane_width / MIN_RADIUS_SCALE_WITH_LANEWIDTH)
    min_rad = max(min_rad, 12.0)

    # Self-hit margin based on road width
    full_width = lanes * lane_width + 2 * shoulder
    self_hit_margin = full_width * SELF_HIT_MARGIN_FACTOR

    # Number of control points: more for fewer lanes
    num_ctrl_target = int(np.interp(lanes, [LANE_COUNT_MIN, LANE_COUNT_MAX], [MAX_CTRL_POINTS, MIN_CTRL_POINTS]))

    # Deviation scale: larger for fewer lanes
    rel = (lanes - 1) / (LANE_COUNT_MAX - 1)
    dev_scale_x = max(MIN_DEV_SCALE, CTRL_DEVIATION_SCALE_X * (1 - rel))
    dev_scale_z = max(MIN_DEV_SCALE, CTRL_DEVIATION_SCALE_Z * (1 - rel))

    path = None
    for _ in range(GEN_MAX_TRIES):
        ctrl_points = [np.array([x, 0.0])]
        prev_x = x
        prev_z = 0.0
        z_advance = size / num_ctrl_target
        while len(ctrl_points) < num_ctrl_target * MAX_CTRL_FACTOR and prev_z < size:
            x_dev = rng.uniform(-dev_scale_x * size, dev_scale_x * size)
            z_dev = rng.uniform(-BACKTRACK_FACTOR * z_advance, z_advance)
            x_next = np.clip(prev_x + x_dev, SIDE_MARGIN * size, (1 - SIDE_MARGIN) * size)
            z_next = prev_z + z_advance + z_dev
            if z_next > size:
                z_next = size
            ctrl_points.append(np.array([x_next, z_next]))
            prev_x = x_next
            prev_z = z_next
        if prev_z < size or len(ctrl_points) < 2:
            continue

        # Parameterize by arc length
        ctrl_array = np.array(ctrl_points)
        ds = np.linalg.norm(np.diff(ctrl_array, axis=0), axis=1)
        s = np.cumsum(np.concatenate(([0.0], ds)))
        cs_x = CubicSpline(s, ctrl_array[:, 0], bc_type='natural')
        cs_z = CubicSpline(s, ctrl_array[:, 1], bc_type='natural')

        # Resample densely
        s_sample = np.linspace(0, s[-1], int(s[-1] / RESAMPLE_STEP) + 1)
        x_sample = cs_x(s_sample)
        z_sample = cs_z(s_sample)
        candidate_path = [np.array([x_val, z_val]) for x_val, z_val in zip(x_sample, z_sample)]

        # Smooth the path
        path_smooth_sigma_dense = np.interp(lanes, [LANE_COUNT_MIN, LANE_COUNT_MAX], [PATH_SMOOTH_SIGMA_MIN, PATH_SMOOTH_SIGMA_MAX])
        if path_smooth_sigma_dense > 0.001:
            xs_smooth = gaussian_filter1d(x_sample, path_smooth_sigma_dense)
            zs_smooth = gaussian_filter1d(z_sample, path_smooth_sigma_dense)
            candidate_path = [np.array([x, z]) for x, z in zip(xs_smooth, zs_smooth)]

        # Check for self-overlap
        check_skip = 5
        check_path = candidate_path[::check_skip]
        ok = True
        n = len(check_path)
        for i in range(n - 1):
            for j in range(i + 20, n - 1):  # skip nearby segments
                if _seg_intersect(check_path[i], check_path[i+1], check_path[j], check_path[j+1]):
                    ok = False
                    break
                if _too_close(check_path[i+1], check_path[j], check_path[j+1], self_hit_margin):
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue

        # Check minimum radius
        if not _estimate_min_radius(candidate_path, min_rad):
            continue

        path = candidate_path
        break

    if path is None:
        # Fall back to straight path
        z_sample = np.linspace(0, size, int(size / RESAMPLE_STEP) + 1)
        x_sample = np.full_like(z_sample, x)
        path = [np.array([x_val, z_val]) for x_val, z_val in zip(x_sample, z_sample)]

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


def add_random_road(
    terrain,
    rng: np.random.Generator | None = None,
    road_type: str = "asphalt",
    weather: str = "dry",
    terrain_type: str = "grass",
    road_color=ROAD_COL,
    skirt_color=SKIRT_COL,
    road_friction: float = 1.0,
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
        spline_fric = RectBivariateSpline(x_old, z_old, terrain.surface_friction.T, kx=1, ky=1)
        terrain.surface_friction = spline_fric(x_new, z_new).T
        spline_road = RectBivariateSpline(x_old, z_old, terrain.road_friction.T, kx=1, ky=1)
        terrain.road_friction = spline_road(x_new, z_new).T
        terrain.res = new_res
        terrain.cell_size = new_cell

    lane_width = float(rng.uniform(LANE_WIDTH_MIN, LANE_WIDTH_MAX))
    lanes = int(rng.integers(LANE_COUNT_MIN, LANE_COUNT_MAX + 1))
    shoulder = float(rng.uniform(SHOULDER_MIN, SHOULDER_MAX))
    ditch_width = float(rng.uniform(DITCH_WIDTH_MIN, DITCH_WIDTH_MAX))
    ditch_depth = float(rng.uniform(DITCH_DEPTH_MIN, DITCH_DEPTH_MAX))
    road_height_min = ROAD_HEIGHT_MIN_BY_LANES.get(lanes, 1.0)
    road_height_max = ROAD_HEIGHT_MAX_BY_LANES.get(lanes, 3.0)
    road_height = float(rng.uniform(road_height_min, road_height_max))
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

    path_arr = np.array(path)
    ds = np.linalg.norm(np.diff(path_arr, axis=0), axis=1)
    path_length = np.sum(ds)
    sinuosity = path_length / terrain.size if terrain.size > 0 else 1.0

    heights_orig = terrain.heights.copy()
    h_min = np.min(heights_orig)
    h_max = np.max(heights_orig)

    min_rad = MIN_RADIUS_BY_LANES.get(lanes, 65.0) * (lane_width / MIN_RADIUS_SCALE_WITH_LANEWIDTH)

    # Print road plan
    print("Generating road plan:")
    print(f"  Weather: {weather}")
    print(f"  Road type: {road_type} (friction {road_friction:.2f})")
    print(
        f"  Terrain type: {terrain_type} (friction {terrain.base_friction:.2f})"
    )
    print(f"  Lanes: {lanes}")
    print(f"  Lane width: {lane_width:.2f} m")
    print(f"  Shoulder width: {shoulder:.2f} m")
    print(f"  Road height above terrain: {road_height:.2f} m")
    print(f"  Cross pitch: {math.degrees(cross_pitch):.2f} deg")
    print(f"  Ditch width: {ditch_width:.2f} m")
    print(f"  Ditch depth: {ditch_depth:.2f} m")
    print(f"  Minimum curve radius: {min_rad:.2f} m")
    print(f"  Number of road points: {len(path)}")
    print(f"  Sinuosity: {sinuosity:.2f}")
    print(f"  Terrain height range: {h_min:.2f} to {h_max:.2f} m")

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
        road_friction,
    )

    params = {
        "lane_width": lane_width,
        "lanes": lanes,
        "shoulder": shoulder,
        "road_height": road_height,
        "cross_pitch": cross_pitch,
        "ditch_width": ditch_width,
        "road_color": road_color,
        "skirt_color": skirt_color,
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
# Rendering mesh (gray surface + green skirts + lane markings)


def build_road_vertices(
    terrain,
    path: List[Tuple[float, float]],
    lane_width: float,
    lanes: int,
    shoulder: float,
    road_height: float,
    cross_pitch: float,
    ditch_width: float | None = None,
    road_color=ROAD_COL,
    skirt_color=SKIRT_COL,
    **_: dict,
) -> np.ndarray:
    if ditch_width is None:
        ditch_width = DITCH_WIDTH_MAX

    def base_height(x: float, z: float) -> float:
        return float(terrain.get_height(x, z))

    cell = terrain.cell_size
    along_step = max(cell * ALONG_STEP_FACTOR, 0.35)

    half_road = 0.5 * lane_width * lanes
    half_plus_sh = half_road + shoulder

    gray_offsets = np.linspace(-half_plus_sh, half_plus_sh, max(6, lanes * 3 + 3))
    skirt_offsets = [half_plus_sh + (j + 1) * (ditch_width / SKIRT_RINGS) for j in range(SKIRT_RINGS)]

    # Compute cumulative distance along path
    s = [0.0]
    for p_prev, p_curr in zip(path[:-1], path[1:]):
        dist = math.sqrt((p_curr[0] - p_prev[0])**2 + (p_curr[1] - p_prev[1])**2)
        s.append(s[-1] + dist)

    # Define lane marking lines: (offset, color, dotted)
    lines_list = []
    left_edge_off = -half_plus_sh
    right_edge_off = half_plus_sh
    left_edge_col = YELLOW_COL if lanes == 1 else WHITE_COL
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
        col = YELLOW_COL if is_yellow else WHITE_COL
        dotted = not is_yellow
        lines_list.append((off, col, dotted))
    lines_list.append((right_edge_off, WHITE_COL, False))

    verts: list[float] = []

    def emit_quad(a, b, c, d, col):
        verts.extend(a + col); verts.extend(b + col); verts.extend(c + col)
        verts.extend(c + col); verts.extend(b + col); verts.extend(d + col)

    path_np = [np.array(p, float) for p in path]
    period = DASH_LENGTH + GAP_LENGTH
    line_half = LINE_WIDTH / 2.0

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
                y0 = base_height(x0, z0) + ROAD_EPS
                deck0.append([x0, y0, z0])

                x1 = c1[0] + nrm2[0] * off
                z1 = c1[1] + nrm2[1] * off
                y1 = base_height(x1, z1) + ROAD_EPS
                deck1.append([x1, y1, z1])
            for j in range(len(gray_offsets) - 1):
                emit_quad(deck0[j], deck0[j + 1], deck1[j], deck1[j + 1], road_color)

            # skirts on each side
            for side in (-1.0, +1.0):
                edge_idx = -1 if side > 0 else 0
                ring_prev0 = deck0[edge_idx]
                ring_prev1 = deck1[edge_idx]
                edge_y0 = ring_prev0[1] - ROAD_EPS  # subtract eps to compute blend, then add back
                edge_y1 = ring_prev1[1] - ROAD_EPS
                for j in range(SKIRT_RINGS):
                    off = skirt_offsets[j]
                    d = off * side
                    x0 = c0[0] + nrm2[0] * d; z0 = c0[1] + nrm2[1] * d
                    x1 = c1[0] + nrm2[0] * d; z1 = c1[1] + nrm2[1] * d
                    t = (off - half_plus_sh) / max(ditch_width, 1e-6)
                    t = min(max(t, 0.0), 1.0)
                    y0 = (1 - t) * edge_y0 + t * float(base_height(x0, z0)) + ROAD_EPS
                    y1 = (1 - t) * edge_y1 + t * float(base_height(x1, z1)) + ROAD_EPS
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
                    should_emit = mod < DASH_LENGTH
                if should_emit:
                    left_off = off - line_half
                    right_off = off + line_half

                    xl0 = c0[0] + nrm2[0] * left_off
                    zl0 = c0[1] + nrm2[1] * left_off
                    yl0 = base_height(xl0, zl0) + ROAD_EPS
                    vert_l0 = [xl0, yl0, zl0]

                    xr0 = c0[0] + nrm2[0] * right_off
                    zr0 = c0[1] + nrm2[1] * right_off
                    yr0 = base_height(xr0, zr0) + ROAD_EPS
                    vert_r0 = [xr0, yr0, zr0]

                    xl1 = c1[0] + nrm2[0] * left_off
                    zl1 = c1[1] + nrm2[1] * left_off
                    yl1 = base_height(xl1, zl1) + ROAD_EPS
                    vert_l1 = [xl1, yl1, zl1]

                    xr1 = c1[0] + nrm2[0] * right_off
                    zr1 = c1[1] + nrm2[1] * right_off
                    yr1 = base_height(xr1, zr1) + ROAD_EPS
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
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
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

# Allow running this file directly without installing the package
import os
import sys
if __package__ is None:  # e.g. `python src/roads/plan.py`
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
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
ROAD_EPS = 0.015    # small offset to prevent z-fighting and ensure mesh above terrain
LINE_EPS = 0.016    # slightly higher offset for lane markings
DRIVE_LINE_WIDTH = 0.35  # m width of green driveline guide
DRIVE_LINE_HEIGHT = 0.05  # m above road to avoid z-fighting for driveline
DRIVE_LINE_STEP = 1.0  # m spacing of driveline samples for rendering

# Colors
ROAD_COL = [0.55, 0.55, 0.55, 1.0]
SKIRT_COL = [34 / 255, 139 / 255, 34 / 255, 1.0]
YELLOW_COL = [1.0, 1.0, 0.0, 1.0]
WHITE_COL = [1.0, 1.0, 1.0, 1.0]

# Speed limit choices in miles per hour for US style roads
# Use 5 mph increments from 25 up to 80
SPEED_LIMIT_CHOICES = list(range(25, 85, 5))

# Minimum distance the driveline stays in a lane before considering a change (m)
LANE_CHANGE_MIN_DIST = 100.0

# Sigmoid steepness for lane changes
LANE_CHANGE_SIGMOID_K = 1.5

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


def _max_tail_heading_change(xs, ys, check_meters=300.0):
    """Return max heading change (deg) in the last `check_meters` section of the spline."""
    pts = np.column_stack([xs, ys])
    y_span = ys[-1] - check_meters
    indices = np.where(ys >= y_span)[0]
    if len(indices) < 3:
        indices = np.arange(len(xs))
    pts = pts[indices]
    v = np.diff(pts, axis=0)
    headings = np.arctan2(v[:,1], v[:,0])
    heading_diff = np.diff(headings)
    heading_diff = (heading_diff + np.pi) % (2 * np.pi) - np.pi
    return np.max(np.abs(np.rad2deg(heading_diff))) if len(heading_diff) > 0 else 0.0

def _generate_bottom_to_top(
    terrain,
    rng: np.random.Generator,
    lanes: int,
    lane_width: float,
    shoulder: float,
    max_spline_turn_deg=45.0,
    check_meters=300.0,
):
    size = terrain.size
    x = rng.uniform(0.15 * size, 0.85 * size)
    y = 0.0

    step = 100.0

    # Lane-dependent max angle: curvier for fewer lanes, straighter for more
    LANES_MIN = 1
    LANES_MAX = 6
    ANGLE_MIN_DEG = 20.0   # for 6 lanes
    ANGLE_MAX_DEG = 80.0   # for 1 lane

    lanes_clamped = np.clip(lanes, LANES_MIN, LANES_MAX)
    start_angle_deg = np.interp(lanes_clamped, [LANES_MIN, LANES_MAX], [ANGLE_MAX_DEG, ANGLE_MIN_DEG])
    max_angle = np.deg2rad(start_angle_deg)
    print(f"[debug] Using max angle {start_angle_deg:.1f}° for {lanes} lanes")

    min_angle = 0.0
    angle_step = np.deg2rad(10)
    min_map_x = 0.05 * size
    max_map_x = 0.95 * size

    points = [np.array([x, y])]
    print(f"[debug] Start point: ({x:.2f}, {y:.2f})")

    while y < size:
        found_point = False
        try_angles = np.arange(max_angle, min_angle - angle_step, -angle_step)
        for angle_offset in try_angles:
            if angle_offset > 0:
                angle = np.pi / 2 + rng.uniform(-angle_offset, angle_offset)
            else:
                angle = np.pi / 2  # perfectly straight up

            # Always accept 0-degree (vertical) as a fail-safe, skip all checks
            if angle_offset <= 0.0 or np.isclose(angle_offset, 0.0):
                dx = 0
                dy = step
                x_new = np.clip(x, min_map_x, max_map_x)
                y_new = y + dy
                if y_new >= size:
                    y_new = size
                points.append(np.array([x_new, y_new]))
                print(f"[debug] Forcibly added straight control point: ({x_new:.2f}, {y_new:.2f})")
                x, y = x_new, y_new
                found_point = True
                break

            # Usual candidate checks
            dx = step * np.cos(angle)
            dy = step * np.sin(angle)
            x_new = np.clip(x + dx, min_map_x, max_map_x)
            y_new = y + dy
            if y_new >= size:
                y_new = size
                x_new = np.clip(x_new + rng.uniform(-30, 30), min_map_x, max_map_x)

            # Try this point
            test_points = points + [np.array([x_new, y_new])]
            xs = [pt[0] for pt in test_points]
            ys = [pt[1] for pt in test_points]
            try:
                cs = CubicSpline(ys, xs, bc_type="natural")
                y_test = np.linspace(max(0, ys[-1] - check_meters), ys[-1], 10)
                x_test = cs(y_test)
                if np.any(x_test < 0) or np.any(x_test > size):
                    continue  # out of bounds

                heading_change = _max_tail_heading_change(x_test, y_test, check_meters)
                if heading_change > max_spline_turn_deg:
                    print(f"[debug] REJECT angle {np.rad2deg(angle_offset):.1f}°: tail heading change {heading_change:.1f}°")
                    continue  # too sharp
            except Exception:
                continue  # Spline fitting failed

            # Accept point
            points.append(np.array([x_new, y_new]))
            print(f"[debug] Control point: ({x_new:.2f}, {y_new:.2f}), angle={np.rad2deg(angle_offset):.1f}°")
            x, y = x_new, y_new
            found_point = True
            break

        if not found_point:
            print("[debug] Could not find a valid control point, ending early")
            break

    # Final spline fit
    xs = [pt[0] for pt in points]
    ys = [pt[1] for pt in points]
    cs = CubicSpline(ys, xs, bc_type="natural")
    n_samples = max(2, int(size // 2))
    y_samples = np.linspace(0, ys[-1], n_samples)
    x_samples = cs(y_samples)

    sigma = np.interp(lanes, [1, 4], [2.0, 0.5])
    if sigma > 1.0:
        x_samples = gaussian_filter1d(x_samples, sigma)

    print(f"[debug] Sampled {len(x_samples)} spline points.")
    for i in range(0, len(x_samples), max(1, len(x_samples)//10)):
        print(f"[debug] Spline pt: ({x_samples[i]:.2f}, {y_samples[i]:.2f})")

    heading_change = _max_tail_heading_change(x_samples, y_samples, check_meters)
    print(f"[debug] FINAL max heading change (tail): {heading_change:.1f}°")
    path = [np.array([float(xi), float(yi)]) for xi, yi in zip(x_samples, y_samples)]
    return path


def _compute_speed_limits(path: List[_vec2], terrain, road_friction: float) -> List[dict]:
    """Compute speed limits every 100 m based on curvature and slope."""
    path_arr = np.array(path)
    xs = path_arr[:, 0]
    zs = path_arr[:, 1]
    dx = np.diff(xs)
    dz = np.diff(zs)
    ds = np.sqrt(dx ** 2 + dz ** 2)
    s = np.concatenate(([0.0], np.cumsum(ds)))
    heights = np.array([terrain.get_height(x, z) for x, z in path_arr])

    segments = []
    total_len = s[-1]
    g = 9.81
    prev_speed = None
    for start in np.arange(0.0, total_len, 100.0):
        end = min(start + 100.0, total_len)
        idx0 = int(np.searchsorted(s, start, side="left"))
        idx1 = int(np.searchsorted(s, end, side="left"))
        idx1 = min(max(idx1, idx0 + 2), len(path_arr) - 1)

        dx_seg = np.diff(xs[idx0:idx1 + 1])
        dz_seg = np.diff(zs[idx0:idx1 + 1])
        ddx = np.diff(dx_seg)
        ddz = np.diff(dz_seg)
        if len(ddx) > 0:
            kappa = (
                np.abs(dx_seg[:-1] * ddz - dz_seg[:-1] * ddx)
                / ((dx_seg[:-1] ** 2 + dz_seg[:-1] ** 2) ** 1.5 + 1e-9)
            )
            max_kappa = float(np.max(kappa)) if kappa.size else 0.0
        else:
            max_kappa = 0.0
        min_radius = math.inf if max_kappa <= 1e-6 else 1.0 / max_kappa

        seg_heights = heights[idx0:idx1 + 1]
        seg_ds = ds[idx0:idx1]
        grades = np.abs(np.diff(seg_heights) / (seg_ds + 1e-6))
        max_grade = float(np.max(grades)) if grades.size else 0.0

        v_mps = math.sqrt(max(road_friction, 0.1) * g * min_radius) if math.isfinite(min_radius) else 100.0
        v_mph = v_mps * 2.23694
        if max_grade > 0.06:
            v_mph *= 0.7
        elif max_grade > 0.04:
            v_mph *= 0.85

        speed = SPEED_LIMIT_CHOICES[0]
        for lim in SPEED_LIMIT_CHOICES:
            if v_mph >= lim:
                speed = lim
        if prev_speed is not None and speed > prev_speed + 15:
            speed = min(speed, prev_speed + 15)
        prev_speed = speed

        # place sign slightly ahead of the segment start so the first is visible
        # to the car when spawning at the bottom of the road
        sign_idx = min(idx0 + 1, len(path_arr) - 1)
        segments.append({
            "start_idx": idx0,
            "end_idx": idx1,
            "start_s": float(start),
            "end_s": float(end),
            "speed_mph": int(speed),
            "sign_idx": sign_idx,
        })

    return segments


def _lane_center_offset(lane: int, lanes: int, lane_width: float) -> float:
    """Return offset from center for a lane index (0 = leftmost).

    Positive offsets point to the left of travel; negative offsets are to the
    right. This matches the left-handed normal used when sweeping road geometry
    so that the rightmost lane ends up on the correct side of the road."""
    half = lane_width * lanes / 2.0
    # Leftmost lane has the largest positive offset; rightmost is most negative
    return half - lane_width * 0.5 - lane * lane_width


def _speed_limit_at(dist: float, segments: List[dict]) -> float:
    for seg in segments:
        if seg["start_s"] <= dist < seg["end_s"]:
            return seg["speed_mph"]
    return segments[-1]["speed_mph"] if segments else SPEED_LIMIT_CHOICES[0]


def _smoothstep(x: float) -> float:
    """Cubic easing with zero slope at both ends."""
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def _generate_drive_line(
    path: List[_vec2],
    segments: List[dict],
    lanes: int,
    lane_width: float,
    rng: np.random.Generator,
) -> List[Tuple[float, float]]:
    path_arr = np.array(path)
    xs = path_arr[:, 0]
    zs = path_arr[:, 1]
    dx = np.diff(xs)
    dz = np.diff(zs)
    ds = np.sqrt(dx ** 2 + dz ** 2)
    s = np.concatenate(([0.0], np.cumsum(ds)))

    x_spline = CubicSpline(s, xs)
    z_spline = CubicSpline(s, zs)
    dx_ds = x_spline.derivative()
    dz_ds = z_spline.derivative()

    total = s[-1]
    step = DRIVE_LINE_STEP
    s_vals = np.arange(0.0, total + step, step)

    start_lane = int(rng.integers(lanes // 2, lanes)) if lanes > 1 else 0
    current_lane = start_lane
    offset = _lane_center_offset(current_lane, lanes, lane_width)
    next_change = LANE_CHANGE_MIN_DIST
    change = None
    drive_line: List[Tuple[float, float]] = []

    for s_i in s_vals:
        if change:
            frac = (s_i - change["start_s"]) / change["dist"]
            t = _smoothstep(frac)
            offset = change["start_off"] + (change["target_off"] - change["start_off"]) * t
            if frac >= 1.0:
                current_lane = change["target_lane"]
                change = None
                next_change = s_i + LANE_CHANGE_MIN_DIST
        else:
            offset = _lane_center_offset(current_lane, lanes, lane_width)
            if s_i >= next_change and lanes >= 3:
                min_lane = lanes // 2
                adj = []
                if current_lane - 1 >= min_lane:
                    adj.append(current_lane - 1)
                if current_lane + 1 < lanes:
                    adj.append(current_lane + 1)
                choices = [current_lane] + adj
                new_lane = int(rng.choice(choices))
                next_change = s_i + LANE_CHANGE_MIN_DIST
                if new_lane != current_lane:
                    speed = _speed_limit_at(s_i, segments)
                    dist_change = float(np.interp(speed, [25, 80], [40.0, 120.0]))
                    change = {
                        "start_s": s_i,
                        "dist": dist_change,
                        "start_off": offset,
                        "target_off": _lane_center_offset(new_lane, lanes, lane_width),
                        "target_lane": new_lane,
                    }

        x = float(x_spline(s_i))
        z = float(z_spline(s_i))
        tdir = np.array([dx_ds(s_i), dz_ds(s_i)], dtype=float)
        n = np.linalg.norm(tdir)
        if n > 1e-6:
            tdir /= n
        else:
            tdir = np.array([0.0, 1.0])
        nrm = np.array([-tdir[1], tdir[0]])
        pos = np.array([x, z]) + nrm * offset
        drive_line.append((float(pos[0]), float(pos[1])))

    return drive_line


def generate_plan(
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

    # Determine speed limits and drive line before returning
    speed_limits = _compute_speed_limits(path, terrain, road_friction)
    drive_line = _generate_drive_line(path, speed_limits, lanes, lane_width, rng)

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

    params = {
        "lane_width": lane_width,
        "lanes": lanes,
        "shoulder": shoulder,
        "road_height": road_height,
        "cross_pitch": cross_pitch,
        "ditch_width": ditch_width,
        "ditch_depth": ditch_depth,
        "bump_amp": bump_amp,
        "hole_amp": hole_amp,
        "noise_f": noise_f,
        "road_friction": road_friction,
        "road_color": road_color,
        "skirt_color": skirt_color,
        "speed_limits": speed_limits,
        "drive_line": drive_line,
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
    height = terrain.get_height(pos_x, pos_z)
    if not np.isfinite(height):
        raise ValueError("Start position lies outside terrain bounds")
    car_pos = np.array([pos_x, height + CAR_HEIGHT_ABOVE_ROAD, pos_z])
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

def preview_plan(terrain, road_points, lanes=1):
    """
    Show a top-down view of the road centreline over the terrain heightmap.
    """
    # terrain.heights: (res × res) array
    heights = terrain.heights
    size = terrain.size

    plt.figure(figsize=(6, 6))
    plt.imshow(
        heights.T,
        origin='lower',
        cmap='terrain',
        extent=[0, size, 0, size],
        alpha=0.8
    )

    xs = [p[0] for p in road_points]
    zs = [p[1] for p in road_points]

    min_width = 2
    max_width = 8
    width = min_width + (max_width - min_width) * ((lanes - 1) / (LANE_COUNT_MAX - 1))
    plt.plot(xs, zs, color='red', linewidth=width, label=f'{lanes} Lane Road')

    plt.xlabel('X (m)')
    plt.ylabel('Z (m)')
    plt.title('Road Plan Preview')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    """Generate a random terrain and stamp a road for quick experimentation."""
    import os
    import sys
    import numpy as np

    # Ensure repo root on sys.path when running as script
    if __package__ is None:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from physics import Terrain  # type: ignore
        from roads.build import apply_plan  # type: ignore
    else:
        from src.physics import Terrain
        from .build import apply_plan

    rng = np.random.default_rng()
    terrain = Terrain(size=800, res=200)
    pts, params = generate_plan(terrain, rng=rng)
    apply_plan(terrain, pts, params, rng=rng)

    print(f"Generated road with {len(pts)} points on {terrain.res}×{terrain.res} terrain")
    preview_plan(terrain, pts, params['lanes'])


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

        # Ensure the path stays within the terrain bounds. Cubic splines can
        # overshoot the control points, so reject any candidate that leaves the
        # playable area before doing the heavier intersection checks.
        if any(
            (p[0] < SIDE_MARGIN * size)
            or (p[0] > (1.0 - SIDE_MARGIN) * size)
            or (p[1] < BOTTOM_MARGIN * size)
            or (p[1] > size)
            for p in candidate_path
        ):
            continue

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

def preview_plan(terrain, road_points):
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
    plt.plot(xs, zs, color='red', linewidth=2, label='Road centreline')

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
    preview_plan(terrain, pts)


"""Helpers to convert a road plan into geometry and terrain updates."""


from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from . import plan as cfg

# ---------------------------------------------------------------------------
# Helpers


_vec2 = np.ndarray


def _sigma_curve(t: float, steepness: float) -> float:
    """Return a 0..1 curve shaped like a logistic sigma."""

    t = float(np.clip(t, 0.0, 1.0))
    if steepness <= 0.0:
        return t
    half = steepness * 0.5
    exp_pos = math.exp(half)
    exp_neg = math.exp(-half)
    # Normalize so curve(0)=0 and curve(1)=1
    denom = 1.0 + exp_neg
    start = 1.0 / (1.0 + exp_pos)
    end = 1.0 / denom
    val = 1.0 / (1.0 + math.exp(-steepness * (t - 0.5)))
    return (val - start) / (end - start + 1e-12)


def _estimate_required_ditch_width(
    terrain,
    base_height,
    c: np.ndarray,
    nrm2: np.ndarray,
    half_road: float,
    shoulder: float,
    base_width: float,
    start_height: float,
) -> float:
    """Extend ditch width so the skirt grade stays within bounds."""

    width = max(base_width, 0.0)
    max_width = base_width + cfg.SKIRT_EXTRA_MAX
    for _ in range(12):
        offset = half_road + shoulder + width
        x = c[0] + nrm2[0] * offset
        z = c[1] + nrm2[1] * offset
        terrain_y = base_height(x, z)
        drop = start_height - terrain_y
        required = drop / max(cfg.SKIRT_MAX_GRADE, 1e-6)
        if required <= width + 0.05:
            break
        width = min(max_width, max(width + 0.5, required))
        if width >= max_width - 1e-6:
            break
    return max(width, 0.0)


def _build_offsets(half_road: float, shoulder: float, ditch_width: float) -> List[float]:
    """Generate symmetric offsets for the ring mesh."""

    spacing = max(cfg.SKIRT_SAMPLE_SPACING, 0.2)
    outer = max(half_road + shoulder + ditch_width, spacing)
    count = max(1, int(math.ceil(outer / spacing)))
    step = outer / count if count > 0 else spacing
    offsets_pos = [0.0]
    for i in range(1, count + 1):
        offsets_pos.append(min(outer, i * step))
    required = {0.0, outer}
    if half_road > 1e-6:
        required.add(min(outer, half_road))
    if shoulder > 1e-6:
        required.add(min(outer, half_road + shoulder))
    offsets_pos.extend(required)
    if abs(offsets_pos[-1] - outer) > 1e-6:
        offsets_pos.append(outer)
    offsets_pos = sorted(set(round(v, 6) for v in offsets_pos))
    neg = [-v for v in reversed(offsets_pos[1:])]
    return neg + offsets_pos


def _road_surface_height(
    terrain,
    base_height,
    sample,
    offset: float,
    half_road: float,
    shoulder: float,
    curb_height: float,
    cross_pitch: float,
    ditch_sigma: float,
    ditch_width: float,
) -> float:
    """Compute the road/skirt height for a given lateral offset."""

    c = sample["center"]
    nrm2 = sample["nrm2"]
    terrain_y = base_height(c[0] + nrm2[0] * offset, c[1] + nrm2[1] * offset)
    center_surface = sample["center_surface"]
    edge_height = center_surface - math.tan(cross_pitch) * half_road
    deck_drop = math.tan(cross_pitch) * min(abs(offset), half_road)
    deck_height = center_surface - deck_drop
    if abs(offset) <= half_road:
        return deck_height + cfg.ROAD_EPS

    x_rel = abs(offset) - half_road
    shoulder_start = edge_height
    ditch_start = edge_height - curb_height
    cap_drop = min(cfg.ROAD_EPS * 0.6, 0.004)
    cap_height = deck_height + cfg.ROAD_EPS - cap_drop
    terrain_min = terrain_y + cfg.SKIRT_EPS

    if shoulder > 1e-6:
        if x_rel <= shoulder:
            frac = x_rel / max(shoulder, 1e-6)
            drop = curb_height * _sigma_curve(frac, cfg.SHOULDER_SIGMA_K)
            surface = shoulder_start - drop
            if terrain_min <= cap_height:
                surface = max(surface, terrain_min)
            return float(min(surface, cap_height))
        x_rel -= shoulder
    else:
        ditch_start = edge_height - curb_height

    if ditch_width <= 1e-6:
        surface = ditch_start
        if terrain_min <= cap_height:
            surface = max(surface, terrain_min)
        return float(min(surface, cap_height))

    t = min(x_rel / max(ditch_width, 1e-6), 1.0)
    blend = _sigma_curve(t, ditch_sigma)
    target = terrain_y + cfg.SKIRT_EPS
    height = (1.0 - blend) * ditch_start + blend * target
    if terrain_min <= cap_height:
        height = max(height, terrain_min)
    return float(min(height, cap_height))


def _collect_profile(
    terrain,
    path_np: list[np.ndarray],
    half_road: float,
    shoulder: float,
    road_height: float,
    cross_pitch: float,
    ditch_width: float,
    ditch_depth: float,
    along_step: float,
):
    """Gather per-sample geometry information shared by several builders."""

    def base_height(x: float, z: float) -> float:
        return float(terrain.get_height(x, z, include_roads=False))

    curb_height = cfg.CURB_HEIGHT
    ditch_sigma = float(
        np.interp(
            ditch_depth,
            [cfg.DITCH_DEPTH_MIN, cfg.DITCH_DEPTH_MAX],
            [cfg.DITCH_SIGMA_MIN, cfg.DITCH_SIGMA_MAX],
        )
    )

    s_path = [0.0]
    for p0, p1 in zip(path_np[:-1], path_np[1:]):
        s_path.append(s_path[-1] + np.linalg.norm(p1 - p0))

    samples: list[dict] = []
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
            samples.append({"center": c, "nrm2": nrm2, "s": s_val, "dir2": dir2})
    if path_np:
        last = path_np[-1]
        if len(path_np) >= 2:
            prev = path_np[-2]
            dir_last = last - prev
            if np.linalg.norm(dir_last) > 1e-6:
                dir_last /= np.linalg.norm(dir_last)
        else:
            dir_last = np.array([0.0, 1.0], dtype=float)
        nrm_last = np.array([-dir_last[1], dir_last[0]], dtype=float)
        samples.append({"center": last, "nrm2": nrm_last, "s": s_path[-1], "dir2": dir_last})

    sample_info: list[dict] = []
    max_ditch_width = 0.0
    for sample in samples:
        c = sample["center"]
        nrm2 = sample["nrm2"]
        center_ground = base_height(c[0], c[1])
        center_surface = center_ground + road_height
        edge_height = center_surface - math.tan(cross_pitch) * half_road
        start_height = edge_height - curb_height
        width_needed = _estimate_required_ditch_width(
            terrain,
            base_height,
            c,
            nrm2,
            half_road,
            shoulder,
            ditch_width,
            start_height,
        )
        max_ditch_width = max(max_ditch_width, width_needed)
        sample_info.append(
            {
                "center": c,
                "nrm2": nrm2,
                "center_surface": center_surface,
                "ditch_width": width_needed,
                "dir2": sample["dir2"],
                "s": sample["s"],
            }
        )

    offsets = _build_offsets(half_road, shoulder, max_ditch_width)

    return {
        "samples": samples,
        "sample_info": sample_info,
        "offsets": offsets,
        "ditch_sigma": ditch_sigma,
        "curb_height": curb_height,
        "half_road": half_road,
        "shoulder": shoulder,
        "cross_pitch": cross_pitch,
        "along_step": along_step,
    }


class RoadSurface:
    """Collision helper that evaluates the procedural road surface."""

    def __init__(self, terrain, profile: dict):
        self.terrain = terrain
        self.sample_info = profile.get("sample_info", [])
        self.half_road = float(profile.get("half_road", 0.0))
        self.shoulder = float(profile.get("shoulder", 0.0))
        self.curb_height = float(profile.get("curb_height", cfg.CURB_HEIGHT))
        self.cross_pitch = float(profile.get("cross_pitch", 0.0))
        self.ditch_sigma = float(profile.get("ditch_sigma", cfg.DITCH_SIGMA_MIN))
        self.search_radius = max(float(profile.get("along_step", 0.5)) * 1.6, 1.5)
        offsets = profile.get("offsets", [0.0])
        if not offsets:
            offsets = [0.0]
        offsets = sorted(float(o) for o in offsets)
        self.offsets = np.array(offsets, dtype=float)
        self._max_offset = (
            float(max(abs(self.offsets[0]), abs(self.offsets[-1])))
            if len(self.offsets)
            else 0.0
        )
        outer_extent = self._max_offset + cfg.SKIRT_SAMPLE_SPACING
        self.search_radius = max(self.search_radius, outer_extent + 0.5)
        self._query_radius = self.search_radius + 0.5
        self._outer_extent = outer_extent
        self._outer_limit = outer_extent + 0.5
        self._query_radius2 = self._query_radius * self._query_radius
        self._cell_size = max(self.search_radius * 0.5, 1.0)
        base_mask = terrain.road_friction > 0
        self._road_mask = base_mask
        if base_mask.size:
            dilated = base_mask.copy()
            for dx in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dz == 0:
                        continue
                    rolled = np.roll(base_mask, shift=(dx, dz), axis=(0, 1))
                    if dx > 0:
                        rolled[:dx, :] = False
                    elif dx < 0:
                        rolled[dx:, :] = False
                    if dz > 0:
                        rolled[:, :dz] = False
                    elif dz < 0:
                        rolled[:, dz:] = False
                    dilated |= rolled
        else:
            dilated = base_mask
        self._road_mask_any = dilated
        self._offset_min = float(self.offsets[0]) if len(self.offsets) else 0.0
        self._offset_max = float(self.offsets[-1]) if len(self.offsets) else 0.0
        self._uniform_offsets = False
        self._offset_step = 0.0
        self._offset_inv_step = 0.0
        if len(self.offsets) >= 2:
            diffs = np.diff(self.offsets)
            first = float(diffs[0])
            if abs(first) > 1e-9 and np.allclose(diffs, first, atol=1e-6, rtol=1e-6):
                self._uniform_offsets = True
                self._offset_step = first
                self._offset_inv_step = 1.0 / first
        centers = (
            np.array([info["center"] for info in self.sample_info], dtype=float)
            if self.sample_info
            else np.empty((0, 2), dtype=float)
        )
        self._centers = centers
        self._last_sample: Optional[dict] = None
        self._reuse_radius2 = (self.search_radius * 0.6) ** 2
        grid: dict[tuple[int, int], list[int]] = {}
        if len(centers):
            for idx, info in enumerate(self.sample_info):
                cx, cz = info["center"]
                ix = int(math.floor(cx / self._cell_size))
                iz = int(math.floor(cz / self._cell_size))
                grid.setdefault((ix, iz), []).append(idx)
        self._grid = grid

        def base_height(px: float, pz: float) -> float:
            return float(self.terrain._height_from_grid(px, pz))

        for info in self.sample_info:
            heights: list[float] = []
            for off in self.offsets:
                h = _road_surface_height(
                    self.terrain,
                    base_height,
                    info,
                    off,
                    self.half_road,
                    self.shoulder,
                    self.curb_height,
                    self.cross_pitch,
                    self.ditch_sigma,
                    info["ditch_width"],
                )
                heights.append(float(h))
            info["heights"] = np.array(heights, dtype=float)

    def _height_from_sample(self, info: dict, offset: float) -> Optional[float]:
        offsets = self.offsets
        heights = info.get("heights")
        if heights is None or len(offsets) == 0:
            return None
        if self._uniform_offsets:
            if offset <= self._offset_min:
                return float(heights[0])
            if offset >= self._offset_max:
                return float(heights[-1])
            t = (offset - self._offset_min) * self._offset_inv_step
            idx = int(math.floor(t))
            if idx >= len(heights) - 1:
                return float(heights[-1])
            frac = t - idx
            h0 = heights[idx]
            h1 = heights[idx + 1]
            return float(h0 + (h1 - h0) * frac)

        if offset <= offsets[0]:
            return float(heights[0])
        if offset >= offsets[-1]:
            return float(heights[-1])
        idx = int(np.searchsorted(offsets, offset))
        if idx <= 0:
            return float(heights[0])
        if idx >= len(offsets):
            return float(heights[-1])
        left_off = offsets[idx - 1]
        right_off = offsets[idx]
        span = right_off - left_off
        if abs(span) < 1e-9:
            return float(heights[idx])
        t = (offset - left_off) / span
        left_h = heights[idx - 1]
        right_h = heights[idx]
        road_h = left_h + (right_h - left_h) * t
        return float(road_h)

    def _candidate_indices(self, x: float, z: float) -> list[int]:
        if not self._grid:
            return []
        cell_size = self._cell_size
        ix = int(math.floor(x / cell_size))
        iz = int(math.floor(z / cell_size))
        candidates: list[int] = []
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                bucket = self._grid.get((ix + dx, iz + dz))
                if bucket:
                    candidates.extend(bucket)
        return candidates

    def height_at(self, x: float, z: float) -> Optional[float]:
        if not self.sample_info:
            return None
        terrain = self.terrain
        if x < 0.0 or x > terrain.width or z < 0.0 or z > terrain.height:
            return None

        ix = min(max(int(x / terrain.cell_size_x), 0), terrain.res_x - 1)
        iz = min(max(int(z / terrain.cell_size_z), 0), terrain.res_z - 1)
        road_mask = self._road_mask_any
        if not road_mask[ix, iz]:
            return None

        best: Optional[dict] = None
        best_dist2 = self._query_radius2
        if self._last_sample is not None:
            dx_last = x - self._last_sample["center"][0]
            dz_last = z - self._last_sample["center"][1]
            dist2_last = dx_last * dx_last + dz_last * dz_last
            if dist2_last <= self._reuse_radius2:
                best = self._last_sample
                best_dist2 = dist2_last

        for idx in self._candidate_indices(x, z):
            info = self.sample_info[idx]
            dx = x - info["center"][0]
            dz = z - info["center"][1]
            dist2 = dx * dx + dz * dz
            if dist2 < best_dist2:
                best = info
                best_dist2 = dist2

        if best is None:
            return None

        dx = x - best["center"][0]
        dz = z - best["center"][1]
        offset = dx * best["nrm2"][0] + dz * best["nrm2"][1]
        if self._outer_extent > 0.0 and abs(offset) > self._outer_limit:
            return None
        road_h = self._height_from_sample(best, offset)
        if road_h is None or not math.isfinite(road_h):
            return None
        self._last_sample = best
        return float(road_h)


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
    ditch_width: Optional[float] = None,
    ditch_depth: Optional[float] = None,
    road_color=cfg.ROAD_COL,
    skirt_color=None,
    drive_line: Optional[List[Tuple[float, float]]] = None,
    **_: dict,
) -> dict[str, np.ndarray]:
    if ditch_width is None:
        ditch_width = cfg.DITCH_WIDTH_MAX
    if ditch_depth is None:
        ditch_depth = cfg.DITCH_DEPTH_MIN

    if skirt_color is None:
        skirt_color = road_color

    def base_height(x: float, z: float) -> float:
        return float(terrain.get_height(x, z, include_roads=False))

    cell = min(terrain.cell_size_x, terrain.cell_size_z)
    along_step = max(cell * cfg.ALONG_STEP_FACTOR, 0.35)
    half_road = 0.5 * lane_width * lanes
    half_plus_sh = half_road + shoulder

    path_np = [np.array(p, float) for p in path]
    profile = _collect_profile(
        terrain,
        path_np,
        half_road,
        shoulder,
        road_height,
        cross_pitch,
        ditch_width,
        ditch_depth,
        along_step,
    )

    samples = profile["samples"]
    sample_info = profile["sample_info"]
    offsets = profile["offsets"]
    ditch_sigma = profile["ditch_sigma"]
    curb_height = profile["curb_height"]

    # prepare lane-marking definitions
    lines_list: list[tuple[float, list[float], bool]] = []
    left_edge_off = -half_road
    right_edge_off = half_road
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

    period = cfg.DASH_LENGTH + cfg.GAP_LENGTH
    line_half = cfg.LINE_WIDTH * 0.5

    rings: list[list[list[float]]] = []
    ring_heights: list[np.ndarray] = []
    for info in sample_info:
        ring: list[list[float]] = []
        heights: list[float] = []
        for off in offsets:
            x = info["center"][0] + info["nrm2"][0] * off
            z = info["center"][1] + info["nrm2"][1] * off
            y = _road_surface_height(
                terrain,
                base_height,
                info,
                off,
                half_road,
                shoulder,
                curb_height,
                cross_pitch,
                ditch_sigma,
                info["ditch_width"],
            )
            ring.append([x, y, z])
            heights.append(float(y))
        rings.append(ring)
        ring_heights.append(np.array(heights, dtype=float))

    deck_vertices: list[float] = []
    skirt_vertices: list[float] = []

    def emit_quad(bucket: list[float], a, b, c, d, cols):
        col_a, col_b, col_c, col_d = (list(cols[0]), list(cols[1]), list(cols[2]), list(cols[3]))
        bucket.extend(a + col_a)
        bucket.extend(b + col_b)
        bucket.extend(c + col_c)
        bucket.extend(c + col_c)
        bucket.extend(b + col_b)
        bucket.extend(d + col_d)

    road_col = list(road_color)
    skirt_col = list(skirt_color)
    shoulder_col = [
        float(0.5 * (rc + sc)) for rc, sc in zip(road_col[:3], skirt_col[:3])
    ]
    if len(road_col) > 3:
        shoulder_alpha = float(0.5 * (road_col[3] + skirt_col[3]))
        shoulder_col.append(shoulder_alpha)
    else:
        shoulder_col.append(1.0)
    if len(shoulder_col) < 4:
        shoulder_col.extend([1.0] * (4 - len(shoulder_col)))
    deck_limit = half_road + shoulder + 1e-6

    for i in range(len(rings) - 1):
        r0, r1 = rings[i], rings[i + 1]
        for j in range(len(r0) - 1):
            off_a = abs(offsets[j])
            off_b = abs(offsets[j + 1])
            if off_a <= deck_limit or off_b <= deck_limit:
                in_lane = (
                    off_a <= half_road + 1e-6 and off_b <= half_road + 1e-6
                )
                if in_lane:
                    col0 = road_col
                    col1 = road_col
                    col2 = road_col
                    col3 = road_col
                else:
                    col0 = shoulder_col
                    col1 = shoulder_col
                    col2 = shoulder_col
                    col3 = shoulder_col
                emit_quad(
                    deck_vertices,
                    r0[j],
                    r0[j + 1],
                    r1[j],
                    r1[j + 1],
                    (col0, col1, col2, col3),
                )
            else:
                emit_quad(
                    skirt_vertices,
                    r0[j],
                    r0[j + 1],
                    r1[j],
                    r1[j + 1],
                    (skirt_col, skirt_col, skirt_col, skirt_col),
                )

    def _line_height(surface_y: float, offset: float) -> float:
        return surface_y + cfg.LINE_EPS

    offsets_np = np.array(offsets, dtype=float)
    uniform_offsets = False
    offset_min = 0.0
    offset_inv_step = 0.0
    if len(offsets_np) >= 2:
        diffs = np.diff(offsets_np)
        step = float(diffs[0])
        if abs(step) > 1e-9 and np.allclose(diffs, step, atol=1e-6, rtol=1e-6):
            uniform_offsets = True
            offset_min = float(offsets_np[0])
            offset_inv_step = 1.0 / step

    def _sample_cached_height(idx: int, offset: float) -> float:
        heights_arr = ring_heights[idx]
        if len(heights_arr) == 0:
            return 0.0
        if uniform_offsets:
            if offset <= offsets_np[0]:
                return float(heights_arr[0])
            if offset >= offsets_np[-1]:
                return float(heights_arr[-1])
            t = (offset - offset_min) * offset_inv_step
            base_idx = int(math.floor(t))
            if base_idx >= len(heights_arr) - 1:
                return float(heights_arr[-1])
            frac = t - base_idx
            h0 = heights_arr[base_idx]
            h1 = heights_arr[base_idx + 1]
            return float(h0 + (h1 - h0) * frac)
        return float(np.interp(offset, offsets_np, heights_arr))

    line_vertices: list[float] = []
    driveline_vertices: list[float] = []
    for off, col, dotted in lines_list:
        prev_pair = None
        prev_emit = False
        for idx, (sample, info) in enumerate(zip(samples, sample_info)):
            c = sample["center"]
            nrm2 = sample["nrm2"]
            s_val = sample["s"]
            emit = True
            if dotted:
                emit = (s_val % period) < cfg.DASH_LENGTH
            left_off = off - line_half
            right_off = off + line_half
            left_y = _sample_cached_height(idx, left_off)
            right_y = _sample_cached_height(idx, right_off)
            a = [
                c[0] + nrm2[0] * left_off,
                _line_height(left_y, left_off),
                c[1] + nrm2[1] * left_off,
            ]
            b = [
                c[0] + nrm2[0] * right_off,
                _line_height(right_y, right_off),
                c[1] + nrm2[1] * right_off,
            ]
            this_pair = [a, b]
            if prev_pair is not None and emit and prev_emit:
                emit_quad(
                    line_vertices,
                    prev_pair[0],
                    prev_pair[1],
                    this_pair[0],
                    this_pair[1],
                    (col, col, col, col),
                )
            prev_pair = this_pair
            prev_emit = emit

    # green driveline extrusion
    road_surface_query: Optional[RoadSurface] = None
    if drive_line:
        road_surface_query = getattr(terrain, "road_surface", None)
        if road_surface_query is None:
            road_surface_query = RoadSurface(terrain, profile)
        half_dl = cfg.DRIVE_LINE_WIDTH * 0.5
        green = [0.0, 1.0, 0.0, 1.0]
        prev_pair = None
        for p0, p1 in zip(drive_line[:-1], drive_line[1:]):
            h0 = road_surface_query.height_at(p0[0], p0[1]) if road_surface_query else None
            h1 = road_surface_query.height_at(p1[0], p1[1]) if road_surface_query else None
            if h0 is None or h1 is None:
                continue
            h0 = h0 + cfg.DRIVE_LINE_HEIGHT
            h1 = h1 + cfg.DRIVE_LINE_HEIGHT
            a0 = np.array([p0[0], h0, p0[1]])
            a1 = np.array([p1[0], h1, p1[1]])
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
                emit_quad(
                    driveline_vertices,
                    prev_pair[0],
                    prev_pair[1],
                    this_pair[0],
                    this_pair[1],
                    (green, green, green, green),
                )
            prev_pair = this_pair

    def _to_array(values: list[float]) -> np.ndarray:
        if not values:
            return np.zeros(0, dtype="f4")
        return np.array(values, dtype="f4")

    return {
        "deck": _to_array(deck_vertices),
        "skirt": _to_array(skirt_vertices),
        "lines": _to_array(line_vertices),
        "driveline": _to_array(driveline_vertices),
    }


def build_speed_sign_vertices(
    terrain,
    path: List[Tuple[float, float]],
    lane_width: float,
    lanes: int,
    shoulder: float,
    speed_limits: Optional[List[dict]] = None,
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


def apply_plan(terrain, path: List[Tuple[float, float]], params: dict, rng: Optional[np.random.Generator] = None) -> None:
    """Apply non-destructive road data to the terrain."""

    terrain.road_surface = None
    if not path:
        return

    road_friction = float(params.get("road_friction", 0.0))
    if road_friction <= 0.0:
        return

    lane_width = params.get("lane_width", cfg.LANE_WIDTH_MIN)
    lanes = params.get("lanes", 1)
    shoulder = params.get("shoulder", 0.0)
    road_height = float(params.get("road_height", 0.0))
    cross_pitch = float(params.get("cross_pitch", 0.0))
    ditch_width = float(params.get("ditch_width", cfg.DITCH_WIDTH_MAX))
    ditch_depth = float(params.get("ditch_depth", cfg.DITCH_DEPTH_MIN))

    half_span = 0.5 * lane_width * lanes + shoulder
    if half_span <= 0.0:
        return

    cell = min(terrain.cell_size_x, terrain.cell_size_z)
    along_step = max(cell * cfg.ALONG_STEP_FACTOR, 0.35)
    offsets = np.linspace(-half_span, half_span, max(6, lanes * 3 + 3))
    path_np = [np.array(p, float) for p in path]

    profile = _collect_profile(
        terrain,
        path_np,
        0.5 * lane_width * lanes,
        shoulder,
        road_height,
        cross_pitch,
        ditch_width,
        ditch_depth,
        along_step,
    )

    for sample in profile["samples"]:
        c = sample["center"]
        nrm2 = sample["nrm2"]
        for off in offsets:
            x = c[0] + nrm2[0] * off
            z = c[1] + nrm2[1] * off
            if x < 0 or x > terrain.width or z < 0 or z > terrain.height:
                continue
            ix = min(max(int(x / terrain.cell_size_x), 0), terrain.res_x - 1)
            iz = min(max(int(z / terrain.cell_size_z), 0), terrain.res_z - 1)
            terrain.road_friction[ix, iz] = road_friction

    terrain.road_surface = RoadSurface(terrain, profile)

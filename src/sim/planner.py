"""Lightweight helpers for generating planner previews for controllers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .control_api import FuturePreview, PlannerTarget

_MPH_TO_MPS = 0.44704


@dataclass
class PlannerConfig:
    """Configuration for planner previews.

    Examples
    --------
    >>> cfg = PlannerConfig(preview_hz=10.0, horizon_seconds=5.0)
    >>> cfg.preview_hz
    10.0
    """

    preview_hz: float = 10.0
    horizon_seconds: float = 5.0
    sample_spacing_m: float = 0.25
    spline_tension: float = 0.0


class PlannerPreviewer:
    """Project vehicle state onto the driveline and emit fixed-rate previews."""

    def __init__(self, config: PlannerConfig | None = None) -> None:
        self.config = config or PlannerConfig()
        self.control_positions = np.zeros((0, 2), dtype=float)
        self.positions = np.zeros((0, 2), dtype=float)
        self.s = np.zeros(0, dtype=float)
        self.tangents = np.zeros((0, 2), dtype=float)
        self.curvature = np.zeros(0, dtype=float)
        self.speed_limits: List[dict] = []
        self._last_projection: Tuple[float, float, float] | None = None
        self._last_index: int | None = None

    def set_plan(self, drive_line: Sequence[Sequence[float]] | None, speed_limits: Iterable[dict] | None) -> None:
        if not drive_line:
            self.control_positions = np.zeros((0, 2), dtype=float)
            self.positions = np.zeros((0, 2), dtype=float)
            self.s = np.zeros(0, dtype=float)
            self.tangents = np.zeros((0, 2), dtype=float)
            self.curvature = np.zeros(0, dtype=float)
            self.speed_limits = []
            self._last_projection = None
            self._last_index = None
            return
        pts = np.array([[p[0], p[1]] for p in drive_line], dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 2:
            raise ValueError("drive_line must contain at least two points")
        self.control_positions = pts
        sampled, sampled_tangents, _sampled_curvature = self._sample_smooth_path(pts)
        self.positions = sampled
        diffs = np.diff(sampled, axis=0)
        ds = np.linalg.norm(diffs, axis=1)
        self.s = np.concatenate(([0.0], np.cumsum(ds)))
        self.tangents = sampled_tangents
        self.curvature = self._compute_curvature(sampled)
        self.speed_limits = list(speed_limits or [])
        self._last_projection = None
        self._last_index = None

    # ------------------------------------------------------------------

    def preview(
        self,
        position: Sequence[float],
        speed: float,
        preview_hz: float | None = None,
    ) -> FuturePreview:
        if self.positions.shape[0] < 2:
            dt = 1.0 / (preview_hz or self.config.preview_hz)
            self._last_projection = None
            return FuturePreview.empty(dt)
        hz = preview_hz or self.config.preview_hz
        hz = max(hz, 1e-3)
        dt = 1.0 / hz
        steps = int(round(self.config.horizon_seconds * hz))
        if steps <= 0:
            self._last_projection = None
            return FuturePreview.empty(dt)

        s_now, lateral_error, path_heading = self._project(position)
        self._last_projection = (s_now, lateral_error, path_heading)
        lat_list: list[float] = []
        roll_lat_list: list[float] = []
        speed_list: list[float] = []
        long_list: list[float] = []
        prev_speed = float(speed)
        cur_s = s_now
        for _ in range(steps):
            target_speed = self._speed_limit_at(cur_s)
            kappa = self._curvature_at(cur_s)
            lat = target_speed * target_speed * kappa
            lat_list.append(float(lat))
            roll_lat_list.append(0.0)  # Road banking is not yet modelled explicitly
            speed_list.append(float(target_speed))
            long_accel = (target_speed - prev_speed) * hz
            long_list.append(float(long_accel))
            prev_speed = target_speed
            cur_s += max(target_speed, 0.0) * dt

        return FuturePreview(
            lat_accel=tuple(lat_list),
            roll_lataccel=tuple(roll_lat_list),
            speed=tuple(speed_list),
            long_accel=tuple(long_list),
            dt=dt,
        )

    def immediate_target(
        self,
        position: Sequence[float],
        speed: float,
        heading: float,
        preview: FuturePreview | None = None,
    ) -> PlannerTarget:
        if preview is None:
            preview = self.preview(position, speed, preview_hz=None)
        if preview.speed:
            proj = self._last_projection
            if proj is None:
                proj = self._project(position)
            _, lateral_error, path_heading = proj
            heading_error = _wrap_angle(path_heading - heading)
            return PlannerTarget(
                speed=preview.speed[0],
                lat_accel=preview.lat_accel[0] if preview.lat_accel else 0.0,
                long_accel=preview.long_accel[0] if preview.long_accel else 0.0,
                roll_lataccel=preview.roll_lataccel[0] if preview.roll_lataccel else 0.0,
                lateral_error=lateral_error,
                heading_error=heading_error,
            )
        return PlannerTarget()

    # ------------------------------------------------------------------

    def _project(self, position: Sequence[float]) -> Tuple[float, float, float]:
        pts = self.positions
        if pts.shape[0] < 2:
            return 0.0, 0.0, 0.0
        pos = np.array([position[0], position[2] if len(position) > 2 else position[1]], dtype=float)

        if self._last_projection is not None and self.s.size >= 2:
            s_guess, _, _ = self._last_projection
            point_guess, tangent_guess, _ = self._sample_path(float(s_guess))
            disp = pos - point_guess
            ds = float(np.dot(disp, tangent_guess))
            s_candidate = float(np.clip(s_guess + ds, self.s[0], self.s[-1]))
            point_cand, tangent_cand, idx_cand = self._sample_path(s_candidate)
            normal_cand = np.array([-tangent_cand[1], tangent_cand[0]])
            lateral_error = float(np.dot(pos - point_cand, normal_cand))
            separation = float(np.linalg.norm(pos - point_cand))
            if separation <= max(25.0, abs(lateral_error) + 5.0):
                self._last_index = idx_cand
                path_heading = self._heading_at(s_candidate)
                return s_candidate, lateral_error, path_heading

        segs = pts[1:] - pts[:-1]
        seg_len_sq = np.sum(segs * segs, axis=1) + 1e-9

        def _project_range(start: int, end: int) -> tuple[int, np.ndarray, float, float]:
            rel_local = pos - pts[start:end]
            seg_local = segs[start:end]
            seg_len_local = seg_len_sq[start:end]
            t_local = np.clip(
                np.sum(rel_local * seg_local, axis=1) / seg_len_local,
                0.0,
                1.0,
            )
            proj_local = pts[start:end] + seg_local * t_local[:, None]
            dists_local = np.linalg.norm(pos - proj_local, axis=1)
            idx_local = int(np.argmin(dists_local))
            idx_global = start + idx_local
            return idx_global, proj_local[idx_local], t_local[idx_local], dists_local[idx_local]

        idx = None
        proj_point = None
        t_val = 0.0
        if self._last_index is not None:
            approx_idx = max(0, min(int(self._last_index), segs.shape[0] - 1))
            window = 6
            start = max(0, approx_idx - window)
            end = min(segs.shape[0], approx_idx + window + 1)
            if end - start > 0:
                idx, proj_point, t_val, dist = _project_range(start, end)
                if dist > 25.0:
                    idx = None

        if idx is None:
            rel = pos - pts[:-1]
            t = np.clip(np.sum(rel * segs, axis=1) / seg_len_sq, 0.0, 1.0)
            proj = pts[:-1] + segs * t[:, None]
            dists = np.linalg.norm(pos - proj, axis=1)
            idx = int(np.argmin(dists))
            proj_point = proj[idx]
            t_val = float(t[idx])

        self._last_index = idx
        s_val = float(self.s[idx] + t_val * (self.s[idx + 1] - self.s[idx]))
        tangent_norm = self._tangent_at(s_val)
        normal = np.array([-tangent_norm[1], tangent_norm[0]])
        lateral_error = float(np.dot(pos - proj_point, normal))
        path_heading = self._heading_at(s_val)
        return s_val, lateral_error, path_heading

    def _sample_path(self, s: float) -> tuple[np.ndarray, np.ndarray, int]:
        if self.positions.shape[0] < 2:
            return np.zeros(2, dtype=float), np.array([1.0, 0.0], dtype=float), 0
        s_clamped = float(np.clip(s, self.s[0], self.s[-1]))
        idx = int(np.searchsorted(self.s, s_clamped, side="right"))
        idx = max(1, min(idx, self.s.size - 1))
        seg_idx = idx - 1
        p0 = self.positions[seg_idx]
        p1 = self.positions[seg_idx + 1]
        seg = p1 - p0
        seg_len = np.linalg.norm(seg)
        tangent = seg / (seg_len + 1e-9)
        seg_s0 = self.s[seg_idx]
        seg_s1 = self.s[seg_idx + 1]
        if seg_s1 <= seg_s0:
            frac = 0.0
        else:
            frac = float((s_clamped - seg_s0) / (seg_s1 - seg_s0))
        point = p0 + seg * frac
        return point, self._tangent_at(s_clamped), seg_idx

    def _tangent_at(self, s: float) -> np.ndarray:
        if self.positions.shape[0] < 2 or self.tangents.shape[0] == 0:
            return np.array([1.0, 0.0], dtype=float)
        s_clamped = float(np.clip(s, self.s[0], self.s[-1]))
        idx = int(np.searchsorted(self.s, s_clamped, side="right"))
        idx = max(1, min(idx, self.s.size - 1))
        lo = idx - 1
        hi = idx
        s_lo = self.s[lo]
        s_hi = self.s[hi]
        if s_hi <= s_lo:
            tangent = self.tangents[hi].copy()
        else:
            frac = float((s_clamped - s_lo) / (s_hi - s_lo))
            tangent = (1.0 - frac) * self.tangents[lo] + frac * self.tangents[hi]
        norm = np.linalg.norm(tangent)
        if norm < 1e-9:
            return self.tangents[hi].copy()
        return tangent / norm

    def _heading_at(self, s: float) -> float:
        tangent = self._tangent_at(s)
        return float(math.atan2(tangent[0], tangent[1]))

    def _compute_tangents(self, pts: np.ndarray, s: np.ndarray) -> np.ndarray:
        n = pts.shape[0]
        tangents = np.zeros_like(pts)
        if n < 2:
            return tangents
        tangents[0] = pts[1] - pts[0]
        tangents[-1] = pts[-1] - pts[-2]
        for i in range(1, n - 1):
            ds = max(s[i + 1] - s[i - 1], 1e-9)
            tangents[i] = (pts[i + 1] - pts[i - 1]) / ds
        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        tangents = tangents / np.maximum(norms, 1e-9)
        return tangents

    def _sample_smooth_path(self, control_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if control_pts.shape[0] < 3:
            diffs = np.diff(control_pts, axis=0)
            if diffs.shape[0] == 0:
                tangents = np.array([[1.0, 0.0]], dtype=float)
            else:
                tangents = np.vstack((diffs[0], diffs))
                norms = np.linalg.norm(tangents, axis=1, keepdims=True)
                tangents = tangents / np.maximum(norms, 1e-9)
            curvature = np.zeros(control_pts.shape[0], dtype=float)
            return control_pts.copy(), tangents, curvature

        n = control_pts.shape[0]
        deriv = np.zeros_like(control_pts)
        tension = max(0.0, min(float(self.config.spline_tension), 1.0))
        scale = 1.0 - tension
        deriv[0] = scale * 0.5 * (-3.0 * control_pts[0] + 4.0 * control_pts[1] - control_pts[2])
        deriv[-1] = scale * 0.5 * (3.0 * control_pts[-1] - 4.0 * control_pts[-2] + control_pts[-3])
        for i in range(1, n - 1):
            deriv[i] = scale * 0.5 * (control_pts[i + 1] - control_pts[i - 1])

        spacing = max(float(self.config.sample_spacing_m), 0.05)
        samples: list[np.ndarray] = [control_pts[0]]
        tangents: list[np.ndarray] = []
        curvature: list[float] = []
        for i in range(n - 1):
            p0 = control_pts[i]
            p1 = control_pts[i + 1]
            m0 = deriv[i]
            m1 = deriv[i + 1]
            seg_len = float(np.linalg.norm(p1 - p0))
            steps = max(2, int(math.ceil(seg_len / spacing)))
            start = 0 if i == 0 else 1
            for j in range(start, steps + 1):
                t = j / steps
                h00 = 2.0 * t * t * t - 3.0 * t * t + 1.0
                h10 = t * t * t - 2.0 * t * t + t
                h01 = -2.0 * t * t * t + 3.0 * t * t
                h11 = t * t * t - t * t
                pos = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
                dp = (
                    (6.0 * t * t - 6.0 * t) * p0
                    + (3.0 * t * t - 4.0 * t + 1.0) * m0
                    + (-6.0 * t * t + 6.0 * t) * p1
                    + (3.0 * t * t - 2.0 * t) * m1
                )
                ddp = (
                    (12.0 * t - 6.0) * p0
                    + (6.0 * t - 4.0) * m0
                    + (-12.0 * t + 6.0) * p1
                    + (6.0 * t - 2.0) * m1
                )
                dp_norm = float(np.linalg.norm(dp))
                tangent = dp / max(dp_norm, 1e-9)
                cross = dp[0] * ddp[1] - dp[1] * ddp[0]
                kappa = float(cross / max(dp_norm ** 3, 1e-9))
                samples.append(pos)
                tangents.append(tangent)
                curvature.append(kappa)

        sampled = np.array(samples, dtype=float)
        tangent_arr = np.array(tangents, dtype=float)
        curvature_arr = np.array(curvature, dtype=float)
        dedup_pos = [sampled[0]]
        dedup_tan = [tangent_arr[0]]
        dedup_curv = [curvature_arr[0]]
        for idx, pt in enumerate(sampled[1:], start=1):
            if np.linalg.norm(pt - dedup_pos[-1]) > 1e-9:
                dedup_pos.append(pt)
                dedup_tan.append(tangent_arr[idx - 1])
                dedup_curv.append(curvature_arr[idx - 1])
        return (
            np.array(dedup_pos, dtype=float),
            np.array(dedup_tan, dtype=float),
            np.array(dedup_curv, dtype=float),
        )

    def _compute_curvature(self, pts: np.ndarray) -> np.ndarray:
        n = pts.shape[0]
        curv = np.zeros(n, dtype=float)
        if n < 3:
            return curv
        step = max(1, min(3, (n - 1) // 8))
        for i in range(n):
            if i < step:
                i0, i1, i2 = 0, min(step, n - 2), min(2 * step, n - 1)
            elif i > n - 1 - step:
                i0, i1, i2 = max(n - 1 - 2 * step, 0), max(n - 1 - step, 1), n - 1
            else:
                i0, i1, i2 = i - step, i, i + step
            p0, p1, p2 = pts[i0], pts[i1], pts[i2]
            a = p1 - p0
            b = p2 - p1
            c = p2 - p0
            la = np.linalg.norm(a)
            lb = np.linalg.norm(b)
            lc = np.linalg.norm(c)
            denom = la * lb * lc
            if denom < 1e-6:
                curv[i] = 0.0
                continue
            cross = a[0] * b[1] - a[1] * b[0]
            curv[i] = 2.0 * cross / denom
        return curv

    def _curvature_at(self, s: float) -> float:
        if self.s.size == 0:
            return 0.0
        s_clamped = min(max(s, self.s[0]), self.s[-1])
        margin = min(self.s[-1], max(self.config.sample_spacing_m * 4.0, 0.0))
        if self.s[-1] > 2.0 * margin:
            s_clamped = min(max(s_clamped, margin), self.s[-1] - margin)
        return float(np.interp(s_clamped, self.s, self.curvature))

    def _speed_limit_at(self, s: float) -> float:
        if not self.speed_limits:
            return 0.0
        s_clamped = min(max(s, 0.0), self.s[-1] if self.s.size else s)
        mph = self.speed_limits[-1].get("speed_mph", 0.0)
        for seg in self.speed_limits:
            if seg["start_s"] <= s_clamped <= seg["end_s"]:
                mph = seg.get("speed_mph", mph)
                break
        return float(mph) * _MPH_TO_MPS


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


__all__ = ["PlannerConfig", "PlannerPreviewer"]

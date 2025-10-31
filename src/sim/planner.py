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


class PlannerPreviewer:
    """Project vehicle state onto the driveline and emit fixed-rate previews."""

    def __init__(self, config: PlannerConfig | None = None) -> None:
        self.config = config or PlannerConfig()
        self.positions = np.zeros((0, 2), dtype=float)
        self.s = np.zeros(0, dtype=float)
        self.curvature = np.zeros(0, dtype=float)
        self.speed_limits: List[dict] = []
        self._last_projection: Tuple[float, float, float] | None = None

    def set_plan(self, drive_line: Sequence[Sequence[float]] | None, speed_limits: Iterable[dict] | None) -> None:
        if not drive_line:
            self.positions = np.zeros((0, 2), dtype=float)
            self.s = np.zeros(0, dtype=float)
            self.curvature = np.zeros(0, dtype=float)
            self.speed_limits = []
            self._last_projection = None
            return
        pts = np.array([[p[0], p[1]] for p in drive_line], dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 2:
            raise ValueError("drive_line must contain at least two points")
        self.positions = pts
        diffs = np.diff(pts, axis=0)
        ds = np.linalg.norm(diffs, axis=1)
        self.s = np.concatenate(([0.0], np.cumsum(ds)))
        self.curvature = self._compute_curvature(pts)
        self.speed_limits = list(speed_limits or [])
        self._last_projection = None

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
        segs = pts[1:] - pts[:-1]
        seg_len_sq = np.sum(segs * segs, axis=1) + 1e-9
        rel = pos - pts[:-1]
        t = np.clip(np.sum(rel * segs, axis=1) / seg_len_sq, 0.0, 1.0)
        proj = pts[:-1] + segs * t[:, None]
        dists = np.linalg.norm(pos - proj, axis=1)
        idx = int(np.argmin(dists))
        s_val = float(self.s[idx] + t[idx] * (self.s[idx + 1] - self.s[idx]))
        tangent = segs[idx]
        tangent_norm = tangent / (np.linalg.norm(tangent) + 1e-9)
        normal = np.array([-tangent_norm[1], tangent_norm[0]])
        lateral_error = float(np.dot(pos - proj[idx], normal))
        path_heading = float(math.atan2(tangent_norm[1], tangent_norm[0]))
        return s_val, lateral_error, path_heading

    def _compute_curvature(self, pts: np.ndarray) -> np.ndarray:
        n = pts.shape[0]
        curv = np.zeros(n, dtype=float)
        if n < 3:
            return curv
        for i in range(1, n - 1):
            p0, p1, p2 = pts[i - 1], pts[i], pts[i + 1]
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
        curv[0] = curv[1]
        curv[-1] = curv[-2]
        return curv

    def _curvature_at(self, s: float) -> float:
        if self.s.size == 0:
            return 0.0
        s_clamped = min(max(s, self.s[0]), self.s[-1])
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

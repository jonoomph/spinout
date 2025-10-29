# effects.py
from __future__ import annotations

from collections import deque
from typing import Dict, Iterable

import numpy as np


class SkidMarkSystem:
    """Accumulate persistent skid mark geometry for rendering."""

    def __init__(self, max_segments: int = 4800) -> None:
        self.max_segments = max_segments
        self._segments: deque[np.ndarray] = deque(maxlen=max_segments)
        self._active: Dict[int, Dict] = {}
        self._vertex_cache = np.zeros((0, 7), dtype="f4")
        self._dirty = True
        self._layer_cursor = 0

    def reset(self) -> None:
        self._segments.clear()
        self._active.clear()
        self._vertex_cache = np.zeros((0, 7), dtype="f4")
        self._dirty = True
        self._layer_cursor = 0

    def step(self, dt: float, events: Iterable[Dict]) -> None:
        if dt <= 0.0:
            return

        stale = [idx for idx, state in self._active.items() if state.get("age", 0.0) > 1.2]
        for idx in stale:
            self._active.pop(idx, None)

        for state in self._active.values():
            state["age"] = state.get("age", 0.0) + dt

        for event in events:
            idx = int(event.get("index", -1))
            if idx < 0:
                continue
            intensity = float(np.clip(event.get("intensity", 0.0), 0.0, 1.0))
            if intensity <= 0.0:
                self._active.pop(idx, None)
                continue

            center = np.array(event.get("position", (0.0, 0.0, 0.0)), dtype="f4")
            right = np.array(event.get("right", (1.0, 0.0, 0.0)), dtype="f4")
            norm = np.linalg.norm(right)
            if norm <= 1e-6:
                self._active.pop(idx, None)
                continue
            right /= norm

            width = float(event.get("width", 0.25))
            half_width = width * 0.5
            left_pt = center - right * half_width
            right_pt = center + right * half_width

            # Slightly layer marks above the terrain to avoid z-fighting when
            # several streaks overlap. We offset subsequent streaks by a few
            # fractions of a millimetre so newer marks render cleanly.
            layer = 0.0006 * (self._layer_cursor % 11)
            self._layer_cursor = (self._layer_cursor + 1) % 1024
            lift = 0.0025 + layer
            left_pt[1] += lift
            right_pt[1] += lift
            center[1] += lift

            base_rgb = np.array(event.get("base_color", (0.06, 0.06, 0.06)), dtype="f4")
            rgba = self._shade_color(base_rgb, intensity)

            prev = self._active.get(idx)
            if prev is not None and prev.get("age", 0.0) < 0.35:
                dist = np.linalg.norm(center - prev["center"])
                if dist < 4.0:
                    prev_left = prev["left"]
                    prev_right = prev["right"]
                    prev_rgba = prev["rgba"]
                    seg = np.array(
                        [
                            list(prev_left) + list(prev_rgba),
                            list(prev_right) + list(prev_rgba),
                            list(right_pt) + list(rgba),
                            list(prev_left) + list(prev_rgba),
                            list(right_pt) + list(rgba),
                            list(left_pt) + list(rgba),
                        ],
                        dtype="f4",
                    )
                    self._segments.append(seg)
                    self._dirty = True

            self._active[idx] = {
                "left": left_pt,
                "right": right_pt,
                "center": center,
                "rgba": rgba,
                "age": 0.0,
            }

    def _shade_color(self, base_rgb: np.ndarray, intensity: float) -> np.ndarray:
        base = np.clip(base_rgb, 0.0, 1.0)
        dark = np.clip(base * (0.35 + 0.25 * (1.0 - intensity)) + 0.02 * intensity, 0.0, 0.45)
        alpha = np.clip(0.18 + 0.6 * intensity, 0.1, 0.85)
        return np.array([dark[0], dark[1], dark[2], alpha], dtype="f4")

    def get_vertices(self) -> np.ndarray:
        if self._dirty:
            if self._segments:
                self._vertex_cache = np.vstack(list(self._segments)).astype("f4")
            else:
                self._vertex_cache = np.zeros((0, 7), dtype="f4")
            self._dirty = False
        return self._vertex_cache

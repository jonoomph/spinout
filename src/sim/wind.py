"""Wind system providing a single fixed wind sample per environment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

_COMPASS16 = (
    "N",
    "NNE",
    "NE",
    "ENE",
    "E",
    "ESE",
    "SE",
    "SSE",
    "S",
    "SSW",
    "SW",
    "WSW",
    "W",
    "WNW",
    "NW",
    "NNW",
)


@dataclass
class WindSample:
    """Snapshot of the current wind state."""

    direction_deg: float
    speed_mps: float
    vector: np.ndarray
    normalized_strength: float
    max_speed: float

    @property
    def speed_mph(self) -> float:
        return float(self.speed_mps * 2.23693629)

    @property
    def compass_label(self) -> str:
        idx = int((self.direction_deg / 22.5) + 0.5) % len(_COMPASS16)
        return _COMPASS16[idx]


class WindSystem:
    """Generates a single wind vector for the simulator."""

    _STATE_SPEEDS = {
        "dry": (0.0, 3.2),
        "wet": (0.6, 5.8),
        "rain": (1.4, 7.2),
    }

    def __init__(
        self,
        rng: np.random.Generator,
        weather: str,
        precipitation: str,
        calm: bool = False,
    ) -> None:
        self._rng = rng
        self._state = self._resolve_state(weather, precipitation)
        lo, hi = self._STATE_SPEEDS[self._state]
        if calm:
            lo, hi = 0.0, 0.0
        self._speed_range: Tuple[float, float] = (float(lo), float(hi))
        self.max_speed = float(hi)

        self._direction = float(self._rng.uniform(0.0, 360.0))
        self._speed = self._pick_speed()
        self._sample = self._build_sample()

    @staticmethod
    def _resolve_state(weather: str, precipitation: str) -> str:
        if precipitation == "rain":
            return "rain"
        if weather == "wet":
            return "wet"
        return "dry"

    def _pick_speed(self) -> float:
        lo, hi = self._speed_range
        if hi <= lo:
            return lo
        span = hi - lo
        if self._state == "rain":
            r = float(self._rng.random() ** 0.75)
        elif self._state == "wet":
            r = float(self._rng.random() ** 0.95)
        else:
            r = float(self._rng.random() ** 1.55)
        return lo + span * r

    def _build_sample(self) -> WindSample:
        speed = max(0.0, float(self._speed))
        direction = self._direction % 360.0
        rad = math.radians(direction)
        vx = math.sin(rad) * speed
        vz = math.cos(rad) * speed
        vec = np.array([vx, 0.0, vz], dtype=float)
        norm = 0.0
        if self.max_speed > 1e-5:
            norm = float(np.clip(speed / max(self.max_speed, 1e-5), 0.0, 1.0))
        return WindSample(
            direction_deg=direction,
            speed_mps=speed,
            vector=vec,
            normalized_strength=norm,
            max_speed=self.max_speed,
        )

    def update(self, dt: float) -> WindSample:
        return self._sample

    @property
    def direction_deg(self) -> float:
        return self._sample.direction_deg

    @property
    def speed_mps(self) -> float:
        return self._sample.speed_mps

    @property
    def normalized_strength(self) -> float:
        return self._sample.normalized_strength

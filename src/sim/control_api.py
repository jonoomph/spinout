"""Control and telemetry data structures shared by the simulator and controllers.

The classes in this module are intentionally tiny – they simply group the values
that the environment exposes each step.  Controllers receive a
:class:`TelemetrySnapshot` and return a :class:`DriverCommand` which the
environment then applies to the car.

Examples
--------
>>> snapshot = TelemetrySnapshot()
>>> snapshot.state.speed
0.0
>>> DriverCommand(steer=0.2, throttle=0.5).as_dict()
{'steer': 0.2, 'throttle': 0.5, 'brake': 0.0}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence, Tuple


@dataclass
class DriverCommand:
    """Container for steering, throttle and brake commands.

    The values are the *normalised* commands expected by the physics layer, so
    steering lives in ``[-1, 1]`` while throttle and brake are clamped to
    ``[0, 1]``.
    """

    steer: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0

    def clipped(self) -> "DriverCommand":
        """Return a copy clamped to the valid control range.

        Examples
        --------
        >>> DriverCommand(steer=2.0, throttle=-1.0, brake=2.0).clipped()
        DriverCommand(steer=1.0, throttle=0.0, brake=1.0)
        """

        return DriverCommand(
            steer=max(-1.0, min(1.0, self.steer)),
            throttle=max(0.0, min(1.0, self.throttle)),
            brake=max(0.0, min(1.0, self.brake)),
        )

    def as_dict(self) -> Dict[str, float]:
        """Return a mapping compatible with the legacy action format."""

        return {"steer": self.steer, "accel": self.throttle, "brake": self.brake}

    @classmethod
    def from_action(cls, action: Dict[str, float] | None) -> "DriverCommand":
        """Build a command from the legacy mapping or a compatible object.

        ``game.py`` used to pass dictionaries with ``{"steer", "accel", "brake"}``
        entries. Some newer call sites use ``throttle`` instead of ``accel``.
        Others pass :class:`DriverCommand` instances or small containers with
        ``steer``, ``throttle`` and ``brake`` attributes. The helper accepts
        any of these shapes to keep the API forgiving while the rest of the
        codebase is updated.
        """

        if not action:
            return cls()

        if isinstance(action, cls):
            return action

        if hasattr(action, "steer") and hasattr(action, "throttle") and hasattr(action, "brake"):
            return cls(
                steer=float(getattr(action, "steer")),
                throttle=float(getattr(action, "throttle")),
                brake=float(getattr(action, "brake")),
            )

        throttle = action.get("throttle")
        if throttle is None:
            throttle = action.get("accel", 0.0)

        return cls(
            steer=float(action.get("steer", 0.0)),
            throttle=float(throttle),
            brake=float(action.get("brake", 0.0)),
        )


@dataclass
class VehicleState:
    """Instantaneous vehicle state expressed in the vehicle frame."""

    speed: float = 0.0
    v_ego: float = 0.0
    lat_velocity: float = 0.0
    lat_accel: float = 0.0
    long_accel: float = 0.0
    roll: float = 0.0
    roll_lataccel: float = 0.0
    yaw: float = 0.0
    yaw_rate: float = 0.0
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def as_dict(self) -> Dict[str, float]:
        """Return a JSON-friendly dictionary for observations/logging."""

        return {
            "speed": self.speed,
            "v_ego": self.v_ego,
            "lat_velocity": self.lat_velocity,
            "lat_accel": self.lat_accel,
            "long_accel": self.long_accel,
            "roll": self.roll,
            "roll_lataccel": self.roll_lataccel,
            "yaw": self.yaw,
            "yaw_rate": self.yaw_rate,
            "position": self.position,
            "velocity": self.velocity,
        }


@dataclass
class PlannerTarget:
    """Immediate planner target for the controller to track."""

    speed: float = 0.0
    lat_accel: float = 0.0
    long_accel: float = 0.0
    roll_lataccel: float = 0.0
    lateral_error: float = 0.0
    heading_error: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "speed": self.speed,
            "lat_accel": self.lat_accel,
            "long_accel": self.long_accel,
            "roll_lataccel": self.roll_lataccel,
            "lateral_error": self.lateral_error,
            "heading_error": self.heading_error,
        }


@dataclass
class FuturePreview:
    """Discrete preview of future plan samples at a fixed cadence."""

    lat_accel: Tuple[float, ...] = field(default_factory=tuple)
    roll_lataccel: Tuple[float, ...] = field(default_factory=tuple)
    speed: Tuple[float, ...] = field(default_factory=tuple)
    long_accel: Tuple[float, ...] = field(default_factory=tuple)
    dt: float = 0.1

    def as_dict(self) -> Dict[str, Sequence[float]]:
        return {
            "lat_accel": self.lat_accel,
            "roll_lataccel": self.roll_lataccel,
            "speed": self.speed,
            "long_accel": self.long_accel,
            "dt": self.dt,
        }

    @classmethod
    def empty(cls, dt: float) -> "FuturePreview":
        return cls((), (), (), (), dt)


@dataclass
class TelemetrySnapshot:
    """Bundle of the data controllers receive each time step."""

    state: VehicleState = field(default_factory=VehicleState)
    target: PlannerTarget = field(default_factory=PlannerTarget)
    future: FuturePreview = field(default_factory=FuturePreview)

    def as_observation(self) -> Dict[str, Dict[str, Sequence[float]]]:
        """Return a serialisable observation for compatibility with Gym APIs."""

        return {
            "state": self.state.as_dict(),
            "target": self.target.as_dict(),
            "future": self.future.as_dict(),
        }


__all__ = [
    "DriverCommand",
    "VehicleState",
    "PlannerTarget",
    "FuturePreview",
    "TelemetrySnapshot",
]

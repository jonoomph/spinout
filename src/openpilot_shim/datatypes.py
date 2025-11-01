from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SimCarParams:
  """Subset of CarParams values required by the stubs."""

  steerLimitTimer: float = 0.5
  steerControlType: str = "torque"
  stopAccel: float = -2.0
  stoppingDecelRate: float = 0.2
  startAccel: float = 0.5
  vEgoStarting: float = 0.3


@dataclass
class SimCarInterface:
  """Placeholder for brand-specific hooks."""

  def get_pid_accel_limits(self, CP: SimCarParams, v_ego: float, v_cruise: float) -> tuple[float, float]:
    del CP, v_ego, v_cruise
    return (-2.5, 1.8)


@dataclass
class SimCarState:
  vEgo: float
  aEgo: float
  steeringAngleDeg: float
  steeringRateDeg: float
  steeringPressed: bool = False
  brakePressed: bool = False
  standstill: bool = False


@dataclass
class SimLiveParameters:
  steerRatio: float = 14.0
  stiffnessFactor: float = 1.0
  angleOffsetDeg: float = 0.0
  roll: float = 0.0


@dataclass
class SimVehicleModel:
  steer_ratio: float = field(default=14.0, init=False)
  stiffness_factor: float = field(default=1.0, init=False)

  def update_params(self, stiffness_factor: float, steer_ratio: float) -> None:
    self.stiffness_factor = stiffness_factor
    self.steer_ratio = steer_ratio

  def get_steer_from_curvature(self, curvature: float, v_ego: float, roll: float) -> float:
    del v_ego, roll
    return curvature * self.steer_ratio

from __future__ import annotations

from typing import Any, Tuple

from .imports import LatControlBase, LongControlBase, LongCtrlState, make_lat_log


class LatControlStub(LatControlBase):
  """No-op lateral controller used for training scaffolds.

  Example::

      lac = LatControlStub(CP, CI)
      steer, desired_angle, log = lac.update(
          active=True,
          CS=car_state,
          VM=vehicle_model,
          params=live_params,
          steer_limited_by_controls=False,
          desired_curvature=0.01,
          calibrated_pose=None,
          curvature_limited=False,
      )
  """

  def __init__(self, CP: Any, CI: Any) -> None:
    super().__init__(CP, CI)

  def update(
    self,
    active: bool,
    CS: Any,
    VM: Any,
    params: Any,
    steer_limited_by_controls: bool,
    desired_curvature: float,
    calibrated_pose: Any,
    curvature_limited: bool,
  ) -> Tuple[float, float, Any]:
    del VM, params, steer_limited_by_controls, desired_curvature, calibrated_pose, curvature_limited

    log_msg = make_lat_log()
    setattr(log_msg, "active", bool(active))
    setattr(log_msg, "steeringAngleDeg", float(getattr(CS, "steeringAngleDeg", 0.0)))
    setattr(log_msg, "steeringAngleDesiredDeg", 0.0)
    setattr(log_msg, "angleError", 0.0)

    steer_output = 0.0
    desired_angle = 0.0
    return steer_output, desired_angle, log_msg


class LongControlStub(LongControlBase):
  """No-op longitudinal controller used for training scaffolds."""

  def __init__(self, CP: Any) -> None:
    super().__init__(CP)

  def update(
    self,
    active: bool,
    CS: Any,
    a_target: float,
    should_stop: bool,
    accel_limits: Tuple[float, float],
  ) -> float:
    del active, CS, a_target, should_stop, accel_limits
    self.long_control_state = LongCtrlState.off
    return 0.0

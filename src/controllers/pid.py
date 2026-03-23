"""Position-domain PID steering controller with curvature feedforward."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .controller import BaseController
from src.sim.control_api import DriverCommand, TelemetrySnapshot

# Feedforward preview window: 0.5 s at <=20 mph, scaling to 0.9 s at 60 mph.
_PREVIEW_SECS_LO: float = 0.5
_PREVIEW_SECS_HI: float = 0.9
_PREVIEW_V_LO: float = 20.0 * 0.44704
_PREVIEW_V_HI: float = 60.0 * 0.44704
# Heading anticipation horizon used when forming the feedback error.
_HEADING_PREVIEW_SECS = 0.3


@dataclass
class PIDGains:
    """Gains for position-error feedback plus curvature feedforward.

    steer = ff + p + i + d

    All gains are expressed in steer-command units (-1..+1).
      kp: steer per metre of effective lateral error
      ki: steer per metre-second of accumulated error
      kd: steer per metre/second of error rate
      k_ff: steer per m/s^2 of planned lateral acceleration
    """
    kp: float = 0.55
    ki: float = 0.005
    kd: float = 0.35
    k_ff: float = 0.08
    integral_limit: float = 0.25


class PIDSteeringController(BaseController):
    """Position-domain PID controller with curvature feedforward.

    Feedback steers the car back to the path using lateral/heading error.
    Feedforward adds the steering needed for the planned road curvature so the
    controller does not have to wait for tracking error before turning.

    Sign conventions (left-positive):
      lateral_error > 0 → car is left of path → needs right steer (negative)
      positive lat_accel → accelerating left
      positive feedforward → road curves left → needs left steer (positive)
    """

    def __init__(
        self,
        gains: Optional[PIDGains] = None,
        steer_limit: float = 1.0,
        control_rate_hz: float = 10.0,
    ) -> None:
        super().__init__("pid", control_rate_hz=control_rate_hz, preview_rate_hz=control_rate_hz)
        self.gains = gains or PIDGains()
        self.steer_limit = steer_limit

        self._integral: float = 0.0
        self._prev_eff_err: Optional[float] = None

        # Diagnostics kept for logging and tests.
        self._last_target_lat: float = 0.0
        self._last_base_target_lat: float = 0.0
        self._last_lateral_term: float = 0.0
        self._last_pid_error: float = 0.0
        self._last_p_term: float = 0.0
        self._last_i_term: float = 0.0
        self._last_d_term: float = 0.0
        self._last_pid_out: float = 0.0
        self._last_ff: float = 0.0

    def reset(self) -> None:  # type: ignore[override]
        self._integral = 0.0
        self._prev_eff_err = None
        self._last_target_lat = self._last_base_target_lat = 0.0
        self._last_lateral_term = self._last_pid_error = 0.0
        self._last_p_term = self._last_i_term = self._last_d_term = 0.0
        self._last_pid_out = self._last_ff = 0.0

    def on_disable(self) -> None:  # type: ignore[override]
        self.reset()

    def _preview_seconds(self, v_ego: float) -> float:
        speed = min(max(v_ego, _PREVIEW_V_LO), _PREVIEW_V_HI)
        frac = (speed - _PREVIEW_V_LO) / (_PREVIEW_V_HI - _PREVIEW_V_LO)
        return _PREVIEW_SECS_LO + frac * (_PREVIEW_SECS_HI - _PREVIEW_SECS_LO)

    def _feedforward_lat_accel(self, telemetry: TelemetrySnapshot) -> float:
        fut = telemetry.future.lat_accel or ()
        if not fut:
            return telemetry.target.lat_accel

        preview_secs = self._preview_seconds(telemetry.state.v_ego)
        preview_hz = self.preview_rate_hz or 10.0
        n = min(len(fut), max(1, round(preview_secs * preview_hz)))
        return float(np.mean(fut[:n]))

    def _effective_lateral_error(self, telemetry: TelemetrySnapshot) -> float:
        return (
            telemetry.target.lateral_error
            + telemetry.target.heading_error * telemetry.state.v_ego * _HEADING_PREVIEW_SECS
        )

    def step(self, telemetry: TelemetrySnapshot, manual: DriverCommand) -> DriverCommand:
        if not self.enabled:
            return manual

        dt = max(self.dt or 0.1, 1e-6)

        base = self._feedforward_lat_accel(telemetry)
        ff = self.gains.k_ff * base

        e = self._effective_lateral_error(telemetry)

        de_dt = 0.0 if self._prev_eff_err is None else (e - self._prev_eff_err) / dt
        self._prev_eff_err = e

        self._integral = max(
            -self.gains.integral_limit,
            min(self.gains.integral_limit, self._integral + e * dt),
        )

        p_term = -self.gains.kp * e
        i_term = -self.gains.ki * self._integral
        d_term = -self.gains.kd * de_dt
        correction = p_term + i_term + d_term
        steer = max(-self.steer_limit, min(self.steer_limit, ff + correction))

        self._last_base_target_lat = base
        self._last_target_lat = base
        self._last_pid_error = e
        self._last_p_term = p_term
        self._last_i_term = i_term
        self._last_d_term = d_term
        self._last_pid_out = correction
        self._last_ff = ff
        self._last_lateral_term = correction

        return DriverCommand(steer=steer, throttle=manual.throttle, brake=manual.brake)

"""Simple PID-based controller that outputs steering, throttle and brake."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np

from .controller import BaseController
from src.sim.control_api import DriverCommand, TelemetrySnapshot


@dataclass
class PIDGains:
    """Bundle of PID gains with a small helper docstring.

    Examples
    --------
    >>> gains = PIDGains(kp=0.3, ki=0.07, kd=-0.1)
    >>> gains.kp
    0.3
    """

    kp: float = 0.3
    ki: float = 0.07
    kd: float = -0.01  # CC uses d*(e-prev_e) with no /dt; our code divides by dt=0.1, so kd=-0.01 ≡ CC d=-0.1
    integral_limit: float = 0.5


class PIDSteeringController(BaseController):
    """PID controller inspired by the Comma.ai controls challenge.

    Tracks lateral acceleration rather than lateral position directly.
    The planner's road-curvature target is augmented with position and
    velocity correction terms to produce a desired lateral acceleration.
    A PID then closes the loop on (target_lat_accel - current_lat_accel),
    with a large feedforward carrying the majority of the curvature demand.

    Sign conventions (left-positive, matching the road planner):
      - positive steer      → left turn
      - positive lat_accel  → accelerating left
      - positive lat_velocity → moving left
      - lateral_error > 0   → car is to the LEFT of the path
      - heading_error > 0   → car is heading LEFT relative to path tangent
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

        # Feedforward scaling (sigmoid linearisation around small steer angles)
        self.steer_factor = 9.0
        self.steer_sat_v = 20.0
        self.steer_command_sat = 2.0

        # Cross-track and lateral-velocity correction gains (simple linear, no tanh)
        self.kp_lat = 1.0   # m/s² per m of lateral error
        self.kd_lat = 1.5   # m/s² per m/s of lateral closing rate
        # Short heading anticipation (pure-pursuit style, 0.2s lookahead).
        # This prevents heading-induced overshoot on straight roads without causing
        # the premature reversal at S-curve peaks that the original 1.0s preview had.
        self.preview_secs = 0.3

        # Feedforward weight (fraction of full sigmoid steer command)
        self.K_ff = 0.7

        self.max_target_lat = 4.0

        self._integral = 0.0
        self._prev_error: Optional[float] = None
        self._prev_lateral_error: Optional[float] = None
        self._step_counter = 0
        self._last_target_lat: float = 0.0

        # Diagnostic terms written each step (readable externally for logging/plotting).
        self._last_p_term: float = 0.0
        self._last_i_term: float = 0.0
        self._last_d_term: float = 0.0
        self._last_pid_out: float = 0.0
        self._last_ff: float = 0.0
        self._last_pid_error: float = 0.0
        self._last_base_target_lat: float = 0.0
        self._last_lateral_term: float = 0.0

    def reset(self) -> None:  # type: ignore[override]
        self._integral = 0.0
        self._prev_error = None
        self._prev_lateral_error = None
        self._step_counter = 0
        self._last_target_lat = 0.0
        self._last_p_term = 0.0
        self._last_i_term = 0.0
        self._last_d_term = 0.0
        self._last_pid_out = 0.0
        self._last_ff = 0.0
        self._last_pid_error = 0.0
        self._last_base_target_lat = 0.0
        self._last_lateral_term = 0.0

    def on_disable(self) -> None:  # type: ignore[override]
        self.reset()

    # ------------------------------------------------------------------

    def _clip(self, value: float, limit: float) -> float:
        return max(-limit, min(limit, value))

    def _compute_pid(self, error: float, dt: float) -> float:
        derivative = 0.0 if self._prev_error is None else (error - self._prev_error) / dt
        self._integral += error * dt
        self._integral = self._clip(self._integral, self.gains.integral_limit)
        output = (
            self.gains.kp * error
            + self.gains.ki * self._integral
            + self.gains.kd * derivative
        )
        self._prev_error = error
        return output

    def step(  # type: ignore[override]
        self, telemetry: TelemetrySnapshot, manual: DriverCommand
    ) -> DriverCommand:
        if not self.enabled:
            return manual

        dt = self.dt or 0.1
        self._step_counter += 1
        if self._step_counter >= int(8.1 / max(dt, 1e-6)):
            self._integral = 0.0
            self._prev_error = None
            self._step_counter = 0

        state = telemetry.state

        # --- 1. Base target: weighted average of near-future curvature ---
        # Weights favour points closer to the car to avoid over-anticipating
        # the far end of a curve before the car arrives.
        if telemetry.future.lat_accel and len(telemetry.future.lat_accel) >= 1:
            n_use = min(4, len(telemetry.future.lat_accel))
            weights = [5, 6, 7, 8][:n_use]
            base_target_lat = float(np.average(
                telemetry.future.lat_accel[:n_use], weights=weights
            ))
        else:
            base_target_lat = telemetry.target.lat_accel
        self._last_base_target_lat = base_target_lat

        dt_safe = max(dt, 1e-6)

        # --- 2. Simple cross-track correction (no tanh, short heading anticipation) ---
        # Project lateral error forward by preview_secs to anticipate heading-induced
        # drift.  Kept short (0.2s vs original 1.0s) to avoid premature reversal at
        # S-curve inflection points while still preventing overshoot on straights.
        heading_error = telemetry.target.heading_error
        lateral_error = telemetry.target.lateral_error
        lateral_error_preview = (
            lateral_error + heading_error * state.v_ego * self.preview_secs
        )
        lateral_term = -self.kp_lat * lateral_error_preview

        # World-frame lateral closing rate: d(lateral_error)/dt.
        # More accurate than car-frame lat_velocity when heading error is large,
        # because the car-frame velocity underestimates the world-frame closing rate
        # when the car is angled across the path.
        if self._prev_lateral_error is not None:
            lateral_error_dot = (lateral_error - self._prev_lateral_error) / dt_safe
        else:
            lateral_error_dot = 0.0
        self._prev_lateral_error = lateral_error
        lat_velocity_term = -self.kd_lat * lateral_error_dot
        self._last_lateral_term = lateral_term

        target_lat = base_target_lat + lateral_term + lat_velocity_term
        target_lat = self._clip(target_lat, self.max_target_lat)
        self._last_target_lat = target_lat

        # --- 3. PID on (target - actual) lateral acceleration ---
        error = target_lat - state.lat_accel

        # Pre-compute derivative before _compute_pid updates _prev_error.
        pre_derivative = (
            0.0 if self._prev_error is None
            else (error - self._prev_error) / dt_safe
        )

        pid_out = self._compute_pid(error, dt_safe)

        self._last_pid_error = error
        self._last_p_term = self.gains.kp * error
        self._last_i_term = self.gains.ki * self._integral
        self._last_d_term = self.gains.kd * pre_derivative
        self._last_pid_out = pid_out

        # --- 4. Feedforward based on road curvature ONLY (not lateral correction) ---
        # Using base_target_lat (road curvature) keeps position-correction terms out of
        # the ff path.  If target_lat were used, a 3 m lateral offset would be amplified
        # ~3× by ff, causing severe overshoot on straight-road centering.
        speed = max(state.v_ego, 1.0)
        steer_cmd = base_target_lat * self.steer_factor / max(self.steer_sat_v, speed)
        steer_cmd = 2 * self.steer_command_sat / (1 + math.exp(-steer_cmd)) - self.steer_command_sat

        ff = self.K_ff * steer_cmd
        self._last_ff = ff

        # --- 5. Output ---
        steer = self._clip(pid_out + ff, self.steer_command_sat)
        steer = self._clip(steer / self.steer_command_sat, self.steer_limit)

        return DriverCommand(
            steer=steer,
            throttle=manual.throttle,
            brake=manual.brake,
        )

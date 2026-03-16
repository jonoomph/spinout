"""Simple PID-based controller that outputs steering, throttle and brake."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

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
    kd: float = -0.1
    integral_limit: float = 1.5


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
        self.steer_factor = 9.0
        self.steer_sat_v = 20.0
        self.steer_command_sat = 2.0

        # Position correction weight (m/s² per metre of lateral offset).
        # lateral_error < 0 (car to RIGHT) → positive lateral_term (turn left).
        # k1 = 1.0 gives ω_n = 1 rad/s; time-constant ≈ 1 s → settles ±3 m in ~6 s.
        self.lateral_error_gain = 1.0

        # Velocity damping using *effective* world-frame lateral velocity:
        #   effective_v_lat = lat_velocity + v_capped * heading_error
        # where v_capped = min(v_ego, v_lat_ref) caps the speed contribution to
        # prevent gain explosion at high speed that would otherwise cause heading
        # oscillations to grow as the car accelerates.
        # heading_error > 0 means car is aimed left → v_ego * heading_error > 0
        # represents world-frame leftward drift.
        # Target ζ ≈ 0.5: k2 = 2*ζ*sqrt(k1) = 2*0.5*1.0 = 1.0 → ~16 % overshoot.
        self.lat_velocity_gain = 1.0
        self.v_lat_ref = 8.0  # m/s; cap for heading_error speed contribution

        # roll_comp_gain: removes road-camber bias from the feedforward.
        self.roll_comp_gain = 0.1

        self.max_target_lat = 4.0
        self.target_slew_rate = 15.0

        self._integral = 0.0
        self._prev_error: Optional[float] = None
        self._step_counter = 0
        self._last_target_lat: Optional[float] = None

    def reset(self) -> None:  # type: ignore[override]
        self._integral = 0.0
        self._prev_error = None
        self._step_counter = 0
        self._last_target_lat = None

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

        # --- Build combined target lateral acceleration ---
        # Start from road curvature (look-ahead preview when available).
        base_target_lat = (
            telemetry.future.lat_accel[0]
            if telemetry.future.lat_accel
            else telemetry.target.lat_accel
        )

        # Position correction: car to RIGHT (lateral_error < 0) → add left accel.
        lateral_term = -self.lateral_error_gain * telemetry.target.lateral_error

        # Velocity damping: use effective world-frame lateral velocity but cap
        # the speed contribution to prevent gain explosion at high speed.
        # Without the cap, v_ego * heading_error grows with vehicle speed, making
        # the damping coefficient speed-dependent and potentially destabilising.
        heading_error = telemetry.target.heading_error
        v_capped = min(state.v_ego, self.v_lat_ref)
        effective_v_lat = state.lat_velocity + v_capped * heading_error
        lat_velocity_term = -self.lat_velocity_gain * effective_v_lat

        # Road-camber compensation.
        roll_term = -self.roll_comp_gain * state.roll_lataccel

        target_lat = (
            base_target_lat
            + lateral_term
            + lat_velocity_term
            + roll_term
        )
        target_lat = self._clip(target_lat, self.max_target_lat)

        # Slew-rate limit to keep the target smooth.
        if self._last_target_lat is None:
            filtered_target = target_lat
        else:
            max_delta = self.target_slew_rate * dt
            delta = target_lat - self._last_target_lat
            if delta > max_delta:
                filtered_target = self._last_target_lat + max_delta
            elif delta < -max_delta:
                filtered_target = self._last_target_lat - max_delta
            else:
                filtered_target = target_lat
        self._last_target_lat = filtered_target
        target_lat = filtered_target

        # --- PID on lateral-acceleration error (not position error) ---
        # kd = -0.1 makes sense here because error *decreases* as the car builds up
        # the desired lateral acceleration, so the negative derivative term acts to
        # maintain the corrective drive.
        current_lat = state.lat_accel
        error = target_lat - current_lat

        pid_factor = max(0.5, 1.0 - 0.23 * abs(target_lat))
        p_dynamic = max(0.1, self.gains.kp - 0.1 * abs(state.long_accel))
        base_gain = max(self.gains.kp, 1e-6)
        pid_out = (
            self._compute_pid(error, max(dt, 1e-6))
            * pid_factor
            * (p_dynamic / base_gain)
        )

        # --- Large feedforward (80%) carries the curvature demand ---
        steer_accel_target = target_lat - state.roll_lataccel
        speed = max(state.v_ego, 1.0)
        steer_cmd = steer_accel_target * self.steer_factor / max(self.steer_sat_v, speed)
        steer_cmd = 2 * self.steer_command_sat / (1 + math.exp(-steer_cmd)) - self.steer_command_sat
        ff = 0.8 * steer_cmd

        steer = self._clip(pid_out + ff, self.steer_command_sat)
        steer = self._clip(steer / self.steer_command_sat, self.steer_limit)

        return DriverCommand(
            steer=steer,
            throttle=manual.throttle,
            brake=manual.brake,
        )

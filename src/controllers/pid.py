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
    integral_limit: float = 1.0


class PIDSteeringController(BaseController):
    """PID controller loosely inspired by the Comma.ai controls challenge."""

    def __init__(
        self,
        gains: Optional[PIDGains] = None,
        steer_limit: float = 1.0,
        control_rate_hz: float = 10.0,
    ) -> None:
        super().__init__("pid", control_rate_hz=control_rate_hz, preview_rate_hz=control_rate_hz)
        self.gains = gains or PIDGains()
        self.steer_limit = steer_limit
        self.steer_factor = 13.0
        self.steer_sat_v = 20.0
        self.steer_command_sat = 2.0
        self.lateral_error_gain = 0.6
        self.heading_error_gain = 1.0
        self.lateral_velocity_gain = 0.3
        self.roll_comp_gain = 0.1
        self.max_target_lat = 4.0
        self.max_error = 4.0
        self.target_slew_rate = 6.0
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
        base_target_lat = (
            telemetry.future.lat_accel[0]
            if telemetry.future.lat_accel
            else telemetry.target.lat_accel
        )
        current_lat = state.lat_accel
        lateral_term = -self.lateral_error_gain * telemetry.target.lateral_error
        heading_term = self.heading_error_gain * telemetry.target.heading_error
        lat_velocity_term = -self.lateral_velocity_gain * state.lat_velocity
        roll_term = -self.roll_comp_gain * state.roll_lataccel

        target_lat = (
            base_target_lat
            + lateral_term
            + heading_term
            + lat_velocity_term
            + roll_term
        )
        target_lat = self._clip(target_lat, self.max_target_lat)

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

        error = target_lat - current_lat
        error = self._clip(error, self.max_error)

        pid_factor = max(0.5, 1 - 0.23 * abs(target_lat))
        p_dynamic = max(0.1, self.gains.kp - 0.1 * abs(state.long_accel))
        base_gain = max(self.gains.kp, 1e-6)
        pid_out = (
            self._compute_pid(error, max(dt, 1e-6))
            * pid_factor
            * (p_dynamic / base_gain)
        )

        roll_lat = state.roll_lataccel
        steer_accel_target = target_lat - roll_lat
        speed = max(state.v_ego, 1.0)
        steer_cmd = steer_accel_target * self.steer_factor / max(self.steer_sat_v, speed)
        steer_cmd = 2 * self.steer_command_sat / (1 + math.exp(-steer_cmd)) - self.steer_command_sat
        ff = 0.8 * steer_cmd

        steer = self._clip(pid_out + ff, self.steer_command_sat)
        steer = self._clip(steer / self.steer_command_sat, self.steer_limit)

        diff_lat_accel = target_lat - current_lat

        # DEBUG steer test
        if diff_lat_accel > 0.0:
            steer = 0.25
        else:
            steer = -0.25

        print(
            f"PID: {steer}, current_lat: {current_lat}, base_target_lat: {base_target_lat}, "
            f"roll_lat: {roll_lat}, diff_lat_accel: {diff_lat_accel}, "
            f"lateral_term: {lateral_term}, heading_term: {heading_term}, "
            f"lat_vel_term: {lat_velocity_term}, roll_term: {roll_term}, target_lat: {target_lat}"
        )


        return DriverCommand(
            steer=steer,
            throttle=manual.throttle,
            brake=manual.brake,
        )

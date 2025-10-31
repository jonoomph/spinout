import numpy as np
import pytest

from src.controllers.controller import BaseController
from src.controllers.pid import PIDSteeringController
from src.sim.control_api import (
    DriverCommand,
    FuturePreview,
    PlannerTarget,
    TelemetrySnapshot,
    VehicleState,
)
from src.sim.environment import Environment


class CountingController(BaseController):
    def __init__(self, rate_hz: float = 10.0):
        super().__init__("counting", control_rate_hz=rate_hz)
        self.calls = 0

    def step(self, telemetry, manual):  # type: ignore[override]
        self.calls += 1
        return DriverCommand(steer=0.1, throttle=manual.throttle, brake=manual.brake)


def test_controller_runs_at_declared_frequency():
    env = Environment({"flat": True, "dt": 1.0 / 60.0, "substeps": 1}, mode="train")
    env.reset()

    controller = CountingController(rate_hz=10.0)
    controller.attach(env)
    env.attach_controller(controller)
    controller.enable()

    applied = []
    for _ in range(30):
        _obs, _reward, terminated, truncated, info = env.step(DriverCommand())
        assert not terminated
        assert not truncated
        applied.append(info["driver_command"]["steer"])

    # 30 steps at 60 Hz cover 0.5s, so a 10 Hz controller should update 5 times.
    assert controller.calls == 5
    assert all(abs(val - 0.1) < 1e-6 for val in applied)


def test_pid_reacts_to_lateral_error():
    snapshot = TelemetrySnapshot(
        state=VehicleState(v_ego=15.0, lat_velocity=0.0, lat_accel=0.0),
        target=PlannerTarget(lat_accel=0.0, lateral_error=1.0, heading_error=0.0),
        future=FuturePreview.empty(0.1),
    )
    controller = PIDSteeringController()
    controller.enable()
    steer_left = controller.step(snapshot, DriverCommand()).steer
    assert steer_left < 0.0

    snapshot.target = PlannerTarget(lat_accel=0.0, lateral_error=-1.0, heading_error=0.0)
    controller = PIDSteeringController()
    controller.enable()
    steer_right = controller.step(snapshot, DriverCommand()).steer
    assert steer_right > 0.0


def test_pid_tracks_future_lat_accel_heading():
    snapshot = TelemetrySnapshot(
        state=VehicleState(v_ego=12.0, lat_velocity=0.0, lat_accel=0.0),
        target=PlannerTarget(lat_accel=3.0, lateral_error=0.0, heading_error=0.1),
        future=FuturePreview(lat_accel=(3.0,), roll_lataccel=(0.0,), speed=(12.0,), long_accel=(0.0,), dt=0.1),
    )
    controller = PIDSteeringController()
    controller.enable()
    command = controller.step(snapshot, DriverCommand())
    assert command.steer > 0.0


def test_environment_reports_centripetal_lat_accel():
    env = Environment({"flat": True, "dt": 0.1, "substeps": 1}, mode="train")
    env.reset()

    assert env.car is not None
    env.car.body.rot.arr[:] = (1.0, 0.0, 0.0, 0.0)
    env.car.body.vel = np.array([0.1, 0.0, 10.0], dtype=float)
    env.car.body.angvel = np.array([0.0, 0.1, 0.0], dtype=float)

    prev_velocity = np.array([0.0, 0.0, 10.0], dtype=float)
    state = env._compute_state(prev_velocity)  # type: ignore[attr-defined]

    expected = state.v_ego * 0.1
    assert state.lat_accel == pytest.approx(expected, rel=1e-3)

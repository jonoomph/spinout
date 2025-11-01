import math

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
from src.sim.planner import PlannerPreviewer


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


def test_environment_provides_recent_lat_accel_to_controller():
    env = Environment({"flat": True, "dt": 0.1, "substeps": 1}, mode="train")
    env.reset()

    assert env.car is not None
    env.car.body.rot.arr[:] = (1.0, 0.0, 0.0, 0.0)
    env.car.body.vel = np.array([0.1, 0.0, 10.0], dtype=float)
    env.car.body.angvel = np.array([0.0, 0.1, 0.0], dtype=float)

    velocity_before = np.array([0.0, 0.0, 10.0], dtype=float)
    yaw_rate = 0.1
    expected = velocity_before[2] * yaw_rate

    env._prev_velocity = velocity_before  # type: ignore[attr-defined]
    telemetry = env._build_snapshot(env._prev_velocity)  # type: ignore[attr-defined]

    assert telemetry.state.lat_accel == pytest.approx(expected, rel=1e-3)


def test_environment_lateral_sign_matches_planner_convention():
    env = Environment({"flat": True, "dt": 0.1, "substeps": 1}, mode="train")
    env.reset()

    assert env.car is not None
    env.car.body.rot.arr[:] = (1.0, 0.0, 0.0, 0.0)
    env.car.body.vel = np.array([-2.0, 0.0, 10.0], dtype=float)
    env.car.body.angvel = np.array([0.0, 0.1, 0.0], dtype=float)

    prev_velocity = env.car.body.vel.copy()
    state = env._compute_state(prev_velocity)  # type: ignore[attr-defined]

    assert state.lat_velocity > 0.0
    assert state.lat_accel == pytest.approx(state.v_ego * 0.1, rel=1e-3)


def test_planner_preview_remains_stable_across_state_changes():
    previewer = PlannerPreviewer()

    angles = np.linspace(0.0, 2.0 * math.pi, 400)
    radius = 30.0
    drive_line = [
        (radius * math.sin(a), radius * math.sin(a) * math.cos(a)) for a in angles
    ]
    previewer.set_plan(
        drive_line,
        (
            {
                "start_s": 0.0,
                "end_s": float("inf"),
                "speed_mph": 35.0,
            },
        ),
    )

    idx = 90
    base_point = np.array(drive_line[idx])
    tangent = np.array(drive_line[idx + 1]) - np.array(drive_line[idx])
    tangent /= np.linalg.norm(tangent)
    normal = np.array([-tangent[1], tangent[0]])
    heading = math.atan2(tangent[1], tangent[0])

    speed = 12.0
    offsets = [0.0, 2.5, -2.5]
    baseline_lat = None
    baseline_future: np.ndarray | None = None

    for offset in offsets:
        pos = (
            base_point[0] + normal[0] * offset,
            0.0,
            base_point[1] + normal[1] * offset,
        )
        preview = previewer.preview(pos, speed, preview_hz=10.0)
        target = previewer.immediate_target(pos, speed, heading, preview)
        current_future = np.array(preview.lat_accel[:5], dtype=float)

        if baseline_lat is None:
            baseline_lat = target.lat_accel
            baseline_future = current_future
            continue

        assert target.lat_accel == pytest.approx(baseline_lat, rel=1e-6, abs=1e-6)
        assert current_future == pytest.approx(baseline_future, rel=1e-6, abs=1e-6)

    slower_speed = speed * 0.5
    preview = previewer.preview(
        (base_point[0], 0.0, base_point[1]), slower_speed, preview_hz=10.0
    )
    target = previewer.immediate_target(
        (base_point[0], 0.0, base_point[1]), slower_speed, heading, preview
    )
    assert target.lat_accel == pytest.approx(baseline_lat, rel=1e-6, abs=1e-6)
    assert np.array(preview.lat_accel[:5], dtype=float) == pytest.approx(
        baseline_future, rel=1e-6, abs=1e-6
    )

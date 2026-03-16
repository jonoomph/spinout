import math
import os

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
from tests.helpers import configure_flat_drive_line, ensure_drive_line_layer

VISUALIZE = bool(int(os.environ.get("VISUALIZE", "1")))
MPS_TO_MPH = 2.2369362920544


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

    expected = -state.v_ego * 0.1
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
    expected = -velocity_before[2] * yaw_rate

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
    assert state.lat_accel == pytest.approx(-state.v_ego * 0.1, rel=1e-3)


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


def _place_car_with_lateral_offset(env: Environment, center_x: float, offset_m: float) -> None:
    """Drop the car at the requested lateral offset relative to the driveline."""

    assert env.car is not None and env.terrain is not None
    z = env.terrain.height * 0.25
    x = center_x + offset_m
    y = env.terrain.get_height(x, z) + env.car.cg_height_m
    env.car.body.pos = np.array([x, y, z], dtype=float)
    env.car.body.vel[:] = 0.0
    env.car.body.angvel[:] = 0.0
    env.car.body.rot.arr[:] = (1.0, 0.0, 0.0, 0.0)
    env._prev_velocity = env.car.body.vel.copy()  # type: ignore[attr-defined]
    env._refresh_initial_telemetry()  # type: ignore[attr-defined]


def _run_pid_centering_trial(offset_m: float, target_speed_mph: float = 25.0) -> tuple[bool, str]:
    """Simulate a short run and report whether the PID controller recentres the car."""

    print(
        f"\n\n=== PID centering trial: offset={offset_m:+.2f} m,"
        f" target_speed={target_speed_mph:.1f} mph ==="
    )

    mode = "eval" if VISUALIZE else "train"
    env = Environment({"flat": True, "dt": 0.02, "substeps": 1}, mode=mode)
    env.reset()
    center_x = configure_flat_drive_line(env)
    if VISUALIZE:
        env.init_renderer()
        ensure_drive_line_layer(env)
    _place_car_with_lateral_offset(env, center_x, offset_m)

    controller = PIDSteeringController()
    env.attach_controller(controller)
    controller.enable()

    stable_time = 0.0
    stable_required = 3.0
    tolerance_m = 0.3  # ~1 ft band around the driving line
    max_time = 13.0
    steps = int(math.ceil(max_time / env.dt))
    crossed_line = False
    last_error = abs(offset_m)

    max_speed_mph = 30.0
    target_speed_mph = min(target_speed_mph, max_speed_mph)
    throttle_cmd = 0.0
    throttle_alpha = 0.25  # keep throttle adjustments smooth

    for _ in range(steps):
        assert env.car is not None
        speed_mph = np.linalg.norm(env.car.body.vel) * MPS_TO_MPH

        speed_error = target_speed_mph - speed_mph
        if speed_mph > max_speed_mph + 2.0:
            desired_throttle = 0.0  # let the car coast back down
        elif speed_mph > max_speed_mph + 0.5:
            desired_throttle = 0.15
        elif speed_error > 5.0:
            desired_throttle = 0.8
        elif speed_error > 2.0:
            desired_throttle = 0.6
        elif speed_error > 0.5:
            desired_throttle = 0.45
        elif speed_error > -0.5:
            desired_throttle = 0.3
        else:
            desired_throttle = 0.2

        throttle_cmd += throttle_alpha * (desired_throttle - throttle_cmd)
        throttle = throttle_cmd
        _, _, terminated, truncated, _ = env.step(DriverCommand(throttle=throttle))
        assert not terminated, "car left the test terrain unexpectedly"
        assert not truncated, "simulation truncated before PID could stabilise"

        lateral_error = env.car.body.pos[0] - center_x
        abs_error = abs(lateral_error)
        if controller.enabled and not crossed_line and offset_m != 0.0:
            crossed_line = (lateral_error > 0) != (offset_m > 0)
        if crossed_line and abs_error > abs(offset_m) + 0.5:
            env.attach_controller(None)
            return False, (
                f"offset={offset_m:.2f} m: crossed driveline but diverged "
                f"(error {abs_error:.2f} m)"
            )

        if abs_error <= tolerance_m:
            stable_time += env.dt
            if stable_time >= stable_required:
                env.attach_controller(None)
                return True, ""
        else:
            stable_time = 0.0
        last_error = abs_error

    env.attach_controller(None)
    return False, (
        f"offset={offset_m:.2f} m: failed to centre within {max_time}s "
        f"(last error {last_error:.2f} m)"
    )


def test_pid_recenters_from_lateral_offsets():
    """Ensure the PID controller can centre the car from both small and large offsets."""

    offsets_m = (1.0, -1.0, 3.0, -3.0)  # ~3 ft and ~10 ft on both sides
    failures = []
    for offset in offsets_m:
        ok, message = _run_pid_centering_trial(offset)
        if not ok:
            failures.append(message)
    if failures:
        pytest.fail("\n".join(failures))

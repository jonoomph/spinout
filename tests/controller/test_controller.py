import math
import os
import csv
from pathlib import Path

import numpy as np
import pytest

from src.controllers.controller import BaseController
from src.controllers.pid import PIDGains, PIDSteeringController
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

VISUALIZE = bool(int(os.environ.get("VISUALIZE", "0")))
PID_DEBUG_PLOTS = bool(int(os.environ.get("PID_DEBUG_PLOTS", "0")))
PID_DEBUG_PLOT_LABEL = os.environ.get("PID_DEBUG_PLOT_LABEL", "").strip()
PID_DEBUG_PLOT_DIR = os.environ.get("PID_DEBUG_PLOT_DIR", "").strip()
S_CURVE_CONTROL_HZ = float(os.environ.get("S_CURVE_CONTROL_HZ", "100.0"))
S_CURVE_PHYSICS_HZ = float(os.environ.get("S_CURVE_PHYSICS_HZ", "300.0"))
S_CURVE_POINT_SCALE = int(os.environ.get("S_CURVE_POINT_SCALE", "1"))
MPS_TO_MPH = 2.2369362920544


class CountingController(BaseController):
    def __init__(self, rate_hz: float = 10.0):
        super().__init__("counting", control_rate_hz=rate_hz)
        self.calls = 0

    def step(self, telemetry, manual):  # type: ignore[override]
        self.calls += 1
        return DriverCommand(steer=0.1, throttle=manual.throttle, brake=manual.brake)


def _snapshot(
    *,
    v_ego: float = 15.0,
    lateral_error: float = 0.0,
    heading_error: float = 0.0,
    target_lat_accel: float = 0.0,
    future_lat_accel: tuple[float, ...] = (),
) -> TelemetrySnapshot:
    dt = 0.1
    return TelemetrySnapshot(
        state=VehicleState(v_ego=v_ego, lat_velocity=0.0, lat_accel=0.0),
        target=PlannerTarget(
            lat_accel=target_lat_accel,
            lateral_error=lateral_error,
            heading_error=heading_error,
        ),
        future=FuturePreview(
            lat_accel=future_lat_accel,
            roll_lataccel=tuple(0.0 for _ in future_lat_accel),
            speed=tuple(v_ego for _ in future_lat_accel),
            long_accel=tuple(0.0 for _ in future_lat_accel),
            dt=dt,
        ) if future_lat_accel else FuturePreview.empty(dt),
    )


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


def test_eval_defaults_can_derive_dt_and_substeps_from_rates():
    env = Environment({"render_fps": 60.0, "physics_hz": 300.0}, mode="eval")
    assert env.dt == pytest.approx(1.0 / 60.0)
    assert env.substeps == 5


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


def test_pid_p_term_isolated():
    controller = PIDSteeringController(
        gains=PIDGains(kp=0.2, ki=0.0, kd=0.0, k_ff=0.0),
    )
    controller.enable()

    command = controller.step(_snapshot(lateral_error=1.5), DriverCommand())

    assert command.steer == pytest.approx(-0.3)
    assert controller._last_p_term == pytest.approx(-0.3)  # type: ignore[attr-defined]
    assert controller._last_i_term == pytest.approx(0.0)   # type: ignore[attr-defined]
    assert controller._last_d_term == pytest.approx(0.0)   # type: ignore[attr-defined]
    assert controller._last_ff == pytest.approx(0.0)       # type: ignore[attr-defined]


def test_pid_i_term_accumulates_isolated():
    controller = PIDSteeringController(
        gains=PIDGains(kp=0.0, ki=1.0, kd=0.0, k_ff=0.0, integral_limit=10.0),
    )
    controller.enable()

    snapshot = _snapshot(lateral_error=0.5)
    first = controller.step(snapshot, DriverCommand())
    second = controller.step(snapshot, DriverCommand())

    assert first.steer == pytest.approx(-0.05)
    assert second.steer == pytest.approx(-0.10)
    assert controller._last_p_term == pytest.approx(0.0)    # type: ignore[attr-defined]
    assert controller._last_i_term == pytest.approx(-0.10)  # type: ignore[attr-defined]
    assert controller._last_d_term == pytest.approx(0.0)    # type: ignore[attr-defined]
    assert controller._last_ff == pytest.approx(0.0)        # type: ignore[attr-defined]


def test_pid_d_term_responds_to_error_rate_isolated():
    controller = PIDSteeringController(
        gains=PIDGains(kp=0.0, ki=0.0, kd=0.05, k_ff=0.0),
    )
    controller.enable()

    controller.step(_snapshot(lateral_error=0.0), DriverCommand())
    command = controller.step(_snapshot(lateral_error=1.0), DriverCommand())

    assert command.steer == pytest.approx(-0.5)
    assert controller._last_p_term == pytest.approx(0.0)    # type: ignore[attr-defined]
    assert controller._last_i_term == pytest.approx(0.0)    # type: ignore[attr-defined]
    assert controller._last_d_term == pytest.approx(-0.5)   # type: ignore[attr-defined]
    assert controller._last_ff == pytest.approx(0.0)        # type: ignore[attr-defined]


def test_pid_feedforward_uses_future_curvature_isolated():
    controller = PIDSteeringController(
        gains=PIDGains(kp=0.0, ki=0.0, kd=0.0, k_ff=0.2),
    )
    controller.enable()

    command = controller.step(
        _snapshot(v_ego=20.0, future_lat_accel=(2.0, 2.0, 2.0, 2.0)),
        DriverCommand(),
    )

    assert command.steer == pytest.approx(0.4)
    assert controller._last_p_term == pytest.approx(0.0)    # type: ignore[attr-defined]
    assert controller._last_i_term == pytest.approx(0.0)    # type: ignore[attr-defined]
    assert controller._last_d_term == pytest.approx(0.0)    # type: ignore[attr-defined]
    assert controller._last_ff == pytest.approx(0.4)        # type: ignore[attr-defined]


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
    seed_point = np.array(drive_line[idx], dtype=float)
    s_ref, _, _ = previewer._project((seed_point[0], 0.0, seed_point[1]))  # type: ignore[attr-defined]
    base_point, tangent, _ = previewer._sample_path(s_ref)  # type: ignore[attr-defined]
    normal = np.array([-tangent[1], tangent[0]])
    heading = math.atan2(tangent[0], tangent[1])

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
    env = Environment({"flat": True, "dt": 0.02, "substeps": 1, "sun_time_hours": 12.0}, mode=mode)
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
    stable_required = 10.0
    tolerance_m = 0.3  # ~1 ft band around the driving line
    max_time = 60.0
    steps = int(math.ceil(max_time / env.dt))
    crossed_line = False
    last_error = abs(offset_m)

    max_speed_mph = 30.0
    target_speed_mph = min(target_speed_mph, max_speed_mph)
    throttle_cmd = 0.0
    throttle_alpha = 0.25  # keep throttle adjustments smooth

    for step_idx in range(steps):
        assert env.car is not None
        speed_mph = np.linalg.norm(env.car.body.vel) * MPS_TO_MPH

        speed_error = target_speed_mph - speed_mph
        brake_cmd = 0.0
        if speed_mph > max_speed_mph + 2.0:
            desired_throttle = 0.0
            brake_cmd = 0.25
        elif speed_mph > max_speed_mph + 0.5:
            desired_throttle = 0.0
            brake_cmd = 0.10
        elif speed_mph > target_speed_mph + 1.0:
            desired_throttle = 0.0  # coast back to target
        elif speed_error > 5.0:
            desired_throttle = 0.8
        elif speed_error > 2.0:
            desired_throttle = 0.6
        elif speed_error > 0.5:
            desired_throttle = 0.45
        elif speed_error > -0.5:
            desired_throttle = 0.10  # light maintenance throttle at target speed
        else:
            desired_throttle = 0.0   # coast when slightly over target

        throttle_cmd += throttle_alpha * (desired_throttle - throttle_cmd)
        _, _, terminated, truncated, info = env.step(DriverCommand(throttle=throttle_cmd, brake=brake_cmd))
        assert not terminated, "car left the test terrain unexpectedly"
        assert not truncated, "simulation truncated before PID could stabilise"

        lateral_error = env.car.body.pos[0] - center_x
        abs_error = abs(lateral_error)

        tele = env._build_snapshot(env._prev_velocity)  # type: ignore[attr-defined]
        if controller.enabled and not crossed_line and offset_m != 0.0:
            crossed_line = (lateral_error > 0) != (offset_m > 0)
        if crossed_line and abs_error > abs(offset_m) + 0.5:
            t_now = round(step_idx * env.dt, 2)
            tele = env._build_snapshot(env._prev_velocity)  # type: ignore[attr-defined]
            print(
                f"  DIVERGE at t={t_now}s: lat_err={lateral_error:.2f}m  "
                f"steer={info['driver_command']['steer']:.3f}  "
                f"lat_accel={tele.state.lat_accel:.2f}  "
                f"lat_vel={tele.state.lat_velocity:.2f}  "
                f"heading_err={tele.target.heading_error:.3f}"
            )
            env.attach_controller(None)
            return False, (
                f"offset={offset_m:.2f} m: crossed driveline but diverged "
                f"(error {abs_error:.2f} m at t={t_now}s)"
            )

        if abs_error <= tolerance_m:
            stable_time += env.dt
            if stable_time >= stable_required:
                env.attach_controller(None)
                return True, ""
        else:
            stable_time = 0.0
        last_error = abs_error

    t_total = steps * env.dt
    env.attach_controller(None)
    return False, (
        f"offset={offset_m:.2f} m: failed to centre within {t_total:.0f}s "
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


# ── S-curve trial helpers ────────────────────────────────────────────────────


def _smoothstep(s: float) -> float:
    """Hermite smooth step — zero first-derivative at s=0 and s=1."""
    s = max(0.0, min(1.0, s))
    return s * s * (3.0 - 2.0 * s)


def _build_s_curve_drive_line(
    terrain_width: float,
    terrain_height: float,
    *,
    amplitude_m: float = 5.0,
    s_length_m: float = 200.0,
    direction: float = 1.0,
) -> tuple[list[tuple[float, float]], float, float, float]:
    """Build a straight → full-period S-curve → straight drive line.

    The S-curve uses a four-phase smoothstep envelope so the path slope is
    zero at the entry, at both peaks, and at the exit — guaranteeing smooth
    transitions to/from the straight sections with no heading kink.

    ``direction=+1``  → curves left first (positive-x bump first).
    ``direction=-1``  → curves right first.

    Returns ``(drive_line, center_x, z_s_start, z_s_end)``.
    """
    center_x = terrain_width * 0.5
    car_start_z = terrain_height * 0.25       # 300 m for a 1200 m terrain
    z_s_start = car_start_z + 80.0            # 380 m  (80 m entry straight)
    z_s_end = z_s_start + s_length_m          # 580 m
    z_exit = terrain_height - 50.0            # 1150 m

    pts: list[tuple[float, float]] = []

    # --- entry straight ---
    n_entry = max(100, 100 * S_CURVE_POINT_SCALE)
    for i in range(n_entry):
        z = z_s_start * i / max(n_entry - 1, 1)
        pts.append((center_x, z))

    # --- S-curve (four smoothstep phases, zero slope at every key point) ---
    n_s = 200 * S_CURVE_POINT_SCALE
    for i in range(1, n_s + 1):
        t = i / n_s
        z = z_s_start + t * s_length_m
        if t <= 0.25:
            s = t / 0.25
            dx = direction * amplitude_m * _smoothstep(s)
        elif t <= 0.5:
            s = (t - 0.25) / 0.25
            dx = direction * amplitude_m * (1.0 - _smoothstep(s))
        elif t <= 0.75:
            s = (t - 0.5) / 0.25
            dx = -direction * amplitude_m * _smoothstep(s)
        else:
            s = (t - 0.75) / 0.25
            dx = -direction * amplitude_m * (1.0 - _smoothstep(s))
        pts.append((center_x + dx, z))

    # --- exit straight ---
    n_exit = 200 * S_CURVE_POINT_SCALE
    for i in range(1, n_exit + 1):
        t = i / n_exit
        z = z_s_end + t * (z_exit - z_s_end)
        pts.append((center_x, z))

    return pts, center_x, z_s_start, z_s_end


def _run_s_curve_trial(
    direction: float,
    label: str,
    max_s_error_m: float = 2.0,
    target_speed_mph: float = 20.0,
    controller_class=None,
) -> tuple[bool, str]:
    """Drive the car through an S-curve road and check lateral tracking.

    Success:
      * During S-curve  : max |lateral_error| < max_s_error_m (default 2.0 m)
      * Exit straight   : within ±0.5 m for 5 s continuously

    Returns ``(ok, message)``.
    """
    print(f"\n\n=== S-curve trial: {label} ===")

    mode = "eval" if VISUALIZE else "train"
    env = Environment(
        {"flat": True, "dt": 1.0 / S_CURVE_PHYSICS_HZ, "substeps": 1, "sun_time_hours": 12.0},
        mode=mode,
    )
    env.reset()

    assert env.terrain is not None
    drive_line, center_x, z_s_start, z_s_end = _build_s_curve_drive_line(
        env.terrain.width, env.terrain.height, direction=direction
    )
    print(
        f"  geometry: z_s_start={z_s_start:.0f} m, z_s_end={z_s_end:.0f} m, "
        f"amplitude=5 m, direction={direction:+.0f}, "
        f"control_hz={S_CURVE_CONTROL_HZ:.0f}, physics_hz={S_CURVE_PHYSICS_HZ:.0f}"
    )

    # Keep planner preview curvature consistent with the speed under test.
    # The PID feedforward uses planner future lat_accel, which scales with the
    # plan speed limit. A fixed 30 mph limit makes the 60 mph trial under-drive
    # feedforward and diverge from the sweep harness.
    speed_limits = (
        {"start_s": 0.0, "end_s": float("inf"), "speed_mph": target_speed_mph * 1.25},
    )
    plan = {
        "lane_width": 3.6, "lanes": 1, "shoulder": 1.5,
        "road_height": 0.02, "cross_pitch": 0.0,
        "ditch_width": 0.0, "ditch_depth": 0.0,
        "road_friction": 1.0,
        "drive_line": drive_line,
        "speed_limits": speed_limits,
    }
    env.plan = plan
    env.rp = drive_line
    env._planner.set_plan(drive_line, speed_limits)  # type: ignore[attr-defined]

    if VISUALIZE:
        env.init_renderer()
        ensure_drive_line_layer(env)

    # Place car on road at terrain.height * 0.25, heading forward at target speed.
    assert env.car is not None
    initial_mps = target_speed_mph / MPS_TO_MPH
    z0 = env.terrain.height * 0.25
    y0 = env.terrain.get_height(center_x, z0) + env.car.cg_height_m
    env.car.body.pos = np.array([center_x, y0, z0], dtype=float)
    env.car.body.vel = np.array([0.0, 0.0, initial_mps], dtype=float)
    env.car.body.angvel[:] = 0.0
    env.car.body.rot.arr[:] = (1.0, 0.0, 0.0, 0.0)
    env._prev_velocity = env.car.body.vel.copy()  # type: ignore[attr-defined]
    env._refresh_initial_telemetry()  # type: ignore[attr-defined]

    ctrl_cls = controller_class if controller_class is not None else PIDSteeringController
    try:
        controller = ctrl_cls(control_rate_hz=S_CURVE_CONTROL_HZ)
    except TypeError:
        controller = ctrl_cls()
    env.attach_controller(controller)
    controller.enable()

    max_speed_mph = target_speed_mph + 10.0
    throttle_cmd = 0.0
    throttle_alpha = 0.25

    max_time = 80.0
    steps = int(math.ceil(max_time / env.dt))

    s_max_error = 0.0
    exit_stable_time = 0.0
    exit_stable_req = 5.0
    exit_tol = 0.5
    past_s_curve = False
    drive_line_z = np.array([pt[1] for pt in drive_line], dtype=float)
    drive_line_x = np.array([pt[0] for pt in drive_line], dtype=float)
    trace: dict[str, list[float]] = {
        "time_s": [],
        "car_z_m": [],
        "car_x_m": [],
        "target_x_m": [],
        "speed_mph": [],
        "steer_cmd": [],
        "steer_delta": [],
        "controller_updated": [],
        "throttle_cmd": [],
        "brake_cmd": [],
        "lateral_error_m": [],
        "heading_error_rad": [],
        "effective_error_m": [],
        "effective_error_rate_mps": [],
        "integral_error_s": [],
        "p_term": [],
        "i_term": [],
        "d_term": [],
        "pid_out": [],
        "ff": [],
        "target_lat_accel": [],
        "preview_lat_accel": [],
        "current_lat_accel": [],
        "target_curvature": [],
        "preview_curvature": [],
        "current_curvature": [],
    }

    def finalize(ok: bool, message: str) -> tuple[bool, str]:
        if PID_DEBUG_PLOTS and (
            not PID_DEBUG_PLOT_LABEL
            or PID_DEBUG_PLOT_LABEL in label
            or PID_DEBUG_PLOT_LABEL in _current_test_name()
        ):
            _plot_s_curve_debug_trace(
                test_name=_current_test_name(),
                trial_label=label,
                trace=trace,
                z_s_start=z_s_start,
                z_s_end=z_s_end,
            )
        if PID_DEBUG_PLOT_DIR:
            _write_s_curve_debug_csv(
                test_name=_current_test_name(),
                trial_label=label,
                trace=trace,
            )
        env.attach_controller(None)
        return ok, message

    for step_idx in range(steps):
        assert env.car is not None
        speed_mph = float(np.linalg.norm(env.car.body.vel)) * MPS_TO_MPH
        car_z = float(env.car.body.pos[2])
        car_x = float(env.car.body.pos[0])

        speed_error = target_speed_mph - speed_mph
        brake_cmd = 0.0
        if speed_mph > max_speed_mph + 2.0:
            desired_throttle = 0.0
            brake_cmd = 0.25
        elif speed_mph > max_speed_mph + 0.5:
            desired_throttle = 0.0
            brake_cmd = 0.10
        elif speed_mph > target_speed_mph + 1.0:
            desired_throttle = 0.0  # coast back to target
        elif speed_error > 5.0:
            desired_throttle = 0.8
        elif speed_error > 2.0:
            desired_throttle = 0.6
        elif speed_error > 0.5:
            desired_throttle = 0.45
        elif speed_error > -0.5:
            desired_throttle = 0.10  # light maintenance throttle at target speed
        else:
            desired_throttle = 0.0   # coast when slightly over target
        throttle_cmd += throttle_alpha * (desired_throttle - throttle_cmd)

        _, _, terminated, truncated, info = env.step(DriverCommand(throttle=throttle_cmd, brake=brake_cmd))
        if terminated:
            return finalize(
                False,
                f"{label}: car left terrain at z={car_z:.0f} m",
            )
        assert not truncated, "simulation truncated"

        tele = env._build_snapshot(env._prev_velocity)  # type: ignore[attr-defined]
        lat_error = tele.target.lateral_error
        v_ego = max(float(tele.state.v_ego), 1e-6)
        target_x = float(np.interp(car_z, drive_line_z, drive_line_x))
        steer_cmd = float(info["driver_command"]["steer"])
        prev_steer_cmd = trace["steer_cmd"][-1] if trace["steer_cmd"] else 0.0
        trace["time_s"].append(step_idx * env.dt)
        trace["car_z_m"].append(car_z)
        trace["car_x_m"].append(car_x)
        trace["target_x_m"].append(target_x)
        trace["speed_mph"].append(speed_mph)
        trace["steer_cmd"].append(steer_cmd)
        trace["steer_delta"].append(steer_cmd - prev_steer_cmd)
        trace["controller_updated"].append(float(abs(steer_cmd - prev_steer_cmd) > 1e-9))
        trace["throttle_cmd"].append(throttle_cmd)
        trace["brake_cmd"].append(brake_cmd)
        trace["lateral_error_m"].append(float(tele.target.lateral_error))
        trace["heading_error_rad"].append(float(tele.target.heading_error))
        trace["effective_error_m"].append(float(controller._last_pid_error))  # type: ignore[attr-defined]
        trace["effective_error_rate_mps"].append(float(controller._last_de_dt))  # type: ignore[attr-defined]
        trace["integral_error_s"].append(float(controller._last_integral_state))  # type: ignore[attr-defined]
        trace["p_term"].append(float(controller._last_p_term))  # type: ignore[attr-defined]
        trace["i_term"].append(float(controller._last_i_term))  # type: ignore[attr-defined]
        trace["d_term"].append(float(controller._last_d_term))  # type: ignore[attr-defined]
        trace["pid_out"].append(float(controller._last_pid_out))  # type: ignore[attr-defined]
        trace["ff"].append(float(controller._last_ff))  # type: ignore[attr-defined]
        trace["target_lat_accel"].append(float(tele.target.lat_accel))
        trace["preview_lat_accel"].append(float(controller._last_base_target_lat))  # type: ignore[attr-defined]
        trace["current_lat_accel"].append(float(tele.state.lat_accel))
        trace["target_curvature"].append(float(tele.target.lat_accel / max(v_ego * v_ego, 1e-6)))
        trace["preview_curvature"].append(float(controller._last_base_target_lat / max(v_ego * v_ego, 1e-6)))  # type: ignore[attr-defined]
        trace["current_curvature"].append(float(tele.state.lat_accel / max(v_ego * v_ego, 1e-6)))

        if z_s_start <= car_z <= z_s_end:
            s_max_error = max(s_max_error, abs(lat_error))
            if s_max_error > max_s_error_m:
                return finalize(
                    False,
                    f"{label}: s_max_error exceeded {max_s_error_m:.1f}m ({s_max_error:.2f}m) at z={car_z:.0f}",
                )

        if car_z > z_s_end:
            past_s_curve = True
            if abs(lat_error) <= exit_tol:
                exit_stable_time += env.dt
                if exit_stable_time >= exit_stable_req:
                    return finalize(
                        True,
                        f"{label}: s_max_error={s_max_error:.2f} m, "
                        f"settled ±{exit_tol} m on exit straight",
                    )
            else:
                exit_stable_time = 0.0

    if not past_s_curve:
        return finalize(
            False,
            f"{label}: car never reached S-curve end (z_s_end={z_s_end:.0f} m)",
        )
    return finalize(
        False,
        f"{label}: s_max_error={s_max_error:.2f} m, "
        f"did not settle ±{exit_tol} m for {exit_stable_req} s "
        f"(exit_stable_time={exit_stable_time:.1f} s)",
    )


def _current_test_name() -> str:
    current = os.environ.get("PYTEST_CURRENT_TEST", "").strip()
    if not current:
        return "pytest"
    return current.split(" ", 1)[0]


def _plot_s_curve_debug_trace(
    *,
    test_name: str,
    trial_label: str,
    trace: dict[str, list[float]],
    z_s_start: float,
    z_s_end: float,
) -> None:
    if not trace["time_s"]:
        return

    import matplotlib.pyplot as plt

    time_s = np.array(trace["time_s"], dtype=float)
    car_z_m = np.array(trace["car_z_m"], dtype=float)
    in_s_curve = (car_z_m >= z_s_start) & (car_z_m <= z_s_end)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    fig.suptitle(f"{test_name}\nS-curve debug trace: {trial_label}")

    ax = axes[0, 0]
    ax.plot(time_s, trace["steer_cmd"], label="steer_cmd", linewidth=2.0, color="black")
    ax.plot(time_s, trace["ff"], label="feedforward", linewidth=1.5)
    ax.plot(time_s, trace["p_term"], label="P", linewidth=1.2)
    ax.plot(time_s, trace["i_term"], label="I", linewidth=1.2)
    ax.plot(time_s, trace["d_term"], label="D", linewidth=1.2)
    ax.plot(time_s, trace["pid_out"], label="PID sum", linewidth=1.2, linestyle="--")
    ax.set_ylabel("Steer command")
    ax.set_title("Controller Terms")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[0, 1]
    ax.plot(time_s, trace["car_x_m"], label="current x", linewidth=2.0)
    ax.plot(time_s, trace["target_x_m"], label="target x", linewidth=2.0, linestyle="--")
    ax.plot(time_s, trace["lateral_error_m"], label="lateral error", linewidth=1.2)
    ax.plot(time_s, trace["effective_error_m"], label="effective error", linewidth=1.2)
    ax.plot(time_s, trace["heading_error_rad"], label="heading error (rad)", linewidth=1.2)
    ax.set_ylabel("Metres / radians")
    ax.set_title("Current vs Target")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[1, 0]
    ax.plot(time_s, trace["current_lat_accel"], label="current lat accel", linewidth=1.8)
    ax.plot(time_s, trace["target_lat_accel"], label="target lat accel", linewidth=1.4, linestyle="--")
    ax.plot(time_s, trace["preview_lat_accel"], label="preview lat accel", linewidth=1.4)
    ax.plot(time_s, trace["current_curvature"], label="current curvature", linewidth=1.0)
    ax.plot(time_s, trace["target_curvature"], label="target curvature", linewidth=1.0, linestyle="--")
    ax.plot(time_s, trace["preview_curvature"], label="preview curvature", linewidth=1.0)
    ax.set_ylabel("m/s^2 and 1/m")
    ax.set_title("Curvature And Lateral Acceleration")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[1, 1]
    ax.plot(time_s, trace["speed_mph"], label="speed mph", linewidth=2.0)
    ax.plot(time_s, trace["throttle_cmd"], label="throttle", linewidth=1.2)
    ax.plot(time_s, trace["brake_cmd"], label="brake", linewidth=1.2)
    ax.plot(time_s, trace["car_z_m"], label="z position", linewidth=1.2)
    ax.set_ylabel("Mixed units")
    ax.set_title("Longitudinal Context")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    for ax in axes.flat:
        ax.set_xlabel("Time (s)")
        if np.any(in_s_curve):
            ax.axvspan(time_s[in_s_curve][0], time_s[in_s_curve][-1], color="orange", alpha=0.10)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    if PID_DEBUG_PLOT_DIR:
        out_dir = Path(PID_DEBUG_PLOT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        slug = _plot_slug(f"{test_name}_{trial_label}")
        fig.savefig(out_dir / f"{slug}.png", dpi=160, bbox_inches="tight")
    plt.show()


def _plot_slug(value: str) -> str:
    chars = []
    for ch in value.lower():
        if ch.isalnum():
            chars.append(ch)
        elif ch in {" ", "-", "_", "/", ":", "[", "]", "(", ")"}:
            chars.append("_")
    slug = "".join(chars).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "plot"


def _write_s_curve_debug_csv(
    *,
    test_name: str,
    trial_label: str,
    trace: dict[str, list[float]],
) -> None:
    out_dir = Path(PID_DEBUG_PLOT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = _plot_slug(f"{test_name}_{trial_label}")
    out_path = out_dir / f"{slug}.csv"
    fieldnames = list(trace.keys())
    row_count = len(trace[fieldnames[0]]) if fieldnames else 0

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(row_count):
            writer.writerow({name: trace[name][idx] for name in fieldnames})


def test_pid_follows_s_curve():
    """PID must track a straight → S-curve → straight road at 20, 40, and 60 mph.

    Success criteria:
      * S-curve section  : max |lateral_error| < 3.0 m  (60 % of 5 m amplitude)
      * Exit straight    : within ±0.5 m for 5 s continuously
    """
    trials = [
        (1.0, "left_first_20mph", 20.0, 3.0),
        (-1.0, "right_first_20mph", 20.0, 3.0),
        (1.0, "left_first_40mph", 40.0, 3.0),
        (-1.0, "right_first_40mph", 40.0, 3.0),
        (1.0, "left_first_60mph", 60.0, 3.0),
        (-1.0, "right_first_60mph", 60.0, 3.0),
    ]
    failures: list[str] = []

    for direction, label, speed_mph, max_err in trials:
        ok, msg = _run_s_curve_trial(
            direction, label, max_s_error_m=max_err, target_speed_mph=speed_mph
        )
        print(f"  {label}: {'PASS' if ok else 'FAIL'} — {msg}")
        if not ok:
            failures.append(msg)

    if failures:
        pytest.fail("\n".join(failures))

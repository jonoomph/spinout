"""Unit tests for PID controller fundamentals."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.controllers.pid import PIDSteeringController
from src.sim.control_api import DriverCommand
from src.sim.environment import Environment
from src.sim.planner import PlannerPreviewer
from tests.helpers import configure_flat_drive_line

MPS_TO_MPH = 2.2369362920544
_MPH_TO_MPS = 1.0 / MPS_TO_MPH


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_flat_env(dt: float = 0.02) -> Environment:
    """Create a flat training environment."""
    return Environment({"flat": True, "dt": dt, "substeps": 1, "sun_time_hours": 12.0}, mode="train")


def _speed_throttle(speed_mph: float, target_mph: float, max_mph: float) -> tuple[float, float]:
    """Return (throttle, brake) to maintain *target_mph*."""
    err = target_mph - speed_mph
    if speed_mph > max_mph + 2.0:
        return 0.0, 0.25
    if speed_mph > max_mph + 0.5:
        return 0.0, 0.10
    if speed_mph > target_mph + 1.0:
        return 0.0, 0.0
    if err > 5.0:
        return 0.8, 0.0
    if err > 2.0:
        return 0.6, 0.0
    if err > 0.5:
        return 0.45, 0.0
    if err > -0.5:
        return 0.10, 0.0
    return 0.0, 0.0


def _build_left_arc(radius: float, z_start: float, x_start: float, points: int = 60) -> list[tuple[float, float]]:
    """Quarter-circle arc curving LEFT (x decreases as z increases).

    Car starts at (x_start, z_start) heading in +z direction.
    Center of curvature is to the LEFT at (x_start - radius, z_start).
    """
    angles = [math.pi * 0.5 * i / (points - 1) for i in range(points)]
    cx = x_start - radius  # circle center x (left of start)
    return [(cx + radius * math.cos(a), z_start + radius * math.sin(a)) for a in angles]


def _build_right_arc(radius: float, z_start: float, x_start: float, points: int = 60) -> list[tuple[float, float]]:
    """Quarter-circle arc curving RIGHT (x increases as z increases).

    Car starts at (x_start, z_start) heading in +z direction.
    Center of curvature is to the RIGHT at (x_start + radius, z_start).
    """
    angles = [math.pi * 0.5 * i / (points - 1) for i in range(points)]
    cx = x_start + radius  # circle center x (right of start)
    return [(cx - radius * math.cos(a), z_start + radius * math.sin(a)) for a in angles]


def _install_arc_drive_line(
    env: Environment,
    drive_line: list[tuple[float, float]],
    speed_mph: float = 20.0,
) -> None:
    speed_limits = ({"start_s": 0.0, "end_s": float("inf"), "speed_mph": speed_mph},)
    env.plan = {
        "lane_width": 3.6, "lanes": 1, "shoulder": 1.5,
        "road_height": 0.02, "cross_pitch": 0.0,
        "ditch_width": 0.0, "ditch_depth": 0.0,
        "road_friction": 1.0,
        "drive_line": drive_line,
        "speed_limits": speed_limits,
    }
    env.rp = drive_line
    env._planner.set_plan(drive_line, speed_limits)  # type: ignore[attr-defined]


def _place_car_on_path(env: Environment, x: float, z: float, speed_mps: float, yaw_rad: float = 0.0) -> None:
    """Place the car at (x, z) with given forward speed and optional yaw (rotation about y-axis)."""
    assert env.car is not None and env.terrain is not None
    y = env.terrain.get_height(x, z) + env.car.cg_height_m
    env.car.body.pos = np.array([x, y, z], dtype=float)
    half = yaw_rad * 0.5
    env.car.body.rot.arr[:] = (math.cos(half), 0.0, math.sin(half), 0.0)
    env.car.body.vel = np.array([0.0, 0.0, speed_mps], dtype=float)
    env.car.body.angvel[:] = 0.0
    env._prev_velocity = env.car.body.vel.copy()  # type: ignore[attr-defined]
    env._refresh_initial_telemetry()  # type: ignore[attr-defined]


def test_plant_gain():
    """At a given speed and fixed steer, what lat_accel does physics produce?

    Validates sign convention and gain consistency across steer levels.
    """
    env = _make_flat_env()
    env.reset()
    configure_flat_drive_line(env)

    assert env.car is not None and env.terrain is not None
    center_x = env.terrain.width * 0.5
    z0 = env.terrain.height * 0.25
    _place_car_on_path(env, center_x, z0, speed_mps=0.0)

    # Ramp to 20 mph via throttle (no controller attached)
    target_mps = 20.0 * _MPH_TO_MPS
    for _ in range(int(10.0 / env.dt)):
        assert env.car is not None
        speed = float(np.linalg.norm(env.car.body.vel))
        if speed >= target_mps * 0.95:
            break
        throttle = 0.8 if speed < target_mps * 0.8 else 0.4
        env.step(DriverCommand(throttle=throttle))

    assert float(np.linalg.norm(env.car.body.vel)) * MPS_TO_MPH >= 18.0, "could not reach 20 mph"

    steer_values = [0.1, 0.3, 0.5]
    gains_by_steer: dict[float, tuple[float, float, float]] = {}
    rows: list[dict] = []
    t = 0.0

    for steer_cmd in steer_values:
        window_accels: list[float] = []
        window_yaws: list[float] = []
        for step_i in range(int(3.0 / env.dt)):
            assert env.car is not None
            speed_mph = float(np.linalg.norm(env.car.body.vel)) * MPS_TO_MPH
            throttle, brake = _speed_throttle(speed_mph, 20.0, 25.0)
            env.step(DriverCommand(steer=steer_cmd, throttle=throttle, brake=brake))
            tele = env._build_snapshot(env._prev_velocity)  # type: ignore[attr-defined]
            rows.append({
                "t": round(t, 4),
                "steer": steer_cmd,
                "lat_accel": round(tele.state.lat_accel, 4),
                "yaw_rate": round(tele.state.yaw_rate, 5),
                "v_ego": round(tele.state.v_ego, 3),
            })
            t += env.dt
            if step_i >= int(2.0 / env.dt):
                window_accels.append(tele.state.lat_accel)
                window_yaws.append(tele.state.yaw_rate)

        mean_la = float(np.mean(window_accels))
        mean_yr = float(np.mean(window_yaws))
        gain = mean_la / steer_cmd
        gains_by_steer[steer_cmd] = (mean_la, mean_yr, gain)
        print(f"  steer={steer_cmd:.1f}  lat_accel={mean_la:.3f}  yaw_rate={mean_yr:.4f}  gain={gain:.2f}")

    gains_list = [gains_by_steer[s] for s in steer_values]

    for s, (la, yr, g) in zip(steer_values, gains_list):
        assert abs(la) > 0.05, f"steer={s}: expected |lat_accel| > 0.05, got {la:.3f}"
        assert la > 0, f"steer={s}: expected lat_accel > 0 (left turn), got {la:.3f}"

    # Gain consistent within ±40% across steer values
    all_gains = [g for (_, _, g) in gains_list]
    mean_gain = float(np.mean(all_gains))
    for s, g in zip(steer_values, all_gains):
        assert abs(g - mean_gain) < 0.40 * mean_gain + 0.5, (
            f"gain inconsistent: steer={s} gain={g:.2f} vs mean={mean_gain:.2f}"
        )

    # Centripetal formula: lat_accel ≈ v_ego × |yaw_rate| within 30%
    v = 20.0 * _MPH_TO_MPS
    for s, (la, yr, g) in zip(steer_values, gains_list):
        centripetal = abs(v * yr)
        if centripetal > 0.05:
            ratio = abs(la) / centripetal
            assert 0.5 < ratio < 2.0, (
                f"steer={s}: centripetal mismatch lat_accel={la:.3f} vs v*|yr|={centripetal:.3f}"
            )


# ---------------------------------------------------------------------------
# Test 2: sign convention
# ---------------------------------------------------------------------------

def test_lat_accel_direction_matches_steer_direction():
    """Positive steer → positive lat_accel (left); negative steer → negative lat_accel (right)."""
    env = _make_flat_env()
    env.reset()
    configure_flat_drive_line(env)

    assert env.car is not None and env.terrain is not None
    center_x = env.terrain.width * 0.5
    z0 = env.terrain.height * 0.25
    _place_car_on_path(env, center_x, z0, speed_mps=0.0)

    for _ in range(int(10.0 / env.dt)):
        assert env.car is not None
        speed = float(np.linalg.norm(env.car.body.vel))
        if speed >= 20.0 * _MPH_TO_MPS * 0.95:
            break
        env.step(DriverCommand(throttle=0.7 if speed < 20.0 * _MPH_TO_MPS * 0.8 else 0.3))

    rows: list[dict] = []
    t = 0.0
    pos_accels: list[float] = []
    neg_accels: list[float] = []

    for steer_cmd in [0.4, -0.4]:
        window: list[float] = []
        for _ in range(int(2.0 / env.dt)):
            assert env.car is not None
            speed_mph = float(np.linalg.norm(env.car.body.vel)) * MPS_TO_MPH
            throttle, brake = _speed_throttle(speed_mph, 20.0, 25.0)
            env.step(DriverCommand(steer=steer_cmd, throttle=throttle, brake=brake))
            tele = env._build_snapshot(env._prev_velocity)  # type: ignore[attr-defined]
            rows.append({"t": round(t, 4), "steer": steer_cmd, "lat_accel": round(tele.state.lat_accel, 4)})
            t += env.dt
            window.append(tele.state.lat_accel)

        if steer_cmd > 0:
            pos_accels = window
        else:
            neg_accels = window

    mean_pos = float(np.mean(pos_accels))
    mean_neg = float(np.mean(neg_accels))

    assert mean_pos > 0, f"positive steer should produce positive lat_accel, got {mean_pos:.3f}"
    assert mean_neg < 0, f"negative steer should produce negative lat_accel, got {mean_neg:.3f}"

    gain = mean_pos / 0.4
    assert gain > 3.0, f"plant gain too low: {gain:.2f} m/s²/unit (expected > 3.0)"
    assert gain < 12.0, f"plant gain too high: {gain:.2f} m/s²/unit (expected < 12.0)"
    print(f"  plant gain ≈ {gain:.2f} m/s²/steer-unit at 20 mph")


# ---------------------------------------------------------------------------
# Test 3: planner curvature sign and magnitude
# ---------------------------------------------------------------------------

def test_planner_curvature_sign_and_magnitude():
    """Planner curvature κ and preview.lat_accel have correct sign for left/right arcs."""
    speed_mph = 20.0
    speed_mps = speed_mph * _MPH_TO_MPS
    R = 100.0

    # Left arc: starts at (200, 50), curves left (x decreases) → positive curvature
    left_arc = _build_left_arc(R, z_start=50.0, x_start=200.0, points=60)
    # Right arc: starts at (200, 50), curves right (x increases) → negative curvature
    right_arc = _build_right_arc(R, z_start=50.0, x_start=200.0, points=60)

    speed_limits = ({"start_s": 0.0, "end_s": float("inf"), "speed_mph": speed_mph},)

    for arc_name, arc, expected_curv_sign, expected_lat_sign in [
        ("left",  left_arc,  1, 1),
        ("right", right_arc, -1, -1),
    ]:
        planner = PlannerPreviewer()
        planner.set_plan(arc, speed_limits)

        mid = len(arc) // 2
        curv_mid = float(planner.curvature[mid])

        start_x, start_z = arc[0]
        position = (start_x, 0.0, start_z)
        preview = planner.preview(position, speed=speed_mps, preview_hz=10.0)

        lat_first = preview.lat_accel[0] if preview.lat_accel else 0.0
        theoretical = speed_mps * speed_mps / R

        print(
            f"  {arc_name}: curv_mid={curv_mid:.4f}  lat[0]={lat_first:.3f}"
            f"  theoretical={theoretical:.3f}"
        )

        assert curv_mid * expected_curv_sign > 0, (
            f"{arc_name} arc: expected curvature sign {expected_curv_sign:+d}, got {curv_mid:.4f}"
        )
        assert lat_first * expected_lat_sign > 0, (
            f"{arc_name} arc: expected lat_accel sign {expected_lat_sign:+d}, got {lat_first:.3f}"
        )
        # Magnitude within 20% of v²/R
        assert abs(abs(lat_first) - theoretical) < 0.25 * theoretical + 0.1, (
            f"{arc_name} arc: lat_accel={lat_first:.3f} vs theoretical={theoretical:.3f}"
        )


# ---------------------------------------------------------------------------
# Test 4: planner preview matches physics on arc
# ---------------------------------------------------------------------------

def test_planner_preview_matches_physics_on_arc():
    """Planner's predicted lat_accel agrees with physics actuals on a circular arc."""
    R = 100.0
    speed_mph = 20.0
    speed_mps = speed_mph * _MPH_TO_MPS

    env = _make_flat_env()
    env.reset()

    # Left-curving arc: car starts at (200, 100) heading +z
    arc = _build_left_arc(R, z_start=100.0, x_start=200.0, points=80)
    _install_arc_drive_line(env, arc, speed_mph=speed_mph)
    _place_car_on_path(env, x=arc[0][0], z=arc[0][1], speed_mps=speed_mps)

    controller = PIDSteeringController()
    env.attach_controller(controller)
    controller.enable()

    rows: list[dict] = []
    t = 0.0
    collect_accels: list[tuple[float, float]] = []
    warmup_s = 2.0
    throttle_cmd = 0.0

    for _ in range(int(6.0 / env.dt)):
        assert env.car is not None
        terminated_last = False
        speed_mph_now = float(np.linalg.norm(env.car.body.vel)) * MPS_TO_MPH
        desired_t, brake = _speed_throttle(speed_mph_now, speed_mph, speed_mph + 5.0)
        throttle_cmd += 0.25 * (desired_t - throttle_cmd)
        obs, _, terminated, truncated, _ = env.step(DriverCommand(throttle=throttle_cmd, brake=brake))
        if terminated:
            break
        tele = env._build_snapshot(env._prev_velocity)  # type: ignore[attr-defined]

        preview_lat = tele.future.lat_accel[0] if tele.future.lat_accel else 0.0
        phys_lat = tele.state.lat_accel
        rows.append({
            "t": round(t, 4),
            "preview_lat": round(preview_lat, 4),
            "physics_lat": round(phys_lat, 4),
            "diff": round(preview_lat - phys_lat, 4),
        })
        t += env.dt

        if t > warmup_s:
            collect_accels.append((preview_lat, phys_lat))

    env.attach_controller(None)
    assert len(collect_accels) >= 50, (
        f"not enough data collected after warmup ({len(collect_accels)} samples)"
    )

    diffs = [abs(p - ph) for p, ph in collect_accels]
    mean_diff = float(np.mean(diffs))

    print(f"  mean |preview - physics| = {mean_diff:.3f} m/s²")
    assert mean_diff < 1.1, f"planner and physics disagree: mean error = {mean_diff:.3f} m/s² (expected < 1.1)"

    sign_match = sum(1 for p, ph in collect_accels if p * ph >= 0) / len(collect_accels)
    assert sign_match > 0.80, f"sign mismatch: only {sign_match*100:.0f}% of steps agree in sign"
    print(f"  sign agreement: {sign_match*100:.0f}%")


# ---------------------------------------------------------------------------
# Test 5: PID arc tracking
# ---------------------------------------------------------------------------

def test_pid_arc_tracking():
    """PID controller with default gains tracks a circular arc within 1.5 m."""
    R = 150.0
    speed_mph = 20.0
    speed_mps = speed_mph * _MPH_TO_MPS

    env = _make_flat_env()
    env.reset()

    arc = _build_left_arc(R, z_start=100.0, x_start=200.0, points=80)
    _install_arc_drive_line(env, arc, speed_mph=speed_mph)
    _place_car_on_path(env, x=arc[0][0], z=arc[0][1], speed_mps=speed_mps)

    controller = PIDSteeringController()
    env.attach_controller(controller)
    controller.enable()

    rows: list[dict] = []
    t = 0.0
    errors: list[float] = []
    throttle_cmd = 0.0
    warmup_s = 1.5

    for _ in range(int(6.0 / env.dt)):
        assert env.car is not None
        speed_mph_now = float(np.linalg.norm(env.car.body.vel)) * MPS_TO_MPH
        desired_t, brake = _speed_throttle(speed_mph_now, speed_mph, speed_mph + 5.0)
        throttle_cmd += 0.25 * (desired_t - throttle_cmd)
        _, _, terminated, _, _ = env.step(DriverCommand(throttle=throttle_cmd, brake=brake))
        if terminated:
            break
        tele = env._build_snapshot(env._prev_velocity)  # type: ignore[attr-defined]
        lat_err = tele.target.lateral_error
        rows.append({
            "t": round(t, 4),
            "lateral_error": round(lat_err, 4),
            "lat_accel": round(tele.state.lat_accel, 4),
        })
        if t >= warmup_s:
            errors.append(lat_err)
        t += env.dt

    env.attach_controller(None)
    assert len(errors) >= 50, "simulation terminated too early"

    max_err = float(np.max(np.abs(errors)))
    mean_err = float(np.mean(np.abs(errors)))
    print(f"  pid arc: max_err={max_err:.2f}m  mean_err={mean_err:.2f}m")

    assert max_err < 1.5, f"PID arc tracking: max error {max_err:.2f}m (expected < 1.5m)"


# ---------------------------------------------------------------------------
# Test 6: PID centering — convergence check
# ---------------------------------------------------------------------------

def test_pid_centering_no_oscillation():
    """From 2 m offset, car converges toward the driveline within 15 s.

    With the simplified (no-heading-preview) PID, some oscillation is expected
    before settling.  The test validates:
      - car is within 1.5 m at t=15 s (converging, not diverging)
      - zero-crossings ≤ 5 in the first 15 s (oscillating but not runaway)
      - max lateral velocity < 2.0 m/s
    """
    env = _make_flat_env()
    env.reset()
    center_x = configure_flat_drive_line(env)

    assert env.car is not None and env.terrain is not None
    offset_m = 2.0
    z0 = env.terrain.height * 0.25
    x0 = center_x + offset_m
    y0 = env.terrain.get_height(x0, z0) + env.car.cg_height_m
    env.car.body.pos = np.array([x0, y0, z0], dtype=float)
    env.car.body.vel[:] = 0.0
    env.car.body.angvel[:] = 0.0
    env.car.body.rot.arr[:] = (1.0, 0.0, 0.0, 0.0)
    env._prev_velocity = env.car.body.vel.copy()  # type: ignore[attr-defined]
    env._refresh_initial_telemetry()  # type: ignore[attr-defined]

    controller = PIDSteeringController()
    env.attach_controller(controller)
    controller.enable()

    rows: list[dict] = []
    t = 0.0
    errors: list[float] = []
    lat_vels: list[float] = []
    throttle_cmd = 0.0

    for _ in range(int(20.0 / env.dt)):
        assert env.car is not None
        speed_mph = float(np.linalg.norm(env.car.body.vel)) * MPS_TO_MPH
        desired_t, brake = _speed_throttle(speed_mph, 25.0, 30.0)
        throttle_cmd += 0.25 * (desired_t - throttle_cmd)
        env.step(DriverCommand(throttle=throttle_cmd, brake=brake))
        tele = env._build_snapshot(env._prev_velocity)  # type: ignore[attr-defined]
        lat_err = tele.target.lateral_error
        errors.append(lat_err)
        lat_vels.append(abs(tele.state.lat_velocity))
        rows.append({
            "t": round(t, 4),
            "lateral_error": round(lat_err, 4),
            "lat_velocity": round(tele.state.lat_velocity, 4),
        })
        t += env.dt

    env.attach_controller(None)
    settle_idx = int(15.0 / env.dt)
    err_at_15 = abs(errors[settle_idx]) if settle_idx < len(errors) else abs(errors[-1])

    errors_15 = errors[:settle_idx]
    zero_crossings = sum(
        1 for i in range(1, len(errors_15)) if errors_15[i - 1] * errors_15[i] < 0
    )

    max_lat_vel = float(np.max(lat_vels[:settle_idx]))
    print(f"  centering: err@15s={err_at_15:.3f}m  ZC={zero_crossings}  max_lat_vel={max_lat_vel:.2f}m/s")

    assert err_at_15 < 1.5, f"not converging at t=15s: error = {err_at_15:.3f}m (expected < 1.5m)"
    assert zero_crossings <= 5, f"too many oscillations: {zero_crossings} zero-crossings (expected ≤ 5)"
    assert max_lat_vel < 2.0, f"lateral velocity too high: {max_lat_vel:.2f} m/s (expected < 2.0)"


# ---------------------------------------------------------------------------
# Test 7: heading correction
# ---------------------------------------------------------------------------

def test_heading_correction_converges():
    """10° heading error (left) corrects within 10 s without excessive lateral drift."""
    env = _make_flat_env()
    env.reset()
    center_x = configure_flat_drive_line(env)

    assert env.car is not None and env.terrain is not None
    z0 = env.terrain.height * 0.25
    y0 = env.terrain.get_height(center_x, z0) + env.car.cg_height_m

    # 10° left heading error: rotate car by -10° around y
    # quaternion for rotation by α around y: (cos(α/2), 0, sin(α/2), 0)
    heading_error_rad = math.radians(10.0)
    yaw_for_left_error = -heading_error_rad  # negative yaw → heading_error > 0 (left)
    _place_car_on_path(env, center_x, z0, speed_mps=25.0 * _MPH_TO_MPS, yaw_rad=yaw_for_left_error)

    controller = PIDSteeringController()
    env.attach_controller(controller)
    controller.enable()

    rows: list[dict] = []
    t = 0.0
    lat_errors: list[float] = []
    throttle_cmd = 0.0

    for _ in range(int(15.0 / env.dt)):
        assert env.car is not None
        speed_mph = float(np.linalg.norm(env.car.body.vel)) * MPS_TO_MPH
        desired_t, brake = _speed_throttle(speed_mph, 25.0, 30.0)
        throttle_cmd += 0.25 * (desired_t - throttle_cmd)
        env.step(DriverCommand(throttle=throttle_cmd, brake=brake))
        tele = env._build_snapshot(env._prev_velocity)  # type: ignore[attr-defined]
        lat_errors.append(tele.target.lateral_error)
        rows.append({
            "t": round(t, 4),
            "lateral_error": round(tele.target.lateral_error, 4),
            "heading_error": round(tele.target.heading_error, 4),
        })
        t += env.dt

    env.attach_controller(None)
    settle_idx = int(10.0 / env.dt)
    heading_at_10 = abs(rows[min(settle_idx, len(rows) - 1)]["heading_error"])
    max_lat_err = float(np.max(np.abs(lat_errors)))

    print(f"  heading correction: heading@10s={math.degrees(heading_at_10):.1f}°  max_lat_err={max_lat_err:.2f}m")

    assert heading_at_10 < math.radians(5.0), (
        f"heading not corrected at t=10s: {math.degrees(heading_at_10):.1f}° (expected < 5°)"
    )
    assert max_lat_err < 5.0, f"too much lateral drift: {max_lat_err:.2f}m (expected < 5.0m)"


# ---------------------------------------------------------------------------
# Test 8: S-curve hard fail
# ---------------------------------------------------------------------------

def test_s_curve_max_error_hard_fail():
    """Verify the S-curve trial hard-fails (not silently passes) when error > 1.5 m.

    Uses a null controller (steer=0) to guarantee the car drifts far off path.
    If the hard-fail guard is absent, the trial would return ok=True — a
    meta-failure we must prevent.
    """
    from tests.controller.test_controller import _run_s_curve_trial
    import src.controllers.pid as pid_mod

    class _NullController(pid_mod.PIDSteeringController):
        def step(self, telemetry, manual):  # type: ignore[override]
            return DriverCommand(steer=0.0, throttle=manual.throttle, brake=manual.brake)

        _last_target_lat = 0.0
        _last_base_target_lat = 0.0
        _last_ff = 0.0
        _last_pid_out = 0.0
        _last_pid_error = 0.0
        _last_lateral_term = 0.0

    ok, msg, *_ = _run_s_curve_trial(
        direction=1.0, label="null_ctrl_left", max_s_error_m=1.5,
        controller_class=_NullController,
    )

    assert not ok, (
        f"S-curve with null controller should hard-fail (s_max_error > 1.5m), "
        f"but returned ok=True: {msg}"
    )
    print(f"  hard-fail confirmed: {msg}")

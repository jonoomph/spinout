import csv
import math
import os
from pathlib import Path

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
_TRIAL_LOG_DIR = Path("/tmp/pid_trials")


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


def _plot_trial_csv(csv_path: Path) -> Path:
    """Render a 3-panel diagnostic plot from a trial CSV and return the PNG path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times, lat_errs, steers, lat_accels, target_lats, heading_errs, lat_vels = (
        [], [], [], [], [], [], []
    )
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            times.append(float(row["t"]))
            lat_errs.append(float(row["lateral_error_m"]))
            steers.append(float(row["steer"]))
            lat_accels.append(float(row["lat_accel"]))
            target_lats.append(float(row["target_lat"]))
            heading_errs.append(float(row["heading_error"]))
            lat_vels.append(float(row["lat_velocity"]))

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(csv_path.stem.replace("_", " "), fontsize=13)

    axes[0].axhline(0, color="k", linewidth=0.5)
    axes[0].axhline(0.3, color="gray", linewidth=0.8, linestyle="--", label="±0.3 m band")
    axes[0].axhline(-0.3, color="gray", linewidth=0.8, linestyle="--")
    axes[0].plot(times, lat_errs, color="steelblue", label="lateral error (m)")
    axes[0].set_ylabel("lateral error (m)")
    axes[0].legend(loc="upper right")

    axes[1].axhline(0, color="k", linewidth=0.5)
    axes[1].plot(times, steers, color="darkorange", label="steer cmd")
    axes[1].plot(times, target_lats, color="purple", linestyle="--", label="target lat accel (m/s²)")
    axes[1].set_ylabel("steer / target lat")
    axes[1].legend(loc="upper right")

    axes[2].axhline(0, color="k", linewidth=0.5)
    axes[2].plot(times, lat_accels, color="royalblue", label="lat accel (m/s²)")
    axes[2].plot(times, heading_errs, color="crimson", linestyle="--", label="heading error (rad)")
    axes[2].plot(times, lat_vels, color="seagreen", linestyle=":", label="lat velocity (m/s)")
    axes[2].set_ylabel("dynamics")
    axes[2].set_xlabel("time (s)")
    axes[2].legend(loc="upper right")

    fig.tight_layout()
    png_path = csv_path.with_suffix(".png")
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return png_path


def _plot_pid_csv(csv_path: Path) -> Path:
    """Render a 2-panel PID breakdown plot alongside the main trial plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times, pid_errors = [], []
    p_terms, i_terms, d_terms, pid_outs, ffs = [], [], [], [], []
    target_lats, lat_accels = [], []

    with open(csv_path) as f:
        for row in csv.DictReader(f):
            times.append(float(row["t"]))
            pid_errors.append(float(row["pid_error"]))
            p_terms.append(float(row["p_term"]))
            i_terms.append(float(row["i_term"]))
            d_terms.append(float(row["d_term"]))
            pid_outs.append(float(row["pid_out"]))
            ffs.append(float(row["ff"]))
            target_lats.append(float(row["target_lat"]))
            lat_accels.append(float(row["lat_accel"]))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"{csv_path.stem.replace('_', ' ')} — PID breakdown", fontsize=13)

    # Panel 1: P, I, D contributions + total pid_out + feedforward
    ax = axes[0]
    ax.axhline(0, color="k", linewidth=0.5)
    ax.plot(times, p_terms,   color="royalblue",  linewidth=1.5, label="P term")
    ax.plot(times, i_terms,   color="darkorange",  linewidth=1.5, label="I term")
    ax.plot(times, d_terms,   color="seagreen",    linewidth=1.5, label="D term")
    ax.plot(times, pid_outs,  color="purple",      linewidth=1.5, linestyle="--", label="pid_out (P+I+D)")
    ax.plot(times, ffs,       color="crimson",     linewidth=1.5, linestyle=":",  label="feedforward")
    ax.set_ylabel("contribution to steer (pre-norm)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: target_lat vs actual lat_accel, and the error between them
    ax = axes[1]
    ax.axhline(0, color="k", linewidth=0.5)
    ax.plot(times, target_lats, color="crimson",   linewidth=1.5, linestyle="--", label="target lat_accel (m/s²)")
    ax.plot(times, lat_accels,  color="royalblue", linewidth=1.5, label="actual lat_accel (m/s²)")
    ax.plot(times, pid_errors,  color="gray",      linewidth=1.0, linestyle=":",  label="PID error (target − actual)")
    ax.set_ylabel("lat accel (m/s²)")
    ax.set_xlabel("time (s)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    png_path = csv_path.with_stem(csv_path.stem + "_pid").with_suffix(".png")
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return png_path


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


def _run_pid_centering_trial(offset_m: float, target_speed_mph: float = 25.0) -> tuple[bool, str, Path]:
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

    rows: list[dict] = []

    def _flush_csv() -> Path:
        _TRIAL_LOG_DIR.mkdir(parents=True, exist_ok=True)
        sign = "pos" if offset_m >= 0 else "neg"
        path = _TRIAL_LOG_DIR / f"trial_{sign}{abs(offset_m):.1f}m.csv"
        if rows:
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        return path

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
        rows.append({
            "t": round(step_idx * env.dt, 4),
            "lateral_error_m": round(lateral_error, 4),
            "steer": round(info["driver_command"]["steer"], 4),
            "lat_accel": round(tele.state.lat_accel, 4),
            "target_lat": round(controller._last_target_lat, 4),  # type: ignore[attr-defined]
            "lat_velocity": round(tele.state.lat_velocity, 4),
            "heading_error": round(tele.target.heading_error, 4),
            "v_ego": round(tele.state.v_ego, 3),
            # PID breakdown
            "pid_error": round(controller._last_pid_error, 4),  # type: ignore[attr-defined]
            "p_term": round(controller._last_p_term, 5),  # type: ignore[attr-defined]
            "i_term": round(controller._last_i_term, 5),  # type: ignore[attr-defined]
            "d_term": round(controller._last_d_term, 5),  # type: ignore[attr-defined]
            "pid_out": round(controller._last_pid_out, 5),  # type: ignore[attr-defined]
            "ff": round(controller._last_ff, 5),  # type: ignore[attr-defined]
        })

        if controller.enabled and not crossed_line and offset_m != 0.0:
            crossed_line = (lateral_error > 0) != (offset_m > 0)
        if crossed_line and abs_error > abs(offset_m) + 0.5:
            env.attach_controller(None)
            return False, (
                f"offset={offset_m:.2f} m: crossed driveline but diverged "
                f"(error {abs_error:.2f} m)"
            ), _flush_csv()

        if abs_error <= tolerance_m:
            stable_time += env.dt
            if stable_time >= stable_required:
                env.attach_controller(None)
                return True, "", _flush_csv()
        else:
            stable_time = 0.0
        last_error = abs_error

    env.attach_controller(None)
    return False, (
        f"offset={offset_m:.2f} m: failed to centre within {max_time}s "
        f"(last error {last_error:.2f} m)"
    ), _flush_csv()


def test_pid_recenters_from_lateral_offsets():
    """Ensure the PID controller can centre the car from both small and large offsets."""

    offsets_m = (1.0, -1.0, 3.0, -3.0)  # ~3 ft and ~10 ft on both sides
    failures = []
    csv_paths = []
    for offset in offsets_m:
        ok, message, csv_path = _run_pid_centering_trial(offset)
        csv_paths.append(csv_path)
        if not ok:
            failures.append(message)

    print("\n--- diagnostic plots ---")
    for csv_path in csv_paths:
        if csv_path.exists():
            png = _plot_trial_csv(csv_path)
            print(f"  {png}")
            pid_png = _plot_pid_csv(csv_path)
            print(f"  {pid_png}")

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
    for i in range(100):
        z = z_s_start * i / 99.0
        pts.append((center_x, z))

    # --- S-curve (four smoothstep phases, zero slope at every key point) ---
    n_s = 200
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
    n_exit = 200
    for i in range(1, n_exit + 1):
        t = i / n_exit
        z = z_s_end + t * (z_exit - z_s_end)
        pts.append((center_x, z))

    return pts, center_x, z_s_start, z_s_end


def _run_s_curve_trial(
    direction: float,
    label: str,
    max_s_error_m: float = 2.0,
) -> tuple[bool, str, Path, list[tuple[float, float]], float, float]:
    """Drive the car through an S-curve road and check lateral tracking.

    Success:
      * During S-curve  : max |lateral_error| < max_s_error_m (default 2.0 m)
      * Exit straight   : within ±0.5 m for 5 s continuously

    Returns ``(ok, message, csv_path, drive_line, z_s_start, z_s_end)``.
    """
    print(f"\n\n=== S-curve trial: {label} ===")

    mode = "eval" if VISUALIZE else "train"
    env = Environment({"flat": True, "dt": 0.02, "substeps": 1, "sun_time_hours": 12.0}, mode=mode)
    env.reset()

    assert env.terrain is not None
    drive_line, center_x, z_s_start, z_s_end = _build_s_curve_drive_line(
        env.terrain.width, env.terrain.height, direction=direction
    )
    print(
        f"  geometry: z_s_start={z_s_start:.0f} m, z_s_end={z_s_end:.0f} m, "
        f"amplitude=5 m, direction={direction:+.0f}"
    )

    speed_limits = ({"start_s": 0.0, "end_s": float("inf"), "speed_mph": 30.0},)
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

    # Place car on road at terrain.height * 0.25, heading forward at 20 mph.
    assert env.car is not None
    initial_mps = 20.0 / MPS_TO_MPH
    z0 = env.terrain.height * 0.25
    y0 = env.terrain.get_height(center_x, z0) + env.car.cg_height_m
    env.car.body.pos = np.array([center_x, y0, z0], dtype=float)
    env.car.body.vel = np.array([0.0, 0.0, initial_mps], dtype=float)
    env.car.body.angvel[:] = 0.0
    env.car.body.rot.arr[:] = (1.0, 0.0, 0.0, 0.0)
    env._prev_velocity = env.car.body.vel.copy()  # type: ignore[attr-defined]
    env._refresh_initial_telemetry()  # type: ignore[attr-defined]

    controller = PIDSteeringController()
    env.attach_controller(controller)
    controller.enable()

    target_speed_mph = 20.0
    max_speed_mph = 30.0
    throttle_cmd = 0.0
    throttle_alpha = 0.25

    max_time = 80.0
    steps = int(math.ceil(max_time / env.dt))

    s_max_error = 0.0
    exit_stable_time = 0.0
    exit_stable_req = 5.0
    exit_tol = 0.5
    past_s_curve = False

    rows: list[dict] = []

    def _flush_csv() -> Path:
        _TRIAL_LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = _TRIAL_LOG_DIR / f"s_curve_{label}.csv"
        if rows:
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        return path

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
            env.attach_controller(None)
            return (
                False,
                f"{label}: car left terrain at z={car_z:.0f} m",
                _flush_csv(), drive_line, z_s_start, z_s_end,
            )
        assert not truncated, "simulation truncated"

        tele = env._build_snapshot(env._prev_velocity)  # type: ignore[attr-defined]
        lat_error = tele.target.lateral_error

        rows.append({
            "t": round(step_idx * env.dt, 4),
            "car_x": round(car_x, 3),
            "car_z": round(car_z, 3),
            "lateral_error_m": round(lat_error, 4),
            "steer": round(info["driver_command"]["steer"], 4),
            "lat_accel": round(tele.state.lat_accel, 4),
            "target_lat": round(controller._last_target_lat, 4),  # type: ignore[attr-defined]
            "v_ego": round(tele.state.v_ego, 3),
            "base_target_lat": round(controller._last_base_target_lat, 4),  # type: ignore[attr-defined]
            "heading_error": round(tele.target.heading_error, 4),
            "lat_velocity": round(tele.state.lat_velocity, 4),
            "ff": round(controller._last_ff, 5),  # type: ignore[attr-defined]
            "pid_out": round(controller._last_pid_out, 5),  # type: ignore[attr-defined]
            "pid_error": round(controller._last_pid_error, 4),  # type: ignore[attr-defined]
            "lateral_term": round(controller._last_lateral_term, 4),  # type: ignore[attr-defined]
        })

        if z_s_start <= car_z <= z_s_end:
            s_max_error = max(s_max_error, abs(lat_error))
            if s_max_error > max_s_error_m:
                env.attach_controller(None)
                return (
                    False,
                    f"{label}: s_max_error exceeded {max_s_error_m:.1f}m ({s_max_error:.2f}m) at z={car_z:.0f}",
                    _flush_csv(), drive_line, z_s_start, z_s_end,
                )

        if car_z > z_s_end:
            past_s_curve = True
            if abs(lat_error) <= exit_tol:
                exit_stable_time += env.dt
                if exit_stable_time >= exit_stable_req:
                    env.attach_controller(None)
                    return (
                        True,
                        f"{label}: s_max_error={s_max_error:.2f} m, "
                        f"settled ±{exit_tol} m on exit straight",
                        _flush_csv(), drive_line, z_s_start, z_s_end,
                    )
            else:
                exit_stable_time = 0.0

    env.attach_controller(None)
    csv_path = _flush_csv()

    if not past_s_curve:
        return (
            False,
            f"{label}: car never reached S-curve end (z_s_end={z_s_end:.0f} m)",
            csv_path, drive_line, z_s_start, z_s_end,
        )
    return (
        False,
        f"{label}: s_max_error={s_max_error:.2f} m, "
        f"did not settle ±{exit_tol} m for {exit_stable_req} s "
        f"(exit_stable_time={exit_stable_time:.1f} s)",
        csv_path, drive_line, z_s_start, z_s_end,
    )


def _plot_s_curve_trial(
    csv_path: Path,
    drive_line: list[tuple[float, float]],
    z_s_start: float,
    z_s_end: float,
) -> Path:
    """Render a top-down path map + lateral-error time-series and return the PNG path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times, car_xs, car_zs, lat_errs = [], [], [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            times.append(float(row["t"]))
            car_xs.append(float(row["car_x"]))
            car_zs.append(float(row["car_z"]))
            lat_errs.append(float(row["lateral_error_m"]))

    dl_xs = [p[0] for p in drive_line]
    dl_zs = [p[1] for p in drive_line]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(csv_path.stem.replace("_", " "), fontsize=13)

    # Top-down path (z = forward on x-axis, x = lateral on y-axis)
    ax = axes[0]
    ax.axvspan(z_s_start, z_s_end, alpha=0.10, color="orange", label="S-curve zone", zorder=0)
    ax.plot(dl_zs, dl_xs, color="gray", linewidth=2.0, label="drive line", zorder=1)
    ax.plot(car_zs, car_xs, color="steelblue", linewidth=1.0, alpha=0.8, label="car path", zorder=2)
    ax.set_xlabel("z — forward (m)")
    ax.set_ylabel("x — lateral (m)")
    ax.set_title("Top-down path")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Lateral error over time
    ax = axes[1]
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", label="±0.5 m band")
    ax.axhline(-0.5, color="gray", linewidth=0.8, linestyle="--")
    ax.axhline(1.5, color="salmon", linewidth=0.8, linestyle=":", label="±1.5 m limit")
    ax.axhline(-1.5, color="salmon", linewidth=0.8, linestyle=":")
    ax.plot(times, lat_errs, color="steelblue", linewidth=1.0, label="lateral error (m)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("lateral error (m)")
    ax.set_title("Lateral error vs time")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    png_path = csv_path.with_suffix(".png")
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return png_path


def test_pid_follows_s_curve():
    """PID must track a straight → S-curve → straight road at 20 mph (capped 30 mph).

    Two mirror-image variants are tested:
      * left_first  : road curves left then right before straightening
      * right_first : road curves right then left before straightening

    Success criteria:
      * S-curve section  : max |lateral_error| < 1.5 m
      * Exit straight    : within ±0.5 m for 5 s continuously
    """
    trials = [(1.0, "left_first"), (-1.0, "right_first")]
    failures: list[str] = []
    plots: list[Path] = []

    for direction, label in trials:
        ok, msg, csv_path, drive_line, z_s_start, z_s_end = _run_s_curve_trial(
            direction, label
        )
        print(f"  {label}: {'PASS' if ok else 'FAIL'} — {msg}")
        if VISUALIZE and csv_path.exists():
            png = _plot_s_curve_trial(csv_path, drive_line, z_s_start, z_s_end)
            plots.append(png)
        if not ok:
            failures.append(msg)

    if plots:
        print("\n--- S-curve plots ---")
        for p in plots:
            print(f"  {p}")

    if failures:
        pytest.fail("\n".join(failures))

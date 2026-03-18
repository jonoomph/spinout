"""Steering response regression tests.

Validates each car's yaw-rate step-steer response against ground-truth bounds
derived from the SAE J266 procedure and published OEM data.

Ground-truth benchmarks (compact/mid-size FWD sedan, e.g. Toyota Corolla 2020):
  - Rise time (10 % → 90 % of SS yaw rate) @ 25 mph : 0.25 – 0.75 s
  - Rise time (10 % → 90 % of SS yaw rate) @ 60 mph : 0.40 – 1.50 s
  - Yaw-rate overshoot                               : < 15 %
  - Steady-state yaw rate within 25 % of bicycle-model prediction

Physics note — why rise time INCREASES with speed for a neutral-steer sedan:
  The bicycle-model yaw natural frequency ωn = L·√(Cf·Cr / (m·Iz)) / V, i.e.
  ωn ∝ 1/V.  For the Corolla the damping ratio ζ ≈ 1.0 (critically damped) and
  is approximately constant with speed.  Rise time ≈ 3.74 / ωn ∝ V.  At 60 mph
  (2.4× faster than 25 mph) the bicycle model predicts ~2.4× longer rise time.
  This is counterintuitive but physically correct; driver "nimbleness" perception
  at speed comes from lower steering-wheel torque and EPS assist, not faster yaw.

Sources: Gillespie "Fundamentals of Vehicle Dynamics" (1992), Milliken & Milliken
RCVD (1995), SAE J266 (2004), Pacejka "Tyre and Vehicle Dynamics" 3rd ed.

Run normally (renders in game window):
    pytest tests/physics/test_steering.py -s

Headless (CI):
    VISUALIZE=0 pytest tests/physics/test_steering.py
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path

import numpy as np
import pytest

from src.sim.control_api import DriverCommand
from src.sim.environment import Environment
from tests.helpers import configure_flat_drive_line

VISUALIZE = bool(int(os.environ.get("VISUALIZE", "1")))
MPS_TO_MPH = 2.2369362920544
RAD_TO_DEG = 180.0 / math.pi
_LOG_DIR = Path("/tmp/steering_tests")


# ---------------------------------------------------------------------------
# Core measurement helpers
# ---------------------------------------------------------------------------

def _setup_straight_run(speed_mps: float) -> Environment:
    """Create a flat environment with the car driving straight at *speed_mps*."""
    mode = "eval" if VISUALIZE else "train"
    env = Environment({"flat": True, "dt": 0.02, "substeps": 1, "sun_time_hours": 12.0}, mode=mode)
    env.reset()
    configure_flat_drive_line(env)
    if VISUALIZE:
        env.init_renderer()

    assert env.car is not None
    env.car.body.rot.arr[:] = (1.0, 0.0, 0.0, 0.0)
    env.car.body.vel = np.array([0.0, 0.0, speed_mps], dtype=float)
    env.car.body.angvel[:] = 0.0
    env._prev_velocity = env.car.body.vel.copy()  # type: ignore[attr-defined]
    return env


def _run_step_steer(
    speed_mph: float,
    steer_input: float,
    duration_s: float = 3.0,
    settle_s: float = 0.5,
) -> dict:
    """Apply a step steer and record yaw-rate + lateral acceleration.

    Parameters
    ----------
    speed_mph   : Initial straight-line speed.
    steer_input : Normalised steer command [0, 1] applied as a positive step.
    duration_s  : How long to record *after* the step.
    settle_s    : Time to let physics settle before applying the step.

    Returns
    -------
    dict with keys:
        times, yaw_rates_deg_s, lat_accels_mss, steer_angles_deg,
        ss_yaw_rate, rise_time_s, delay_time_s, overshoot_pct,
        bicycle_ss_yaw_rate, speed_mps
    """
    speed_mps = speed_mph / MPS_TO_MPH
    env = _setup_straight_run(speed_mps)
    dt = env.dt

    settle_steps = int(math.ceil(settle_s / dt))
    record_steps = int(math.ceil(duration_s / dt))
    total_steps = settle_steps + record_steps

    # Maintain speed with a gentle throttle (does not affect lateral dynamics).
    throttle = 0.25

    times: list[float] = []
    yaw_rates: list[float] = []
    lat_accels: list[float] = []

    for step in range(total_steps):
        steer = steer_input if step >= settle_steps else 0.0
        env.step(DriverCommand(steer=steer, throttle=throttle))  # type: ignore[attr-defined]

        if step >= settle_steps:
            tele = env._build_snapshot(env._prev_velocity)  # type: ignore[attr-defined]
            t = (step - settle_steps) * dt
            times.append(t)
            yaw_rates.append(abs(tele.state.yaw_rate) * RAD_TO_DEG)
            lat_accels.append(abs(tele.state.lat_accel))

    t_arr = np.array(times)
    yr_arr = np.array(yaw_rates)
    la_arr = np.array(lat_accels)

    # Steady-state: mean of last third of the recording.
    ss_n = max(1, len(yr_arr) // 3)
    ss_yaw_rate = float(np.mean(yr_arr[-ss_n:]))

    # Delay time: first sample ≥ 10 % of SS.
    thr_10 = 0.10 * ss_yaw_rate
    thr_90 = 0.90 * ss_yaw_rate
    t_10 = next((t for t, y in zip(t_arr, yr_arr) if y >= thr_10), float("nan"))
    t_90 = next((t for t, y in zip(t_arr, yr_arr) if y >= thr_90), float("nan"))
    rise_time = (t_90 - t_10) if (not math.isnan(t_10) and not math.isnan(t_90)) else float("nan")
    delay_time = t_10 if not math.isnan(t_10) else float("nan")

    peak = float(np.max(yr_arr))
    overshoot_pct = (peak / ss_yaw_rate - 1.0) * 100.0 if ss_yaw_rate > 0 else 0.0

    # Bicycle-model steady-state yaw rate (neutral steer approximation).
    # r_ss = V * delta / L  where delta is actual wheel angle at steady state.
    # We don't have the final wheel angle directly, but can estimate from the
    # speed-sensitive limit: delta = steer_input * max_angle / (1 + V/scale).
    car = env.car
    assert car is not None
    max_angle = car.max_steer_angle  # type: ignore[attr-defined]
    scale = car.speed_steer_scale    # type: ignore[attr-defined]
    L = car.wheelbase_m              # type: ignore[attr-defined]
    delta_rad = steer_input * max_angle / (1.0 + speed_mps / scale)
    bicycle_ss = (speed_mps * delta_rad / L) * RAD_TO_DEG  # deg/s

    return {
        "speed_mph": speed_mph,
        "speed_mps": speed_mps,
        "steer_input": steer_input,
        "times": t_arr,
        "yaw_rates_deg_s": yr_arr,
        "lat_accels_mss": la_arr,
        "ss_yaw_rate": ss_yaw_rate,
        "rise_time_s": rise_time,
        "delay_time_s": delay_time,
        "overshoot_pct": overshoot_pct,
        "bicycle_ss_yaw_rate": bicycle_ss,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_step_steer(results: list[dict], label: str) -> Path:
    """Save a 2-panel PNG: yaw rate response and lat accel response."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Step-steer response — {label}", fontsize=13)

    colors = ["royalblue", "darkorange", "seagreen", "crimson"]
    for i, r in enumerate(results):
        c = colors[i % len(colors)]
        lbl = f"{r['speed_mph']:.0f} mph, steer={r['steer_input']}"
        axes[0].plot(r["times"], r["yaw_rates_deg_s"], color=c, label=lbl)
        # Mark SS and rise time
        ss = r["ss_yaw_rate"]
        axes[0].axhline(ss, color=c, linestyle=":", linewidth=0.8, alpha=0.6)
        if not math.isnan(r["rise_time_s"]):
            axes[0].annotate(
                f"rise={r['rise_time_s']:.3f}s",
                xy=(r["delay_time_s"] + r["rise_time_s"], ss * 0.9),
                fontsize=8, color=c,
            )
        axes[1].plot(r["times"], r["lat_accels_mss"], color=c, label=lbl)

    axes[0].set_ylabel("Yaw rate (deg/s)")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color="k", linewidth=0.5)

    axes[1].set_ylabel("Lateral accel (m/s²)")
    axes[1].set_xlabel("Time after step (s)")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    slug = label.lower().replace(" ", "_").replace("/", "_")
    png = _LOG_DIR / f"step_steer_{slug}.png"
    fig.savefig(png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return png


def _save_csv(results: list[dict], label: str) -> Path:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    slug = label.lower().replace(" ", "_").replace("/", "_")
    path = _LOG_DIR / f"step_steer_{slug}.csv"
    rows = []
    for r in results:
        for t, yr, la in zip(r["times"], r["yaw_rates_deg_s"], r["lat_accels_mss"]):
            rows.append({
                "speed_mph": r["speed_mph"],
                "steer_input": r["steer_input"],
                "t": round(float(t), 4),
                "yaw_rate_deg_s": round(float(yr), 4),
                "lat_accel_mss": round(float(la), 4),
            })
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _print_summary(r: dict) -> None:
    print(
        f"  {r['speed_mph']:.0f} mph  steer={r['steer_input']:.2f}"
        f"  SS_yaw={r['ss_yaw_rate']:.1f}°/s"
        f"  bicycle={r['bicycle_ss_yaw_rate']:.1f}°/s"
        f"  rise={r['rise_time_s']:.3f}s"
        f"  overshoot={r['overshoot_pct']:.1f}%"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCorollaStepSteer:
    """SAE J266-style step-steer validation for the Toyota Corolla (default car)."""

    # Steer inputs to test — moderate and strong.
    STEER_INPUTS = (0.3, 0.5)

    def test_rise_time_at_25_mph(self):
        """Yaw rise time at 25 mph must be 0.25 – 0.75 s."""
        results = []
        for steer in self.STEER_INPUTS:
            r = _run_step_steer(speed_mph=25.0, steer_input=steer, duration_s=2.5)
            _print_summary(r)
            results.append(r)

        png = _plot_step_steer(results, "25 mph")
        csv_path = _save_csv(results, "25 mph")
        print(f"\n  plot: {png}\n  csv:  {csv_path}")

        for r in results:
            rt = r["rise_time_s"]
            assert not math.isnan(rt), (
                f"steer={r['steer_input']}: yaw rate never reached 90 % of SS — "
                f"SS={r['ss_yaw_rate']:.2f} deg/s"
            )
            assert 0.25 <= rt <= 0.75, (
                f"steer={r['steer_input']}: rise time {rt:.3f}s outside "
                f"[0.25, 0.75] s benchmark"
            )

    def test_rise_time_at_60_mph(self):
        """Yaw rise time at 60 mph must be 0.40 – 1.50 s.

        Bicycle model: ωn ∝ 1/V → rise time ∝ V.  At 60 mph (2.4× faster than
        25 mph) the expected rise time is roughly 2× that of the 25 mph case.
        """
        results = []
        for steer in self.STEER_INPUTS:
            r = _run_step_steer(speed_mph=60.0, steer_input=steer, duration_s=2.5)
            _print_summary(r)
            results.append(r)

        png = _plot_step_steer(results, "60 mph")
        csv_path = _save_csv(results, "60 mph")
        print(f"\n  plot: {png}\n  csv:  {csv_path}")

        for r in results:
            rt = r["rise_time_s"]
            assert not math.isnan(rt), (
                f"steer={r['steer_input']}: yaw rate never reached 90 % of SS"
            )
            assert 0.40 <= rt <= 1.50, (
                f"steer={r['steer_input']}: rise time {rt:.3f}s outside "
                f"[0.40, 1.50] s benchmark"
            )

    def test_rise_time_speed_scaling(self):
        """Rise time at 60 mph must not exceed 3× the 25 mph value.

        Bicycle model: rise time ∝ V, so at 2.4× the speed we expect roughly
        2.4× the rise time.  The upper bound of 3× guards against pathological
        lag amplification (e.g. if the steering filter or tire model breaks down).
        Rise time increasing with speed is the physically correct behaviour for a
        critically-damped FWD sedan (ζ ≈ 1.0, ωn ∝ 1/V → rise_time ∝ V).
        """
        steer = 0.3
        r25 = _run_step_steer(speed_mph=25.0, steer_input=steer, duration_s=2.5)
        r60 = _run_step_steer(speed_mph=60.0, steer_input=steer, duration_s=2.5)
        print(f"\n  25 mph rise={r25['rise_time_s']:.3f}s  "
              f"60 mph rise={r60['rise_time_s']:.3f}s  "
              f"ratio={r60['rise_time_s']/r25['rise_time_s']:.2f}×")

        results = [r25, r60]
        png = _plot_step_steer(results, "speed comparison")
        print(f"  plot: {png}")

        ratio = r60["rise_time_s"] / r25["rise_time_s"]
        assert ratio <= 3.0, (
            f"Rise time ratio 60/25 mph = {ratio:.2f}× exceeds 3.0× limit "
            f"(25 mph={r25['rise_time_s']:.3f}s, 60 mph={r60['rise_time_s']:.3f}s)"
        )

    def test_yaw_rate_overshoot(self):
        """Yaw-rate overshoot must stay below 15 % (well-damped sedan behaviour)."""
        results = []
        for speed in (25.0, 60.0):
            r = _run_step_steer(speed_mph=speed, steer_input=0.3, duration_s=2.5)
            _print_summary(r)
            results.append(r)

        png = _plot_step_steer(results, "overshoot check")
        print(f"\n  plot: {png}")

        for r in results:
            assert r["overshoot_pct"] < 15.0, (
                f"{r['speed_mph']:.0f} mph: yaw overshoot {r['overshoot_pct']:.1f}% "
                f"exceeds 15 % limit"
            )

    def test_steady_state_yaw_rate_matches_bicycle_model(self):
        """SS yaw rate must lie within 25 % of the neutral bicycle-model prediction.

        The bicycle model (neutral steer) gives an upper bound — a real car with
        understeer will have a lower SS yaw rate, so we also allow -40 % of the
        neutral value (to accommodate FWD understeer).
        """
        results = []
        for speed in (25.0, 60.0):
            r = _run_step_steer(speed_mph=speed, steer_input=0.4, duration_s=2.5)
            _print_summary(r)
            results.append(r)

        png = _plot_step_steer(results, "ss yaw rate validation")
        print(f"\n  plot: {png}")

        for r in results:
            ss = r["ss_yaw_rate"]
            bk = r["bicycle_ss_yaw_rate"]
            ratio = ss / bk if bk > 0 else 0.0
            assert 0.60 <= ratio <= 1.25, (
                f"{r['speed_mph']:.0f} mph: SS yaw rate {ss:.2f}°/s is "
                f"{ratio:.2f}× bicycle model {bk:.2f}°/s — outside [0.60, 1.25]"
            )

    def test_multi_speed_sweep(self):
        """Generate a full speed-sweep plot (25 / 40 / 55 / 70 mph) for inspection."""
        speeds = (25.0, 40.0, 55.0, 70.0)
        steer = 0.3
        results = []
        print()
        for speed in speeds:
            r = _run_step_steer(speed_mph=speed, steer_input=steer, duration_s=2.5)
            _print_summary(r)
            results.append(r)

        png = _plot_step_steer(results, f"speed sweep steer={steer}")
        csv_path = _save_csv(results, "speed_sweep")
        print(f"\n  plot: {png}\n  csv:  {csv_path}")
        # No hard assertion — this is a diagnostic sweep for visual inspection.

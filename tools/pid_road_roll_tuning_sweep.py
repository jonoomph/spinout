#!/usr/bin/env python3
"""Offline tuning sweep for PID road-roll compensation.

Evaluates a straight-road, smooth left-roll then right-roll scenario across
20 / 40 / 60 mph and two bank amplitudes:
  - 1.00x current test bank
  - 1.25x steeper bank

The sweep focuses on roll-aware PID tuning:
  - `k_roll`     : feedback on current roll_lataccel tracking error
  - `k_roll_ff`  : feedforward on previewed future road roll

It also allows small retunes of the existing lateral gains to see whether the
best road-roll result still prefers the current baseline.
"""

from __future__ import annotations

import itertools
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.controllers.pid import PIDGains, PIDSteeringController
from src.sim.control_api import DriverCommand
from src.sim.environment import Environment
from src.sim.roads.build import apply_plan

MPS_TO_MPH = 2.2369362920544

KP_VALUES = [0.45, 0.55, 0.65]
KD_VALUES = [0.25, 0.35, 0.45]
K_FF_VALUES = [0.08]
K_ROLL_VALUES = [-0.06, -0.03, 0.0, 0.03, 0.06]
K_ROLL_FF_VALUES = [-0.12, -0.06, 0.0, 0.06, 0.12]
KI = 0.005
SPEEDS_MPH = [20.0, 40.0, 60.0]
BANK_SCALE_VALUES = [1.0, 1.25]
MAX_WORKERS = 8

CURRENT = dict(kp=0.55, kd=0.35, k_ff=0.08, k_roll=0.0, k_roll_ff=0.0)


def _smoothstep(s: float) -> float:
    s = max(0.0, min(1.0, s))
    return s * s * (3.0 - 2.0 * s)


def _make_roll_profile(z_roll_start: float, roll_length_m: float, max_bank_rad: float):
    def cross_pitch_profile(s_value: float) -> float:
        if s_value <= z_roll_start or s_value >= z_roll_start + roll_length_m:
            return 0.0
        t = (s_value - z_roll_start) / max(roll_length_m, 1e-6)
        if t <= 0.25:
            return max_bank_rad * _smoothstep(t / 0.25)
        if t <= 0.5:
            return max_bank_rad * (1.0 - _smoothstep((t - 0.25) / 0.25))
        if t <= 0.75:
            return -max_bank_rad * _smoothstep((t - 0.5) / 0.25)
        return -max_bank_rad * (1.0 - _smoothstep((t - 0.75) / 0.25))

    return cross_pitch_profile


def _build_banked_straight_plan(env: Environment, target_speed_mph: float, bank_scale: float) -> tuple[dict, float, float, float]:
    assert env.terrain is not None
    terrain = env.terrain
    center_x = terrain.width * 0.5
    z_roll_start = terrain.height * 0.25 + 80.0
    roll_length_m = 240.0
    z_roll_end = z_roll_start + roll_length_m
    max_bank_rad = math.radians(8.0 * bank_scale)
    drive_line = [(center_x, float(z)) for z in np.linspace(0.0, terrain.height, 300)]
    speed_limits = (
        {"start_s": 0.0, "end_s": float("inf"), "speed_mph": target_speed_mph * 1.25},
    )
    plan = {
        "lane_width": 3.6,
        "lanes": 1,
        "shoulder": 1.5,
        "road_height": 1.0,
        "cross_pitch": 0.0,
        "cross_pitch_profile": _make_roll_profile(z_roll_start, roll_length_m, max_bank_rad),
        "cross_pitch_mode": "bank",
        "ditch_width": 0.0,
        "ditch_depth": 0.0,
        "road_friction": 1.0,
        "drive_line": drive_line,
        "speed_limits": speed_limits,
    }
    return plan, center_x, z_roll_start, z_roll_end


class TrialResult(NamedTuple):
    ok: bool
    max_error: float
    rms_error: float
    exit_stable: bool


def _run_trial(kp: float, kd: float, k_ff: float, k_roll: float, k_roll_ff: float, speed_mph: float, bank_scale: float) -> TrialResult:
    env = Environment(
        {"flat": True, "dt": 1.0 / 300.0, "substeps": 1, "sun_time_hours": 17.0},
        mode="train",
    )
    env.reset()
    assert env.terrain is not None and env.car is not None

    plan, center_x, z_roll_start, z_roll_end = _build_banked_straight_plan(env, speed_mph, bank_scale)
    env.plan = plan
    env.rp = plan["drive_line"]
    apply_plan(env.terrain, env.rp, plan)
    env._planner.set_plan(env.rp, plan["speed_limits"], plan)  # type: ignore[attr-defined]

    initial_mps = speed_mph / MPS_TO_MPH
    z0 = env.terrain.height * 0.25
    y0 = env.terrain.get_height(center_x, z0) + env.car.cg_height_m
    env.car.body.pos = np.array([center_x, y0, z0], dtype=float)
    env.car.body.vel = np.array([0.0, 0.0, initial_mps], dtype=float)
    env.car.body.angvel[:] = 0.0
    env.car.body.rot.arr[:] = (1.0, 0.0, 0.0, 0.0)
    env._prev_velocity = env.car.body.vel.copy()  # type: ignore[attr-defined]
    env._refresh_initial_telemetry()  # type: ignore[attr-defined]

    controller = PIDSteeringController(
        gains=PIDGains(
            kp=kp,
            ki=KI,
            kd=kd,
            k_ff=k_ff,
            k_roll=k_roll,
            k_roll_ff=k_roll_ff,
        ),
        control_rate_hz=100.0,
    )
    env.attach_controller(controller)
    controller.enable()

    max_speed_mph = speed_mph + 10.0
    throttle_cmd = 0.0
    throttle_alpha = 0.25
    max_time = 80.0
    steps = int(math.ceil(max_time / env.dt))
    exit_stable_time = 0.0
    errors: list[float] = []
    past_roll = False

    for _ in range(steps):
        speed_now = float(np.linalg.norm(env.car.body.vel)) * MPS_TO_MPH
        car_z = float(env.car.body.pos[2])
        speed_error = speed_mph - speed_now
        brake_cmd = 0.0
        if speed_now > max_speed_mph + 2.0:
            desired_throttle = 0.0
            brake_cmd = 0.25
        elif speed_now > max_speed_mph + 0.5:
            desired_throttle = 0.0
            brake_cmd = 0.10
        elif speed_now > speed_mph + 1.0:
            desired_throttle = 0.0
        elif speed_error > 5.0:
            desired_throttle = 0.8
        elif speed_error > 2.0:
            desired_throttle = 0.6
        elif speed_error > 0.5:
            desired_throttle = 0.45
        elif speed_error > -0.5:
            desired_throttle = 0.10
        else:
            desired_throttle = 0.0
        throttle_cmd += throttle_alpha * (desired_throttle - throttle_cmd)

        _, _, terminated, truncated, _ = env.step(DriverCommand(throttle=throttle_cmd, brake=brake_cmd))
        if terminated or truncated:
            env.attach_controller(None)
            return TrialResult(False, 999.0, 999.0, False)

        tele = env._build_snapshot(env._prev_velocity)  # type: ignore[attr-defined]
        lat_error = float(tele.target.lateral_error)
        if z_roll_start <= car_z <= z_roll_end:
            errors.append(lat_error)
            if abs(lat_error) > 5.0:
                env.attach_controller(None)
                return TrialResult(False, abs(lat_error), 999.0, False)
        if car_z > z_roll_end:
            past_roll = True
            if abs(lat_error) <= 0.5:
                exit_stable_time += env.dt
                if exit_stable_time >= 5.0:
                    break
            else:
                exit_stable_time = 0.0

    env.attach_controller(None)
    if not past_roll or not errors:
        return TrialResult(False, 999.0, 999.0, False)
    abs_err = np.abs(np.array(errors, dtype=float))
    return TrialResult(exit_stable_time >= 5.0, float(np.max(abs_err)), float(np.sqrt(np.mean(abs_err ** 2))), exit_stable_time >= 5.0)


def _score_candidate(kp: float, kd: float, k_ff: float, k_roll: float, k_roll_ff: float):
    results = {}
    penalty = 0.0
    for bank_scale, speed_mph in itertools.product(BANK_SCALE_VALUES, SPEEDS_MPH):
        trial = _run_trial(kp, kd, k_ff, k_roll, k_roll_ff, speed_mph, bank_scale)
        results[(bank_scale, speed_mph)] = trial
        penalty += trial.max_error
        penalty += 0.5 * trial.rms_error
        if not trial.exit_stable:
            penalty += 25.0
    return {
        "kp": kp,
        "kd": kd,
        "k_ff": k_ff,
        "k_roll": k_roll,
        "k_roll_ff": k_roll_ff,
        "score": penalty,
        "results": results,
    }


def _format_candidate(row: dict) -> str:
    summary = []
    for bank_scale in BANK_SCALE_VALUES:
        for speed_mph in SPEEDS_MPH:
            trial = row["results"][(bank_scale, speed_mph)]
            summary.append(
                f"{int(speed_mph):02d}mph@{bank_scale:.2f}x"
                f":max={trial.max_error:.2f},rms={trial.rms_error:.2f},"
                f"{'settled' if trial.exit_stable else 'unstable'}"
            )
    return (
        f"score={row['score']:.2f}  kp={row['kp']:.2f} kd={row['kd']:.2f} "
        f"k_ff={row['k_ff']:.2f} k_roll={row['k_roll']:.3f} k_roll_ff={row['k_roll_ff']:.3f}\n"
        f"  {' | '.join(summary)}"
    )


def main() -> int:
    grid = list(
        itertools.product(
            KP_VALUES,
            KD_VALUES,
            K_FF_VALUES,
            K_ROLL_VALUES,
            K_ROLL_FF_VALUES,
        )
    )
    print(f"Running {len(grid)} road-roll candidates across {len(SPEEDS_MPH) * len(BANK_SCALE_VALUES)} scenarios each")
    rows = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(_score_candidate, kp, kd, k_ff, k_roll, k_roll_ff): (kp, kd, k_ff, k_roll, k_roll_ff)
            for kp, kd, k_ff, k_roll, k_roll_ff in grid
        }
        for future in as_completed(futures):
            rows.append(future.result())

    rows.sort(key=lambda row: row["score"])
    print("\nTop candidates:")
    for row in rows[:10]:
        print(_format_candidate(row))

    current = _score_candidate(**CURRENT)
    print("\nCurrent baseline:")
    print(_format_candidate(current))
    if rows:
        print("\nRecommended:")
        print(_format_candidate(rows[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

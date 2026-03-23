#!/usr/bin/env python3
"""Offline tuning sweep for the PID S-curve controller.

This script is a development tool, not part of the production test suite.
Use it to compare controller parameter candidates across 20 / 40 / 60 mph
S-curve trials before promoting a tuning change into ``src/controllers/pid.py``.

Includes a speed-scaling exponent axis: gains are multiplied by
    (V_REF / max(v_ego, V_REF)) ** v_scale_exp
so v_scale_exp=0 means no scaling, 0.5=sqrt, 1.0=linear.

Also sweeps two high-speed-specific stabilizers:
  - corr_limit_hi: speed-dependent cap on the P+I+D correction only
  - preview_hi_secs: shorter curvature preview at 60 mph to reduce
    over-anticipating the opposite half of the S-curve

Usage:
    python tools/pid_scurve_tuning_sweep.py
"""
from __future__ import annotations

import itertools
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ── Parameter grid (coarse) ──────────────────────────────────────────────────

KP_VALUES      = [0.30, 0.40, 0.55]
KD_VALUES      = [0.25, 0.35, 0.50]
K_FF_VALUES    = [0.08, 0.12, 0.20]
V_SCALE_EXPS   = [0.0, 0.5, 1.0]   # 0=none  0.5=sqrt  1.0=linear
CORR_LIMIT_HI_VALUES = [1.0, 0.35]  # 1.0 ~= no extra limit at 60 mph
PREVIEW_HI_VALUES    = [1.3, 0.9]   # 1.3=current behavior, 0.9=shorter lookahead
KI             = 0.005              # fixed
V_REF_MPS      = 20.0 * 0.44704    # reference speed (20 mph) for scaling
SPEEDS_MPH     = [20.0, 40.0, 60.0]
MAX_WORKERS    = 8

# Current defaults (for highlighting in output)
CURRENT = dict(
    kp=0.55,
    kd=0.50,
    k_ff=0.08,
    v_scale_exp=0.0,
    corr_limit_hi=1.0,
    preview_hi_secs=1.3,
)


# ── Geometry helpers (mirrors test_controller.py) ────────────────────────────

def _smoothstep(s: float) -> float:
    s = max(0.0, min(1.0, s))
    return s * s * (3.0 - 2.0 * s)


def _build_drive_line(terrain_width: float, terrain_height: float,
                      direction: float = 1.0, amplitude_m: float = 5.0,
                      s_length_m: float = 200.0):
    cx = terrain_width * 0.5
    z_s_start = terrain_height * 0.25 + 80.0
    z_s_end   = z_s_start + s_length_m
    z_exit    = terrain_height - 50.0
    pts: list[tuple[float, float]] = []

    for i in range(100):
        pts.append((cx, z_s_start * i / 99.0))

    for i in range(1, 201):
        t = i / 200.0
        z = z_s_start + t * s_length_m
        if   t <= 0.25: dx = direction * amplitude_m *  _smoothstep(t / 0.25)
        elif t <= 0.50: dx = direction * amplitude_m *  (1.0 - _smoothstep((t - 0.25) / 0.25))
        elif t <= 0.75: dx = -direction * amplitude_m * _smoothstep((t - 0.50) / 0.25)
        else:           dx = -direction * amplitude_m * (1.0 - _smoothstep((t - 0.75) / 0.25))
        pts.append((cx + dx, z))

    for i in range(1, 201):
        t = i / 200.0
        pts.append((cx, z_s_end + t * (z_exit - z_s_end)))

    return pts, cx, z_s_start, z_s_end


# ── Single trial ─────────────────────────────────────────────────────────────

class TrialResult(NamedTuple):
    max_error:    float
    steer_sigma:  float
    sign_changes: int
    settled:      bool


def _run_trial(kp: float, kd: float, k_ff: float, ki: float,
               v_scale_exp: float, corr_limit_hi: float,
               preview_hi_secs: float, speed_mph: float) -> TrialResult:
    """One S-curve trial with optional speed-scaled gains."""
    try:
        from src.controllers.pid import (
            PIDGains,
            PIDSteeringController,
            _PREVIEW_SECS_LO,
            _PREVIEW_V_LO,
            _PREVIEW_V_HI,
            _HEADING_PREVIEW_SECS,
        )
        from src.sim.control_api import DriverCommand
        from src.sim.environment import Environment

        MPS = 2.2369362920544

        env = Environment(
            {"flat": True, "dt": 0.02, "substeps": 1, "sun_time_hours": 12.0},
            mode="train",
        )
        env.reset()
        assert env.terrain is not None

        dl, cx, z_s_start, z_s_end = _build_drive_line(
            env.terrain.width, env.terrain.height
        )
        speed_limits = ({"start_s": 0.0, "end_s": float("inf"),
                         "speed_mph": speed_mph * 1.25},)
        env._planner.set_plan(dl, speed_limits)  # type: ignore[attr-defined]
        env.plan = {
            "drive_line": dl, "speed_limits": speed_limits,
            "lane_width": 3.6, "lanes": 1, "shoulder": 1.5,
            "road_height": 0.02, "cross_pitch": 0.0,
            "ditch_width": 0.0, "ditch_depth": 0.0, "road_friction": 1.0,
        }
        env.rp = dl

        assert env.car is not None
        v0 = speed_mph / MPS
        z0 = env.terrain.height * 0.25
        y0 = env.terrain.get_height(cx, z0) + env.car.cg_height_m
        env.car.body.pos = np.array([cx, y0, z0], dtype=float)
        env.car.body.vel = np.array([0.0, 0.0, v0], dtype=float)
        env.car.body.angvel[:] = 0.0
        env.car.body.rot.arr[:] = (1.0, 0.0, 0.0, 0.0)
        env._prev_velocity = env.car.body.vel.copy()  # type: ignore[attr-defined]
        env._refresh_initial_telemetry()  # type: ignore[attr-defined]

        # Build a subclass that applies speed-scaled gains at each step
        class _ScaledPID(PIDSteeringController):
            def step(self, telemetry, manual):  # type: ignore[override]
                if not self.enabled:
                    return manual

                dt = max(self.dt or 0.1, 1e-6)
                state = telemetry.state
                v_ego = max(state.v_ego, V_REF_MPS)
                scale = (V_REF_MPS / v_ego) ** v_scale_exp if v_scale_exp > 0.0 else 1.0

                fut = telemetry.future.lat_accel or ()
                hi_frac = max(0.0, min(1.0, (min(state.v_ego, _PREVIEW_V_HI) - _PREVIEW_V_LO) /
                                       (_PREVIEW_V_HI - _PREVIEW_V_LO)))
                t_preview = _PREVIEW_SECS_LO + hi_frac * (preview_hi_secs - _PREVIEW_SECS_LO)
                n = min(len(fut), max(1, round(t_preview * (self.preview_rate_hz or 10.0))))
                base = float(np.mean(fut[:n])) if n > 0 and len(fut) > 0 else telemetry.target.lat_accel
                ff = self.gains.k_ff * base

                e = (
                    telemetry.target.lateral_error
                    + telemetry.target.heading_error * state.v_ego * _HEADING_PREVIEW_SECS
                )
                de_dt = 0.0 if self._prev_eff_err is None else (e - self._prev_eff_err) / dt
                self._prev_eff_err = e

                self._integral = max(
                    -self.gains.integral_limit,
                    min(self.gains.integral_limit, self._integral + e * dt),
                )

                kp_eff = kp * scale
                kd_eff = kd * scale
                p_term = -kp_eff * e
                i_term = -self.gains.ki * self._integral
                d_term = -kd_eff * de_dt
                correction = p_term + i_term + d_term

                corr_limit = 1.0 + hi_frac * (corr_limit_hi - 1.0)
                correction = max(-corr_limit, min(corr_limit, correction))
                steer = max(-self.steer_limit, min(self.steer_limit, ff + correction))

                self._last_base_target_lat = base
                self._last_target_lat = base
                self._last_pid_error = e
                self._last_p_term = p_term
                self._last_i_term = i_term
                self._last_d_term = d_term
                self._last_pid_out = correction
                self._last_ff = ff
                self._last_lateral_term = correction

                return DriverCommand(steer=steer, throttle=manual.throttle, brake=manual.brake)

        gains = PIDGains(kp=kp, ki=ki, kd=kd, k_ff=k_ff)
        ctrl  = _ScaledPID(gains=gains)
        env.attach_controller(ctrl)
        ctrl.enable()

        target_mph = speed_mph
        max_mph    = speed_mph + 10.0
        throttle   = 0.0

        max_time = 120.0
        steps    = int(math.ceil(max_time / env.dt))

        s_max_err       = 0.0
        exit_stable     = 0.0
        exit_stable_req = 5.0
        past_curve      = False
        steers: list[float] = []
        hard_fail_m     = 4.8

        for _ in range(steps):
            car_z  = float(env.car.body.pos[2])
            spd    = float(np.linalg.norm(env.car.body.vel)) * MPS
            err    = target_mph - spd
            brake  = 0.0

            if   spd > max_mph + 2.0: desired_th = 0.0; brake = 0.25
            elif spd > target_mph + 1.0: desired_th = 0.0
            elif err >  5.0: desired_th = 0.80
            elif err >  2.0: desired_th = 0.60
            elif err >  0.5: desired_th = 0.45
            elif err > -0.5: desired_th = 0.10
            else: desired_th = 0.0
            throttle += 0.25 * (desired_th - throttle)

            _, _, terminated, _, info = env.step(
                DriverCommand(throttle=throttle, brake=brake)
            )
            if terminated:
                env.attach_controller(None)
                return TrialResult(999.0, 0.0, 0, False)

            steers.append(info["driver_command"]["steer"])

            tele    = env._build_snapshot(env._prev_velocity)  # type: ignore[attr-defined]
            lat_err = abs(tele.target.lateral_error)

            if z_s_start <= car_z <= z_s_end:
                s_max_err = max(s_max_err, lat_err)
                if s_max_err > hard_fail_m:
                    env.attach_controller(None)
                    return TrialResult(s_max_err, float(np.std(steers)), 0, False)

            if car_z > z_s_end:
                past_curve = True
                if lat_err <= 0.5:
                    exit_stable += env.dt
                    if exit_stable >= exit_stable_req:
                        env.attach_controller(None)
                        s_arr = np.array(steers)
                        sc = int(np.sum(np.diff(np.sign(s_arr)) != 0))
                        return TrialResult(s_max_err, float(np.std(s_arr)), sc, True)
                else:
                    exit_stable = 0.0

        env.attach_controller(None)
        s_arr = np.array(steers) if steers else np.zeros(1)
        sc = int(np.sum(np.diff(np.sign(s_arr)) != 0))
        return TrialResult(s_max_err, float(np.std(s_arr)), sc, past_curve)

    except Exception:
        return TrialResult(999.0, 0.0, 0, False)


# ── Combo runner ──────────────────────────────────────────────────────────────

def _run_combo(args: tuple) -> dict:
    kp, kd, k_ff, v_scale_exp, corr_limit_hi, preview_hi_secs = args
    per_speed = {}
    for mph in SPEEDS_MPH:
        per_speed[mph] = _run_trial(
            kp, kd, k_ff, KI, v_scale_exp, corr_limit_hi, preview_hi_secs, mph
        )
    return {
        "kp": kp,
        "kd": kd,
        "k_ff": k_ff,
        "vse": v_scale_exp,
        "corr_limit_hi": corr_limit_hi,
        "preview_hi_secs": preview_hi_secs,
        "results": per_speed,
    }


# ── Scoring (lower = better) ──────────────────────────────────────────────────

def _score(combo: dict) -> float:
    r = combo["results"]
    e20, e40, e60 = r[20.0].max_error, r[40.0].max_error, r[60.0].max_error
    if max(e20, e40, e60) >= 4.8:
        return 999.0
    err    = 2.5 * e20 + 1.5 * e40 + 1.0 * e60
    smooth = 0.05 * (r[20.0].steer_sigma + r[40.0].steer_sigma + r[60.0].steer_sigma)
    settle = sum(0 if r[s].settled else 0.5 for s in SPEEDS_MPH)
    return err + smooth + settle


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    combos = list(itertools.product(
        KP_VALUES,
        KD_VALUES,
        K_FF_VALUES,
        V_SCALE_EXPS,
        CORR_LIMIT_HI_VALUES,
        PREVIEW_HI_VALUES,
    ))
    n = len(combos)
    print(f"Sweeping {n} combos × {len(SPEEDS_MPH)} speeds = {n * len(SPEEDS_MPH)} trials "
          f"({MAX_WORKERS} workers)")
    print(
        f"kp={KP_VALUES}  kd={KD_VALUES}  k_ff={K_FF_VALUES}  "
        f"v_scale_exp={V_SCALE_EXPS}  corr_limit_hi={CORR_LIMIT_HI_VALUES}  "
        f"preview_hi_secs={PREVIEW_HI_VALUES}\n"
    )

    results: list[dict] = []
    done = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_run_combo, c): c for c in combos}
        for fut in as_completed(futures):
            done += 1
            results.append(fut.result())
            print(f"\r  {done}/{n} done …", end="", flush=True)
    print()

    results.sort(key=_score)

    SEP = "-" * 92
    header = (f"{'#':>3}  {'kp':>5} {'kd':>5} {'k_ff':>5} {'vse':>4} "
              f"{'cl60':>5} {'pv60':>5}  "
              f"{'e_20':>6} {'e_40':>6} {'e_60':>6}  "
              f"{'sc20':>5} {'sc40':>5} {'sc60':>5}  "
              f"{'score':>7}")
    print(SEP)
    print(header)
    print(SEP)

    for rank, combo in enumerate(results[:30], 1):
        r    = combo["results"]
        e20  = r[20.0].max_error
        e40  = r[40.0].max_error
        e60  = r[60.0].max_error
        sc20 = r[20.0].sign_changes
        sc40 = r[40.0].sign_changes
        sc60 = r[60.0].sign_changes
        sc   = _score(combo)

        ok20 = "✓" if r[20.0].settled else "✗"
        ok40 = "✓" if r[40.0].settled else "✗"
        ok60 = "✓" if r[60.0].settled else "✗"

        is_current = (
            combo["kp"] == CURRENT["kp"]
            and combo["kd"] == CURRENT["kd"]
            and combo["k_ff"] == CURRENT["k_ff"]
            and combo["vse"] == CURRENT["v_scale_exp"]
            and combo["corr_limit_hi"] == CURRENT["corr_limit_hi"]
            and combo["preview_hi_secs"] == CURRENT["preview_hi_secs"]
        )
        marker = "  ← current" if is_current else ""
        print(
            f"{rank:>3}  {combo['kp']:>5.2f} {combo['kd']:>5.2f} {combo['k_ff']:>5.2f} "
            f"{combo['vse']:>4.1f} {combo['corr_limit_hi']:>5.2f} {combo['preview_hi_secs']:>5.2f}  "
            f"{e20:>5.2f}{ok20} {e40:>5.2f}{ok40} {e60:>5.2f}{ok60}  "
            f"{sc20:>5} {sc40:>5} {sc60:>5}  {sc:>7.3f}{marker}"
        )

    print(SEP)

    best = results[0]
    br   = best["results"]
    print(
        f"\nRECOMMENDED  kp={best['kp']}  ki={KI}  kd={best['kd']}  "
        f"k_ff={best['k_ff']}  v_scale_exp={best['vse']}  "
        f"corr_limit_hi={best['corr_limit_hi']}  preview_hi_secs={best['preview_hi_secs']}"
    )
    for mph in SPEEDS_MPH:
        t = br[mph]
        tag = "✓ settled" if t.settled else "✗ unsettled"
        print(f"  {mph:.0f} mph:  error={t.max_error:.2f} m  "
              f"σ={t.steer_sigma:.3f}  sign_changes={t.sign_changes}  {tag}")


if __name__ == "__main__":
    main()

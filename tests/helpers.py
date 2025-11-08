"""Shared utilities for the Spinout test suite."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from src.sim.environment import Environment


def configure_flat_drive_line(
    env: Environment,
    *,
    points: int = 200,
    lane_width: float = 3.6,
    lanes: int = 1,
    shoulder: float = 1.5,
    speed_mph: float = 35.0,
) -> float:
    """Install a straight driveline down the terrain centre for deterministic tests.

    Returns the road centre ``x`` coordinate so callers can spawn vehicles relative to it.
    """

    assert env.terrain is not None
    center_x = env.terrain.width * 0.5
    z_vals = np.linspace(0.0, env.terrain.height, points)
    drive_line = [(center_x, float(z)) for z in z_vals]
    speed_limits = (
        {
            "start_s": 0.0,
            "end_s": float("inf"),
            "speed_mph": speed_mph,
        },
    )
    plan = {
        "lane_width": lane_width,
        "lanes": lanes,
        "shoulder": shoulder,
        "road_height": 0.02,
        "cross_pitch": 0.0,
        "ditch_width": 0.0,
        "ditch_depth": 0.0,
        "road_friction": 1.0,
        "drive_line": drive_line,
        "speed_limits": speed_limits,
    }
    env.plan = plan
    env.rp = drive_line
    env._planner.set_plan(drive_line, speed_limits)  # type: ignore[attr-defined]
    if getattr(env, "render_ctx", None):
        env._build_road_layers()  # type: ignore[attr-defined]
    return center_x


def ensure_drive_line_layer(
    env: Environment,
    *,
    width: float = 0.4,
    height_offset: float = 0.05,
) -> None:
    """Ensure ``env.road_layers['driveline']`` contains a visible green strip."""

    render_ctx = getattr(env, "render_ctx", None)
    if render_ctx is None:
        return
    plan = getattr(env, "plan", None) or {}
    drive_line: Sequence[Sequence[float]] | None = plan.get("drive_line")
    terrain = env.terrain
    if terrain is None or not drive_line or len(drive_line) < 2:
        return
    verts: list[float] = []
    half = 0.5 * width
    color = [0.0, 1.0, 0.0, 1.0]
    for (x0, z0), (x1, z1) in zip(drive_line[:-1], drive_line[1:]):
        dx = x1 - x0
        dz = z1 - z0
        length = math.hypot(dx, dz)
        if length < 1e-6:
            continue
        nx = -dz / length
        nz = dx / length
        y0 = terrain.get_height(x0, z0) + height_offset
        y1 = terrain.get_height(x1, z1) + height_offset
        left0 = [x0 + nx * half, y0, z0 + nz * half]
        right0 = [x0 - nx * half, y0, z0 - nz * half]
        left1 = [x1 + nx * half, y1, z1 + nz * half]
        right1 = [x1 - nx * half, y1, z1 - nz * half]
        verts.extend(left0 + color)
        verts.extend(right0 + color)
        verts.extend(left1 + color)
        verts.extend(left1 + color)
        verts.extend(right0 + color)
        verts.extend(right1 + color)
    if not verts:
        return
    data = np.array(verts, dtype="f4")
    vbo = render_ctx.ctx.buffer(data.tobytes())
    vao = render_ctx.ctx.vertex_array(render_ctx.prog, vbo, "in_vert", "in_color")
    env.road_layers["driveline"] = (vao, vbo, 0.0)


__all__ = ["configure_flat_drive_line", "ensure_drive_line_layer"]

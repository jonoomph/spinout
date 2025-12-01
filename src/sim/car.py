# car.py
import math
import numpy as np
from .colors import (
    CAR_BODY_COLOR,
    WHEEL_DEFAULT_COLOR,
    WIND_COLOR,
    AMBIENT_WIND_COLOR,
)
from .constants import AIR_DENSITY


def _wheel_points(offset_pos, axle_dir, v1, v2, angle, radius, num):
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    t = np.linspace(0, 2 * math.pi, num, endpoint=False)
    local = np.outer(np.cos(t), v1) + np.outer(np.sin(t), v2)
    rot = (
        local * cos_a
        - np.cross(axle_dir, local) * sin_a
        + np.outer(np.dot(local, axle_dir), axle_dir) * (1 - cos_a)
    )
    pts = offset_pos + rot * radius
    return list(pts)


def collect_car_vertices(car, car_up, car_dir, dt, wheel_spin_accum):
    main_vertices = []  # Car body, tires, wind lines
    shock_vertices = []  # Shocks for thicker rendering

    # Car body using dimensions from car, offset by body_offset
    half_length = car.dimensions["length"] / 2
    half_width = car.dimensions["width"] / 2
    half_height = car.dimensions["height"] / 2
    corners_rel = [
        np.array([half_width, half_height + car.body_offset, half_length]),
        np.array([half_width, half_height + car.body_offset, -half_length]),
        np.array([half_width, -half_height + car.body_offset, half_length]),
        np.array([half_width, -half_height + car.body_offset, -half_length]),
        np.array([-half_width, half_height + car.body_offset, half_length]),
        np.array([-half_width, half_height + car.body_offset, -half_length]),
        np.array([-half_width, -half_height + car.body_offset, half_length]),
        np.array([-half_width, -half_height + car.body_offset, -half_length]),
    ]
    world_corners = [car.body.pos + car.body.rot.rotate(c) for c in corners_rel]
    edges = [
        (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6),
        (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)
    ]
    car_color = list(CAR_BODY_COLOR)
    for a, b in edges:
        pa = world_corners[a]
        pb = world_corners[b]
        main_vertices.extend(list(pa) + car_color)
        main_vertices.extend(list(pb) + car_color)

    # Wheels
    for idx, wheel in enumerate(car.wheels):
        rel = wheel.rel_pos + np.array([0.0, wheel.compression, 0.0])
        hub_pos = car.body.pos + car.body.rot.rotate(rel)
        # Suspension compression is computed during physics update and stored on
        # each wheel. Use that value for consistent visualization.
        compression = wheel.compression
        compression_ratio = max(0.0, min(1.0, compression / wheel.suspension_travel))
        suspension_length = max(0.05, wheel.suspension_travel - compression)
        # Color shocks from green (fully extended) to red (fully compressed)
        # to visualize suspension travel clearly.
        susp_color = [compression_ratio, 1 - compression_ratio, 0.0, 1.0]
        if wheel.is_grounded:
            slip = max(0.0, min(1.0, wheel.slip_ratio))
            base = WHEEL_DEFAULT_COLOR
            tire_color = [
                base[0] + (1.0 - base[0]) * slip,
                base[1] * (1.0 - slip),
                base[2] * (1.0 - slip),
                1.0,
            ]
        else:
            tire_color = list(WHEEL_DEFAULT_COLOR)

        local_steer = wheel.steer_angle
        local_axle = np.array([math.cos(local_steer), 0, -math.sin(local_steer)])
        axle_dir = car.body.rot.rotate(local_axle)
        axle_dir_norm = np.linalg.norm(axle_dir)
        axle_dir = axle_dir / axle_dir_norm if axle_dir_norm > 0 else axle_dir
        arbitrary = np.array([0, 1, 0]) if abs(axle_dir[1]) < 0.9 else np.array([1, 0, 0])
        v1 = np.cross(axle_dir, arbitrary)
        v1_norm = np.linalg.norm(v1)
        v1 = v1 / v1_norm if v1_norm > 0 else v1
        v2 = np.cross(axle_dir, v1)
        v2_norm = np.linalg.norm(v2)
        v2 = v2 / v2_norm if v2_norm > 0 else v2
        tire_width = car.tire_width
        offsets = [-tire_width / 2, tire_width / 2]
        num_points = 16
        spin_angle = wheel_spin_accum[idx]
        points_lists = [
            _wheel_points(
                hub_pos + axle_dir * off,
                axle_dir,
                v1,
                v2,
                spin_angle,
                wheel.radius,
                num_points,
            )
            for off in offsets
        ]
        for points in points_lists:
            for i in range(num_points):
                p1 = points[i]
                p2 = points[(i + 1) % num_points]
                main_vertices.extend(list(p1) + tire_color)
                main_vertices.extend(list(p2) + tire_color)
        connect_indices = list(range(0, num_points, num_points // 8))
        for i in connect_indices:
            p1 = points_lists[0][i]
            p2 = points_lists[1][i]
            main_vertices.extend(list(p1) + tire_color)
            main_vertices.extend(list(p2) + tire_color)
        # Flip sign so visual spin matches forward travel direction
        wheel_spin_accum[idx] = spin_angle - wheel.ang_vel * dt

        shock_start = hub_pos + car_up * wheel.radius
        shock_end = shock_start + car_up * suspension_length
        shock_vertices.extend(list(shock_start) + susp_color)
        shock_vertices.extend(list(shock_end) + susp_color)

    show_wind_vectors = bool(getattr(car, "show_wind_vectors", False))
    wind_vec = car.wind_velocity
    wind_mag = float(np.linalg.norm(wind_vec))

    if show_wind_vectors:
        # Wind resistance lines follow the relative airflow acting on the car body.
        rel_air = car.body.vel - wind_vec
        rel_speed = np.linalg.norm(rel_air)
        if rel_speed > 5.0:
            drag_mag = (
                0.5
                * AIR_DENSITY
                * car.drag_coeff
                * car.frontal_area
                * rel_speed**2
            )
            drag_dir = rel_air / rel_speed
            line_length = min(drag_mag / 100, 5)
            rear_top_left = car.body.pos + car.body.rot.rotate(
                np.array([half_width, half_height + car.body_offset, -half_length])
            )
            rear_top_right = car.body.pos + car.body.rot.rotate(
                np.array([-half_width, half_height + car.body_offset, -half_length])
            )
            wind_color = list(WIND_COLOR)
            for start_pos in [rear_top_left, rear_top_right]:
                end_pos = start_pos - drag_dir * line_length
                main_vertices.extend(list(start_pos) + wind_color)
                main_vertices.extend(list(end_pos) + wind_color)

        # Ambient wind indicator arrows rendered along the top edges for quick cues.
        if wind_mag > 0.05:
            wind_dir = wind_vec / wind_mag
            indicator_length = min(0.6 + wind_mag * 0.45, 3.2)
            ambient_color = list(AMBIENT_WIND_COLOR)
            top_edges = ((0, 1), (1, 5), (5, 4), (4, 0))
            for a, b in top_edges:
                start_pos = 0.5 * (world_corners[a] + world_corners[b])
                end_pos = start_pos + wind_dir * indicator_length
                main_vertices.extend(list(start_pos) + ambient_color)
                main_vertices.extend(list(end_pos) + ambient_color)

    return main_vertices, shock_vertices

# car.py
import math
import numpy as np


def _wheel_points(offset_pos, axle_dir, v1, v2, angle, radius, num):
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    pts = []
    for i in range(num):
        t = 2 * math.pi * i / num
        local = v1 * math.cos(t) + v2 * math.sin(t)
        rot = (
            local * cos_a
            - np.cross(axle_dir, local) * sin_a
            + axle_dir * np.dot(axle_dir, local) * (1 - cos_a)
        )
        pts.append(offset_pos + rot * radius)
    return pts


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
    car_color = [1.0, 0.0, 0.0, 1.0]
    for a, b in edges:
        pa = world_corners[a]
        pb = world_corners[b]
        main_vertices.extend(list(pa) + car_color)
        main_vertices.extend(list(pb) + car_color)

    # Wheels
    for idx, wheel in enumerate(car.wheels):
        hub_pos = car.body.pos + car.body.rot.rotate(wheel.rel_pos)
        ground_h = car.terrain.get_height(hub_pos[0], hub_pos[2])
        if math.isfinite(ground_h):
            compression = ground_h + wheel.radius - hub_pos[1]
            suspension_length = max(0.1, wheel.suspension_rest - compression)
            compression_ratio = max(0, min(1, compression / wheel.suspension_rest))
        else:
            compression = 0
            suspension_length = wheel.suspension_rest
            compression_ratio = 0
        susp_color = [1 - compression_ratio, compression_ratio, 0.0, 1.0]
        tire_color = [wheel.slip_ratio, 0.0, 0.0, 1.0] if wheel.is_grounded else [0.5, 0.5, 0.5, 1.0]

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
        wheel_spin_accum[idx] = spin_angle + wheel.ang_vel * dt

        shock_start = hub_pos + car_up * wheel.radius
        shock_end = shock_start + car_up * suspension_length
        shock_vertices.extend(list(shock_start) + susp_color)
        shock_vertices.extend(list(shock_end) + susp_color)

    # Wind resistance lines
    vel_mag = np.linalg.norm(car.body.vel)
    if vel_mag > 5:
        drag_mag = car.drag_coeff * vel_mag**2
        line_length = min(drag_mag / 100, 5)
        rear_top_left = car.body.pos + car.body.rot.rotate(np.array([half_width, half_height + car.body_offset, -half_length]))
        rear_top_right = car.body.pos + car.body.rot.rotate(np.array([-half_width, half_height + car.body_offset, -half_length]))
        wind_color = [200/255, 200/255, 255/255, 1.0]
        for start_pos in [rear_top_left, rear_top_right]:
            end_pos = start_pos - car_dir * line_length
            main_vertices.extend(list(start_pos) + wind_color)
            main_vertices.extend(list(end_pos) + wind_color)

    return main_vertices, shock_vertices

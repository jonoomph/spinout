# main.py
import pygame
from pygame.locals import *
import math
import numpy as np
from physics import Quaternion, RigidBody, Wheel, Terrain, Car
from controls import get_controls
from hud import render_hud

pygame.init()
width, height = 1024, 768
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Simple Car Driving Sim")
clock = pygame.time.Clock()

terrain = Terrain(size=400)
car = Car(terrain)
start_x, start_z = terrain.size / 2, terrain.size / 2
car.body.pos = np.array([start_x, terrain.get_height(start_x, start_z) + 2, start_z])

fov = 60
fov_scale = width / 2 / math.tan(math.radians(fov / 2))

def project(camera_pos, camera_right, camera_up, camera_forward, p):
    rel = p - camera_pos
    dx = np.dot(rel, camera_right)
    dy = np.dot(rel, camera_up)
    dz = np.dot(rel, camera_forward)
    if dz < 0.1:
        return None
    sx = dx / dz * fov_scale + width / 2
    sy = -dy / dz * fov_scale + height / 2
    return (int(sx), int(sy))

running = True
font = pygame.font.SysFont(None, 24)
view_dist = 150  # Increased slightly for larger terrain/screen
substeps = 2  # 2x physics steps
wheel_spin_accum = [0.0] * 4  # Accumulated spin angle per wheel

while running:
    dt = clock.tick(60) / 1000.0
    if dt > 0.05:
        dt = 0.05

    for event in pygame.event.get():
        if event.type == QUIT:
            running = False

    keys = pygame.key.get_pressed()
    car.steer, car.accel, car.brake = get_controls(keys)

    # Physics substeps
    sub_dt = dt / substeps
    for _ in range(substeps):
        car.update(sub_dt)

    screen.fill((135, 206, 235))

    # Camera setup
    car_dir = car.body.rot.rotate(np.array([0, 0, 1]))
    car_up = car.body.rot.rotate(np.array([0, 1, 0]))
    cam_dist = 8
    cam_height = 2
    camera_pos = car.body.pos - car_dir * cam_dist + np.array([0, cam_height, 0])
    camera_forward = car.body.pos - camera_pos
    camera_forward /= np.linalg.norm(camera_forward)
    camera_right = np.cross(camera_forward, np.array([0, 1, 0]))
    camera_right /= np.linalg.norm(camera_right)
    camera_up = np.cross(camera_right, camera_forward)
    camera_up /= np.linalg.norm(camera_up)

    # Draw terrain with culling
    for i in range(terrain.res - 1):
        for j in range(terrain.res - 1):
            cell_x = (i + 0.5) * terrain.cell_size
            cell_z = (j + 0.5) * terrain.cell_size
            cell_pos = np.array([cell_x, 0, cell_z])
            dist = np.linalg.norm(cell_pos - camera_pos)
            if dist > view_dist:
                continue
            x1 = i * terrain.cell_size
            z1 = j * terrain.cell_size
            x2 = (i + 1) * terrain.cell_size
            z2 = (j + 1) * terrain.cell_size
            p1 = np.array([x1, terrain.heights[i, j], z1])
            p2 = np.array([x2, terrain.heights[i + 1, j], z1])
            p3 = np.array([x2, terrain.heights[i + 1, j + 1], z2])
            p4 = np.array([x1, terrain.heights[i, j + 1], z2])
            pp1 = project(camera_pos, camera_right, camera_up, camera_forward, p1)
            pp2 = project(camera_pos, camera_right, camera_up, camera_forward, p2)
            pp3 = project(camera_pos, camera_right, camera_up, camera_forward, p3)
            pp4 = project(camera_pos, camera_right, camera_up, camera_forward, p4)
            if pp1 and pp2:
                pygame.draw.line(screen, (34, 139, 34), pp1, pp2)
            if pp2 and pp3:
                pygame.draw.line(screen, (34, 139, 34), pp2, pp3)
            if pp3 and pp4:
                pygame.draw.line(screen, (34, 139, 34), pp3, pp4)
            if pp4 and pp1:
                pygame.draw.line(screen, (34, 139, 34), pp4, pp1)

    # Draw car body
    half_length = 1.2
    half_width = 0.8
    half_height = 0.6
    corners_rel = [
        np.array([half_width, half_height, half_length]),
        np.array([half_width, half_height, -half_length]),
        np.array([half_width, -half_height, half_length]),
        np.array([half_width, -half_height, -half_length]),
        np.array([-half_width, half_height, half_length]),
        np.array([-half_width, half_height, -half_length]),
        np.array([-half_width, -half_height, half_length]),
        np.array([-half_width, -half_height, -half_length]),
    ]
    world_corners = [car.body.pos + car.body.rot.rotate(c) for c in corners_rel]
    proj_corners = [project(camera_pos, camera_right, camera_up, camera_forward, p) for p in world_corners]
    edges = [
        (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6),
        (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)
    ]
    for a, b in edges:
        pa, pb = proj_corners[a], proj_corners[b]
        if pa and pb:
            pygame.draw.line(screen, (255, 0, 0), pa, pb)

    # Draw wheels with accurate spinning and suspension shocks
    for idx, wheel in enumerate(car.wheels):
        hub_pos = car.body.pos + car.body.rot.rotate(wheel.rel_pos)
        ground_h = terrain.get_height(hub_pos[0], hub_pos[2])
        compression = ground_h + wheel.radius - hub_pos[1]
        suspension_length = max(0.1, wheel.suspension_rest - compression)  # Avoid zero length
        # Color based on compression: Red (short) to Green (tall)
        compression_ratio = max(0, min(1, compression / wheel.suspension_rest))  # 0 = fully compressed, 1 = no compression
        r = int(255 * compression_ratio)  # Wait, user said Red=short (compressed), Green=tall (not compressed)
        g = int(255 * (1 - compression_ratio))
        b = 0
        color = (g, r, b)  # Swapped r/g: compressed (low length) = red, uncompressed = green

        local_steer = wheel.steer_angle
        local_axle = np.array([math.cos(local_steer), 0, -math.sin(local_steer)])
        axle_dir = car.body.rot.rotate(local_axle)
        axle_dir /= np.linalg.norm(axle_dir)
        arbitrary = np.array([0, 1, 0]) if abs(axle_dir[1]) < 0.9 else np.array([1, 0, 0])
        v1 = np.cross(axle_dir, arbitrary)
        v1 /= np.linalg.norm(v1)
        v2 = np.cross(axle_dir, v1)
        v2 /= np.linalg.norm(v2)
        tire_width = 0.2
        offsets = [-tire_width / 2, tire_width / 2]
        num_points = 4  # Reduced for perf (square-ish wheel)
        points_lists = []
        spin_angle = wheel_spin_accum[idx]  # Use accumulated angle
        cos_spin = math.cos(spin_angle)
        sin_spin = math.sin(spin_angle)
        for offset in offsets:
            offset_pos = hub_pos + axle_dir * offset
            points = []
            for i in range(num_points):
                theta = 2 * math.pi * i / num_points
                local_point = v1 * math.cos(theta) + v2 * math.sin(theta)
                # Rotate around axle for spinning
                rotated_point = local_point * cos_spin - np.cross(axle_dir, local_point) * sin_spin + axle_dir * np.dot(axle_dir, local_point) * (1 - cos_spin)
                point = offset_pos + rotated_point * wheel.radius
                points.append(point)
            points_lists.append(points)
        # Draw circles
        for points in points_lists:
            for i in range(num_points):
                pp1 = project(camera_pos, camera_right, camera_up, camera_forward, points[i])
                pp2 = project(camera_pos, camera_right, camera_up, camera_forward, points[(i + 1) % num_points])
                if pp1 and pp2:
                    pygame.draw.line(screen, (0, 0, 0), pp1, pp2, 2)
        # Draw connecting lines
        connect_indices = [0, num_points // 2]  # Reduced for perf
        for i in connect_indices:
            pp1 = project(camera_pos, camera_right, camera_up, camera_forward, points_lists[0][i])
            pp2 = project(camera_pos, camera_right, camera_up, camera_forward, points_lists[1][i])
            if pp1 and pp2:
                pygame.draw.line(screen, (0, 0, 0), pp1, pp2, 2)
        # Update spin accumulation based on wheel speed
        wheel_spin_accum[idx] = spin_angle + wheel.ang_vel * dt

        # Draw suspension shock above tire
        shock_start = hub_pos + car_up * wheel.radius  # Start above hub
        shock_end = shock_start + car_up * suspension_length  # Upward
        pp_start = project(camera_pos, camera_right, camera_up, camera_forward, shock_start)
        pp_end = project(camera_pos, camera_right, camera_up, camera_forward, shock_end)
        if pp_start and pp_end:
            pygame.draw.line(screen, color, pp_start, pp_end, 6)  # Thicker (6 px)

    # Draw wind resistance lines from back top vertices
    vel_mag = np.linalg.norm(car.body.vel)
    if vel_mag > 5:
        drag_mag = car.drag_coeff * vel_mag**2
        line_length = min(drag_mag / 100, 5)
        rear_top_left = car.body.pos + car.body.rot.rotate(np.array([half_width, half_height, -half_length]))
        rear_top_right = car.body.pos + car.body.rot.rotate(np.array([-half_width, half_height, -half_length]))
        for start_pos in [rear_top_left, rear_top_right]:
            end_pos = start_pos - car_dir * line_length
            pp_start = project(camera_pos, camera_right, camera_up, camera_forward, start_pos)
            pp_end = project(camera_pos, camera_right, camera_up, camera_forward, end_pos)
            if pp_start and pp_end:
                pygame.draw.line(screen, (200, 200, 255), pp_start, pp_end, 2)

    # HUD
    speed_mph = np.linalg.norm(car.body.vel) * 2.23694
    render_fps = clock.get_fps()
    physics_fps = render_fps * substeps
    steer_angle = car.wheels[0].steer_angle if car.wheels[0].is_front else 0  # Use front-left wheel for steer angle
    render_hud(screen, font, speed_mph, render_fps, physics_fps, steer_angle)

    pygame.display.flip()

pygame.quit()
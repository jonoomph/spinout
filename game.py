# main.py
import pygame
from pygame.locals import *
import math
from physics import Vector3, Quaternion, RigidBody, Wheel, Terrain, Car
from controls import get_controls

pygame.init()
width, height = 1024, 768
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Simple Car Driving Sim")
clock = pygame.time.Clock()

terrain = Terrain(size=400, res=50, height_scale=100, sigma=6)
car = Car(terrain)
start_x, start_z = terrain.size / 2, terrain.size / 2
car.body.pos = Vector3(start_x, terrain.get_height(start_x, start_z) + 2, start_z)

fov = 60
fov_scale = width / 2 / math.tan(math.radians(fov / 2))

def project(camera_pos, camera_right, camera_up, camera_forward, p):
    rel = p - camera_pos
    dx = rel.dot(camera_right)
    dy = rel.dot(camera_up)
    dz = rel.dot(camera_forward)
    if dz < 0.1:
        return None
    sx = dx / dz * fov_scale + width / 2
    sy = -dy / dz * fov_scale + height / 2
    return (int(sx), int(sy))

running = True
font = pygame.font.SysFont(None, 24)  # Font for speed display
view_dist = 100  # View distance for culling terrain cells

while running:
    dt = clock.tick(60) / 1000.0
    if dt > 0.05:
        dt = 0.05

    for event in pygame.event.get():
        if event.type == QUIT:
            running = False

    keys = pygame.key.get_pressed()
    car.steer, car.accel, car.brake = get_controls(keys)

    car.update(dt)

    screen.fill((135, 206, 235))  # Sky blue

    # Camera setup
    car_dir = car.body.rot.rotate(Vector3(0, 0, 1))
    car_up = car.body.rot.rotate(Vector3(0, 1, 0))
    cam_dist = 8
    cam_height = 2
    camera_pos = car.body.pos - car_dir * cam_dist + Vector3(0, cam_height, 0)
    camera_forward = (car.body.pos - camera_pos).normalize()
    camera_right = camera_forward.cross(Vector3(0, 1, 0)).normalize()
    camera_up = camera_right.cross(camera_forward).normalize()

    # Draw terrain (wireframe for simplicity) with culling
    for i in range(terrain.res - 1):
        for j in range(terrain.res - 1):
            cell_x = (i + 0.5) * terrain.cell_size
            cell_z = (j + 0.5) * terrain.cell_size
            cell_pos = Vector3(cell_x, 0, cell_z)
            dist = (cell_pos - camera_pos).magnitude()
            if dist > view_dist:
                continue
            x1 = i * terrain.cell_size
            z1 = j * terrain.cell_size
            x2 = (i + 1) * terrain.cell_size
            z2 = (j + 1) * terrain.cell_size
            p1 = Vector3(x1, terrain.heights[i, j], z1)
            p2 = Vector3(x2, terrain.heights[i + 1, j], z1)
            p3 = Vector3(x2, terrain.heights[i + 1, j + 1], z2)
            p4 = Vector3(x1, terrain.heights[i, j + 1], z2)
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

    # Draw car body (simple box wireframe)
    half_length = 1.2
    half_width = 0.8
    half_height = 0.6
    corners_rel = [
        Vector3(half_width, half_height, half_length),
        Vector3(half_width, half_height, -half_length),
        Vector3(half_width, -half_height, half_length),
        Vector3(half_width, -half_height, -half_length),
        Vector3(-half_width, half_height, half_length),
        Vector3(-half_width, half_height, -half_length),
        Vector3(-half_width, -half_height, half_length),
        Vector3(-half_width, -half_height, -half_length),
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

    # Draw wheels as wireframe cylinders
    for wheel in car.wheels:
        hub_pos = car.body.pos + car.body.rot.rotate(wheel.rel_pos)
        local_steer = wheel.steer_angle
        local_axle = Vector3(math.cos(local_steer), 0, -math.sin(local_steer))
        axle_dir = car.body.rot.rotate(local_axle).normalize()
        arbitrary = Vector3(0, 1, 0) if abs(axle_dir.y) < 0.9 else Vector3(1, 0, 0)
        v1 = axle_dir.cross(arbitrary).normalize()
        v2 = axle_dir.cross(v1).normalize()
        tire_width = 0.2
        offsets = [-tire_width / 2, tire_width / 2]
        num_points = 6
        points_lists = []
        for offset in offsets:
            offset_pos = hub_pos + axle_dir * offset
            points = []
            for i in range(num_points):
                theta = 2 * math.pi * i / num_points
                point = offset_pos + (v1 * math.cos(theta) + v2 * math.sin(theta)) * wheel.radius
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
        connect_indices = [0, num_points // 3, 2 * num_points // 3]
        for i in connect_indices:
            pp1 = project(camera_pos, camera_right, camera_up, camera_forward, points_lists[0][i])
            pp2 = project(camera_pos, camera_right, camera_up, camera_forward, points_lists[1][i])
            if pp1 and pp2:
                pygame.draw.line(screen, (0, 0, 0), pp1, pp2, 2)

    # Display speed in mph
    speed_mph = car.body.vel.magnitude() * 2.23694  # m/s to mph conversion
    text = font.render(f"Speed: {speed_mph:.1f} mph", True, (0, 0, 0))
    screen.blit(text, (10, 10))

    pygame.display.flip()

pygame.quit()
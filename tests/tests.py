import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.physics import Terrain, Car

TARGET_SPEED_MPS = 60 * 0.44704  # 60 mph to m/s

VISUALIZE = 0

def simulate(car, accel=True):
    dt = 0.01
    substeps = 5
    time = 0.0
    if accel:
        car.accel = 1
        car.brake = 0
        condition = lambda: np.linalg.norm(car.body.vel) < TARGET_SPEED_MPS
    else:
        car.accel = 0
        car.brake = 1
        condition = lambda: np.linalg.norm(car.body.vel) > 0.1

    while condition() and time < 60:
        for _ in range(substeps):
            car.update(dt / substeps)
        time += dt
    return time


def run_test(car_data, visualize=VISUALIZE):
    """Run acceleration and braking tests on a flat, sufficiently large terrain."""
    terrain = Terrain(size=800, res=200, height_scale=0, sigma=0)
    terrain.heights[:] = 0

    car = Car(terrain, car_data)
    start = terrain.size / 4
    # Place the car so the wheels rest on the ground to avoid an initial drop
    rest_y = terrain.get_height(start, start) + car.wheels[0].radius + car.wheels[0].suspension_rest
    car.body.pos = np.array([start, rest_y, start])

    # Let the suspension settle before the timed tests
    car.accel = 0
    car.brake = 1
    for _ in range(200):
        car.update(0.01 / 5)

    if visualize:
        return run_test_visual(car, terrain)

    accel_time = simulate(car, accel=True)

    # brief coast before braking
    car.accel = 0
    for _ in range(5):
        car.update(0.002)

    brake_time = simulate(car, accel=False)
    return accel_time, brake_time


def run_test_visual(car, terrain):
    import pygame
    from src.render import RenderContext
    from src.utils import compute_mvp
    from src.terrain import build_terrain_vertices
    from src.car import collect_car_vertices

    pygame.init()
    width, height = 800, 600
    pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)
    clock = pygame.time.Clock()
    render_ctx = RenderContext(width, height)
    t_vbo = render_ctx.ctx.buffer(build_terrain_vertices(terrain).tobytes())
    t_vao = render_ctx.ctx.vertex_array(render_ctx.prog, t_vbo, "in_vert", "in_color")
    wheel_spin = [0.0] * 4
    substeps = 5

    def draw(dt):
        car_dir = car.body.rot.rotate(np.array([0, 0, 1]))
        car_up = car.body.rot.rotate(np.array([0, 1, 0]))
        cam_pos = car.body.pos - car_dir * 10 + np.array([0, 3, 0])
        cam_fwd = -(car.body.pos - cam_pos)
        cam_fwd /= np.linalg.norm(cam_fwd)
        cam_right = np.cross(cam_fwd, np.array([0, 1, 0]))
        cam_right /= np.linalg.norm(cam_right)
        cam_up = np.cross(cam_right, cam_fwd)
        cam_up /= np.linalg.norm(cam_up)
        mvp = compute_mvp(width, height, cam_pos, cam_right, cam_fwd, cam_up)
        render_ctx.clear()
        render_ctx.render_terrain(t_vao, mvp)
        verts = collect_car_vertices(car, car_up, car_dir, dt, wheel_spin)
        render_ctx.render_car(verts, mvp)
        pygame.display.flip()

    time = 0.0
    car.accel = 1
    car.brake = 0
    while np.linalg.norm(car.body.vel) < TARGET_SPEED_MPS and time < 60:
        dt = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return 0.0, 0.0
        for _ in range(substeps):
            car.update(dt / substeps)
        draw(dt)
        time += dt
    accel_time = time

    car.accel = 0
    for _ in range(5):
        for _ in range(substeps):
            car.update(0.002 / substeps)
        draw(0.002)

    time = 0.0
    car.brake = 1
    while np.linalg.norm(car.body.vel) > 0.1 and time < 60:
        dt = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return accel_time, time
        for _ in range(substeps):
            car.update(dt / substeps)
        draw(dt)
        time += dt
    brake_time = time
    pygame.quit()
    return accel_time, brake_time

with open(os.path.join(os.path.dirname(__file__), '../data/cars.json')) as f:
    CARS = json.load(f)


import pytest

@pytest.mark.parametrize('car_data', CARS, ids=[f"{c['make']}_{c['model']}" for c in CARS])
def test_car_performance(car_data):
    accel_exp = car_data['tests']['0_60_mph_accel_s']
    brake_exp = car_data['tests']['60_0_mph_brake_s']
    accel_t, brake_t = run_test(car_data, visualize=VISUALIZE)
    assert abs(accel_t - accel_exp) / accel_exp <= 0.05
    assert abs(brake_t - brake_exp) / brake_exp <= 0.05

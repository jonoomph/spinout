import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.physics import Car, Terrain

TARGET_SPEED_MPS = 60 * 0.44704  # 60 mph to m/s
SUCCESS_DIFF = 0.75
VISUALIZE = 0


def simulate(car, accel=True):
    """
    Run a simple physics simulation and return the elapsed time.

    On accel:
      - Interpolates the last step to hit TARGET_SPEED_MPS exactly.

    On brake:
      - Stops as soon as the per‑step deceleration stays below a threshold
        for a few consecutive steps (to ignore tiny post‑stop jitter).
    """
    dt = 0.01
    substeps = 5
    time = 0.0

    if accel:
        car.accel = 1
        car.brake = 0
        target_speed = TARGET_SPEED_MPS
        speed_fn = lambda: np.linalg.norm(car.body.vel)
    else:
        car.accel = 0
        car.brake = 1
        fwd_vector = car.body.rot.rotate(np.array([0, 0, 1]))
        target_speed = 0.0
        speed_fn = lambda: np.dot(car.body.vel, fwd_vector)

        # settings to ignore tiny sliding after full stop
        decel_threshold = 0.5   # m/s²
        consec_limit    = 5     # how many consecutive small steps before we break
        small_count     = 0

    prev_speed = speed_fn()

    while time < 60:
        for _ in range(substeps):
            car.update(dt / substeps)

        curr_speed = speed_fn()
        time += dt

        if accel:
            # interpolate the final accel step
            if prev_speed < target_speed <= curr_speed:
                span = (curr_speed - prev_speed) + 1e-8
                ratio = (target_speed - prev_speed) / span
                time -= dt * (1 - ratio)
                break

        else:
            # compute full‑step acceleration
            acc = (curr_speed - prev_speed) / dt

            # count tiny decelerations
            if abs(acc) < decel_threshold:
                small_count += 1
            else:
                small_count = 0

            # break when we've truly stopped
            if curr_speed <= 0 or small_count >= consec_limit:
                break

        prev_speed = curr_speed

    return time


def run_test(car_data, visualize=VISUALIZE):
    """Run acceleration and braking tests on a flat, sufficiently large terrain."""
    terrain = Terrain(size=800, res=200, height_scale=0, sigma=0)
    terrain.heights[:] = 0

    car = Car(terrain, car_data)
    start = terrain.size / 4
    # Place the car so the wheels rest on the ground to avoid an initial drop
    rest_y = (
        terrain.get_height(start, start)
        + car.wheels[0].radius
        + car.wheels[0].suspension_rest
    )
    car.body.pos = np.array([start, rest_y, start])

    # Let the suspension settle before the timed tests
    car.accel = 0
    car.brake = 1
    for _ in range(200):
        car.update(0.01 / 5)

    if visualize:
        accel_exp = car_data["tests"]["0_60_mph_accel_s"]
        brake_exp = car_data["tests"]["60_0_mph_brake_s"]
        return run_test_visual(car, terrain, accel_exp, brake_exp)

    accel_time = simulate(car, accel=True)

    # brief coast before braking
    car.accel = 0
    for _ in range(5):
        car.update(0.002)

    brake_time = simulate(car, accel=False)
    return accel_time, brake_time


def run_test_visual(car, terrain, accel_expected, brake_expected):
    """Visualize the test run while displaying HUD information and timers."""
    import pygame

    from src.car import collect_car_vertices
    from src.render import RenderContext
    from src.terrain import build_terrain_vertices
    from src.utils import compute_mvp

    pygame.init()
    width, height = 800, 600
    pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)
    clock = pygame.time.Clock()
    render_ctx = RenderContext(width, height)
    t_vbo = render_ctx.ctx.buffer(build_terrain_vertices(terrain).tobytes())
    t_vao = render_ctx.ctx.vertex_array(render_ctx.prog, t_vbo, "in_vert", "in_color")
    wheel_spin = [0.0] * 4
    substeps = 5
    font = pygame.font.SysFont(None, 24)
    hud_surf = pygame.Surface((width, height), pygame.SRCALPHA)
    accel_elapsed = 0.0
    brake_elapsed = 0.0
    diff_surf_accel = None
    diff_surf_brake = None

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
        hud_surf.fill((0, 0, 0, 0))
        speed_mph = np.linalg.norm(car.body.vel) * 2.23694
        text_speed = font.render(
            f"Speed: {speed_mph:.1f} mph [gear {car.current_gear} @ {int(car.engine_rpm)} RPM]",
            True,
            (255, 255, 255, 255),
        )
        accel_label = f"Accel: {accel_elapsed:.2f}s"
        brake_label = f"Brake: {brake_elapsed:.2f}s"
        surf_accel = font.render(accel_label, True, (255, 255, 255, 255))
        surf_brake = font.render(brake_label, True, (255, 255, 255, 255))
        x_accel = 10
        x_brake = x_accel + surf_accel.get_width() + 20
        hud_surf.blit(text_speed, (10, 10))
        hud_surf.blit(surf_accel, (x_accel, 40))
        hud_surf.blit(surf_brake, (x_brake, 40))
        if diff_surf_accel is not None:
            x_accel_diff = x_accel + (surf_accel.get_width() - diff_surf_accel.get_width()) / 2
            hud_surf.blit(diff_surf_accel, (x_accel_diff, 60))
        if diff_surf_brake is not None:
            x_brake_diff = x_brake + (surf_brake.get_width() - diff_surf_brake.get_width()) / 2
            hud_surf.blit(diff_surf_brake, (x_brake_diff, 60))
        render_ctx.render_hud(hud_surf)
        pygame.display.set_caption(
            f"{speed_mph:.1f} mph | ACCEL: {accel_elapsed:.2f}s BRAKE: {brake_elapsed:.2f}s"
        )
        pygame.display.flip()

    time = 0.0
    accel_elapsed = 0.0
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
        accel_elapsed = time
    accel_time = time

    diff_a = accel_time - accel_expected
    color_a = (0, 200, 0) if abs(diff_a) <= SUCCESS_DIFF else (200, 0, 0)
    diff_surf_accel = font.render(f"{diff_a:+.2f}s", True, color_a)

    car.accel = 0
    for _ in range(5):
        for _ in range(substeps):
            car.update(0.002 / substeps)
        draw(0.002)

    time = 0.0
    brake_elapsed = 0.0
    car.brake = 1
    fwd = car.body.rot.rotate(np.array([0, 0, 1]))
    prev_fwd = np.dot(car.body.vel, fwd)
    while prev_fwd > 0 and time < 60:
        dt = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return accel_time, time
        for _ in range(substeps):
            car.update(dt / substeps)
        curr_fwd = np.dot(car.body.vel, fwd)
        draw(dt)
        time += dt
        brake_elapsed = time
        if prev_fwd > 0 and curr_fwd <= 0:
            span = abs(curr_fwd - prev_fwd) + 1e-8
            ratio = abs(prev_fwd) / span
            time -= dt * (1 - ratio)
            brake_elapsed = time
            draw(0)
            break
        prev_fwd = curr_fwd
    brake_time = time
    diff_b = brake_time - brake_expected
    color_b = (0, 200, 0) if abs(diff_b) <= SUCCESS_DIFF else (200, 0, 0)
    diff_surf_brake = font.render(f"{diff_b:+.2f}s", True, color_b)

    end = pygame.time.get_ticks() + 1500
    while pygame.time.get_ticks() < end:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return accel_time, brake_time
        draw(0)
        clock.tick(60)
    pygame.quit()
    return accel_time, brake_time


with open(os.path.join(os.path.dirname(__file__), "../data/cars.json")) as f:
    CARS = json.load(f)


import pytest


def _make_class(car_data, name):
    class CarTest:
        @classmethod
        def setup_class(cls):
            cls.accel_expected = car_data["tests"]["0_60_mph_accel_s"]
            cls.brake_expected = car_data["tests"]["60_0_mph_brake_s"]
            cls.accel_time, cls.brake_time = run_test(car_data, visualize=VISUALIZE)

        def test_accel(self):
            assert abs(self.accel_time - self.accel_expected) <= SUCCESS_DIFF

        def test_brake(self):
            assert abs(self.brake_time - self.brake_expected) <= SUCCESS_DIFF

    CarTest.__name__ = f"Test_{name}"
    return CarTest


for car in CARS:
    name = f"{car['make']}_{car['model']}".replace(" ", "_").replace("-", "_")
    globals()[f"Test_{name}"] = _make_class(car, name)

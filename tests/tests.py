"""Simple regression tests for vehicle acceleration and braking."""

import json
import os
import sys

import numpy as np
import pytest

# Allow running tests directly from repository root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environment import Environment

# Target speeds and tolerances for the 0-60/60-0 tests
TARGET_SPEED_MPS = 60 * 0.44704  # 60 mph to m/s
SUCCESS_DIFF = 0.75
VISUALIZE = int(os.environ.get("VISUALIZE", 0))


def simulate(env: Environment, accel: bool = True) -> float:
    """Run a simple acceleration or braking simulation and return elapsed time."""

    car = env.car
    dt = env.dt
    substeps = env.substeps
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

        decel_threshold = 0.5
        consec_limit = 5
        small_count = 0

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


def run_test(car_data, visualize: int = VISUALIZE):
    """Return measured 0→60 and 60→0 times for ``car_data``."""

    idx = CARS.index(car_data)
    cfg = {"flat": True, "car_index": idx}
    mode = "eval" if visualize else "train"
    env = Environment(cfg, mode=mode)
    env.reset()

    if visualize:
        return run_test_visual(
            env,
            car_data["tests"]["0_60_mph_accel_s"],
            car_data["tests"]["60_0_mph_brake_s"],
        )

    car = env.car
    car.accel = 0
    car.brake = 1
    for _ in range(200):  # settle suspension
        car.update(env.dt / env.substeps)

    accel_time = simulate(env, accel=True)

    car.accel = 0
    for _ in range(5):  # short coast before braking
        car.update(env.dt / env.substeps)

    brake_time = simulate(env, accel=False)
    return accel_time, brake_time


def run_test_visual(env: Environment, accel_expected, brake_expected):
    """Interactive visualisation for manual inspection of test runs."""

    import pygame

    from src.car import collect_car_vertices
    from src.render import RenderContext
    from src.terrain import build_terrain_triangles
    from src.utils import compute_mvp

    pygame.init()
    width, height = 800, 600
    pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)
    clock = pygame.time.Clock()
    render_ctx = RenderContext(width, height)
    render_ctx.set_mode(1)
    render_ctx.setup_weather("dry", env.terrain.terrain_type, "asphalt")
    t_basic, _ = build_terrain_triangles(env.terrain)
    t_vbo = render_ctx.ctx.buffer(t_basic.tobytes())
    t_vao = render_ctx.ctx.vertex_array(render_ctx.prog, t_vbo, "in_vert", "in_color")
    wheel_spin = [0.0] * 4
    font = pygame.font.SysFont(None, 24)
    hud_surf = pygame.Surface((width, height), pygame.SRCALPHA)
    accel_elapsed = 0.0
    brake_elapsed = 0.0
    diff_surf_accel = None
    diff_surf_brake = None

    def draw(dt):
        """Render terrain, car and text overlays for the current frame."""

        car = env.car
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
        render_ctx.set_camera(cam_pos)
        render_ctx.clear()
        render_ctx.render_terrain(t_vao, mvp, render_ctx.road_noise)
        verts = collect_car_vertices(car, car_up, car_dir, dt, wheel_spin)
        render_ctx.render_car(verts, mvp)
        hud_surf.fill((0, 0, 0, 0))
        speed_mph = np.linalg.norm(car.body.vel) * 2.23694
        text_speed = font.render(
            f"Speed: {speed_mph:.1f} mph", True, (255, 255, 255, 255)
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
            x_accel_diff = x_accel + (
                surf_accel.get_width() - diff_surf_accel.get_width()
            ) / 2
            hud_surf.blit(diff_surf_accel, (x_accel_diff, 60))
        if diff_surf_brake is not None:
            x_brake_diff = x_brake + (
                surf_brake.get_width() - diff_surf_brake.get_width()
            ) / 2
            hud_surf.blit(diff_surf_brake, (x_brake_diff, 60))
        render_ctx.render_hud(hud_surf)
        pygame.display.set_caption(
            f"{speed_mph:.1f} mph | ACCEL: {accel_elapsed:.2f}s BRAKE: {brake_elapsed:.2f}s"
        )
        pygame.display.flip()

    time_elapsed = 0.0
    env.car.accel = 1
    env.car.brake = 0
    while np.linalg.norm(env.car.body.vel) < TARGET_SPEED_MPS and time_elapsed < 60:
        dt = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return 0.0, 0.0
        for _ in range(env.substeps):
            env.car.update(dt / env.substeps)
        draw(dt)
        time_elapsed += dt
        accel_elapsed = time_elapsed
    accel_time = time_elapsed

    diff_a = accel_time - accel_expected
    color_a = (0, 200, 0) if abs(diff_a) <= SUCCESS_DIFF else (200, 0, 0)
    diff_surf_accel = font.render(f"{diff_a:+.2f}s", True, color_a)

    env.car.accel = 0
    for _ in range(5):
        for _ in range(env.substeps):
            env.car.update(0.002 / env.substeps)
        draw(0.002)

    time_elapsed = 0.0
    env.car.brake = 1
    fwd = env.car.body.rot.rotate(np.array([0, 0, 1]))
    prev_fwd = np.dot(env.car.body.vel, fwd)
    while prev_fwd > 0 and time_elapsed < 60:
        dt = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return accel_time, time_elapsed
        for _ in range(env.substeps):
            env.car.update(dt / env.substeps)
        curr_fwd = np.dot(env.car.body.vel, fwd)
        draw(dt)
        time_elapsed += dt
        brake_elapsed = time_elapsed
        if prev_fwd > 0 and curr_fwd <= 0:
            span = abs(curr_fwd - prev_fwd) + 1e-8
            ratio = abs(prev_fwd) / span
            time_elapsed -= dt * (1 - ratio)
            brake_elapsed = time_elapsed
            draw(0)
            break
        prev_fwd = curr_fwd
    brake_time = time_elapsed
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


def _make_class(car_data, name):
    """Create a ``pytest`` class for ``car_data``."""

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


# Dynamically generate a test class for each car configuration
for car in CARS:
    name = f"{car['make']}_{car['model']}".replace(" ", "_").replace("-", "_")
    globals()[f"Test_{name}"] = _make_class(car, name)

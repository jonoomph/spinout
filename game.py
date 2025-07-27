# game.py
import pygame
from pygame.locals import *
from physics import Terrain, Car
from controls import get_controls
from hud import render_hud
from render import RenderContext
from utils import compute_mvp
from terrain import build_terrain_vertices
from car import collect_car_vertices
import numpy as np

pygame.init()
width, height = 1920, 1080
screen = pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)
pygame.display.set_caption("Simple Car Driving Sim")
clock = pygame.time.Clock()

render_ctx = RenderContext(width, height)
terrain = Terrain(size=400)
car = Car(terrain)
start_x, start_z = terrain.size / 2, terrain.size / 2
car.body.pos = np.array([start_x, terrain.get_height(start_x, start_z) + 2, start_z])

terrain_vbo = render_ctx.ctx.buffer(build_terrain_vertices(terrain).tobytes())
terrain_vao = render_ctx.ctx.vertex_array(render_ctx.prog, terrain_vbo, 'in_vert', 'in_color')

running = True
font = pygame.font.SysFont(None, 24)
substeps = 2
wheel_spin_accum = [0.0] * 4

while running:
    dt = clock.tick(60) / 1000.0
    if dt > 0.05:
        dt = 0.05

    for event in pygame.event.get():
        if event.type == QUIT:
            running = False

    keys = pygame.key.get_pressed()
    car.steer, car.accel, car.brake = get_controls(keys)

    for _ in range(substeps):
        car.update(dt / substeps)

    car_dir = car.body.rot.rotate(np.array([0, 0, 1]))
    car_up = car.body.rot.rotate(np.array([0, 1, 0]))
    cam_dist = 8
    cam_height = 2
    camera_pos = car.body.pos - car_dir * cam_dist + np.array([0, cam_height, 0])
    camera_forward = -(car.body.pos - camera_pos)
    camera_forward /= np.linalg.norm(camera_forward)
    camera_right = np.cross(camera_forward, np.array([0, 1, 0]))
    camera_right /= np.linalg.norm(camera_right)
    camera_up = np.cross(camera_right, camera_forward)
    camera_up /= np.linalg.norm(camera_up)

    mvp = compute_mvp(width, height, camera_pos, camera_right, camera_forward, camera_up)
    render_ctx.clear()
    render_ctx.render_terrain(terrain_vao, mvp)
    render_ctx.render_car(collect_car_vertices(car, car_up, car_dir, dt, wheel_spin_accum), mvp)

    speed_mph = np.linalg.norm(car.body.vel) * 2.23694
    render_fps = clock.get_fps()
    physics_fps = render_fps * substeps
    steer_angle = car.wheels[0].steer_angle if car.wheels[0].is_front else 0
    hud_surf = pygame.Surface((width, height), pygame.SRCALPHA)
    hud_surf.fill((0, 0, 0, 0))
    render_hud(hud_surf, font, speed_mph, render_fps, physics_fps, steer_angle)
    render_ctx.render_hud(hud_surf)

    pygame.display.flip()

pygame.quit()
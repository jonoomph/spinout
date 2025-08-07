# game.py
import pygame
from pygame.locals import *
import time
import numpy as np
import json
import moderngl

from src.physics import Terrain, Car
from src.roads.plan import generate_plan, get_safe_start_position_and_rot
from src.roads.build import apply_plan, build_road_vertices, build_speed_sign_vertices
from src.signs import generate_speed_limit_sign
from src.controls import get_controls
from src.hud import render_hud
from src.render import RenderContext
from src.utils import compute_mvp
from src.terrain import build_terrain_triangles
from src.car import collect_car_vertices
from src.bbmodel import load_bbmodel, collect_car_model_vertices
from src.colors import (
    ROAD_ASPHALT_COLOR,
    ROAD_CONCRETE_COLOR,
    ROAD_GRAVEL_COLOR,
    TERRAIN_GRASS_COLOR,
    TERRAIN_SAND_COLOR,
    TERRAIN_SNOW_COLOR,
    TERRAIN_DIRT_COLOR,
)

# Screen dimensions and constants
WIDTH, HEIGHT = 1854, 1168
WEATHER_MODIFIERS = {"dry": 1.0, "wet": 0.7}

ROAD_TYPES = {
    "asphalt": {"color": ROAD_ASPHALT_COLOR, "friction": 1.0},
    "concrete": {"color": ROAD_CONCRETE_COLOR, "friction": 0.95},
    "gravel": {"color": ROAD_GRAVEL_COLOR, "friction": 0.8},
}

TERRAIN_TYPES = {
    "grass": {"color": TERRAIN_GRASS_COLOR, "friction": 0.7},
    "sand": {"color": TERRAIN_SAND_COLOR, "friction": 0.6},
    "dirt": {"color": TERRAIN_DIRT_COLOR, "friction": 0.6},
    "snow": {"color": TERRAIN_SNOW_COLOR, "friction": 0.5},
}


def show_loading(progress, message, surface, font):
    """Draw a centered loading bar with text."""
    w, h = surface.get_size()
    surface.fill((0, 0, 0))
    bar_w, bar_h = int(w * 0.6), 40
    x, y = (w - bar_w) // 2, (h - bar_h) // 2
    pygame.draw.rect(surface, (255, 255, 255), (x, y, bar_w, bar_h), 2)
    inner_w = int(bar_w * progress)
    pygame.draw.rect(surface, (255, 255, 255), (x, y, inner_w, bar_h))
    txt = font.render(message, True, (255, 255, 255))
    surface.blit(txt, txt.get_rect(center=(w // 2, y - 30)))
    pygame.display.flip()


# Initialize Pygame and show loading in software mode
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Loading...")
loading_font = pygame.font.SysFont(None, 48)

# Terrain and road generation
show_loading(0.2, "Generating terrain...", screen, loading_font)
rng = np.random.default_rng()
weather = rng.choice(["dry", "wet"], p=[0.7, 0.3])
road_type = rng.choice(list(ROAD_TYPES), p=[0.7, 0.2, 0.1])
terrain_type = rng.choice(list(TERRAIN_TYPES), p=[0.55, 0.15, 0.15, 0.15])
weather_mod = WEATHER_MODIFIERS[weather]

surface_info = f"{weather.title()} {road_type.title()} | {terrain_type.title()}"

show_loading(0.5, "Laying roads...", screen, loading_font)
t = TERRAIN_TYPES[terrain_type]
terrain = Terrain(size=800, res=200,
                  terrain_type=terrain_type,
                  color=t["color"],
                  friction=t["friction"] * weather_mod)
rp, plan = generate_plan(terrain, rng=rng,
                         road_type=road_type,
                         weather=weather,
                         terrain_type=terrain_type,
                         road_color=ROAD_TYPES[road_type]["color"],
                         skirt_color=terrain.color,
                         road_friction=ROAD_TYPES[road_type]["friction"] * weather_mod)
apply_plan(terrain, rp, plan, rng=rng)

# Build meshes and switch to OpenGL mode
show_loading(0.8, "Building meshes...", screen, loading_font)
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)
pygame.display.set_caption("Spinout")
clock = pygame.time.Clock()
render_ctx = RenderContext(WIDTH, HEIGHT)
render_ctx.setup_weather(weather, terrain_type, road_type)

# Load Blockbench car model and upload its texture
car_model_data = load_bbmodel("data/car.bbmodel")
car_tex = render_ctx.ctx.texture(
    car_model_data["texture_size"], 4, car_model_data["texture_bytes"]
)
car_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
render_ctx.car_model_tex = car_tex

# Road VAOs
road_verts = build_road_vertices(terrain, rp, **plan)
road_vbo = render_ctx.ctx.buffer(road_verts.tobytes())
road_vao = render_ctx.ctx.vertex_array(render_ctx.prog, road_vbo, 'in_vert', 'in_color')
road_lit = np.hstack([road_verts.reshape(-1, 7)[:, :3],
                      np.tile([0, 1, 0], (road_verts.size // 7, 1)),
                      road_verts.reshape(-1, 7)[:, 3:7]]).astype('f4')
road_vbo_lit = render_ctx.ctx.buffer(road_lit.tobytes())
road_vao_lit = render_ctx.ctx.vertex_array(render_ctx.prog_lit,
                                           road_vbo_lit,
                                           'in_vert', 'in_normal', 'in_color')

# Speed limit signs
posts, quads = build_speed_sign_vertices(terrain, rp,
                                         lane_width=plan["lane_width"],
                                         lanes=plan["lanes"],
                                         shoulder=plan["shoulder"],
                                         speed_limits=plan.get("speed_limits"))
sign_post_vao = None
if posts.size:
    vbo = render_ctx.ctx.buffer(posts.tobytes())
    sign_post_vao = render_ctx.ctx.vertex_array(render_ctx.prog,
                                                vbo,
                                                'in_vert', 'in_color')
sign_billboards = []
for quad in quads:
    vbo = render_ctx.ctx.buffer(quad["verts"].tobytes())
    vao = render_ctx.ctx.vertex_array(render_ctx.prog_tex, vbo,
                                      'in_vert', 'in_tex')
    img = generate_speed_limit_sign(quad["speed"])
    tex = render_ctx.ctx.texture(img.size, 4, img.tobytes())
    tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
    sign_billboards.append((vao, tex))

# Terrain VAOs
tb, tl = build_terrain_triangles(terrain)
terrain_vbo = render_ctx.ctx.buffer(tb.tobytes())
terrain_vao = render_ctx.ctx.vertex_array(render_ctx.prog,
                                          terrain_vbo,
                                          'in_vert', 'in_color')
terrain_vbo_lit = render_ctx.ctx.buffer(tl.tobytes())
terrain_vao_lit = render_ctx.ctx.vertex_array(render_ctx.prog_lit,
                                              terrain_vbo_lit,
                                              'in_vert', 'in_normal', 'in_color')

# Final loading
show_loading(1.0, "Starting engines...", screen, loading_font)
time.sleep(0.5)

# Load cars and initial placement
with open("data/cars.json") as f:
    cars = json.load(f)
idx = 0
car = Car(terrain, cars[idx])
pos, rot = get_safe_start_position_and_rot(terrain, rp, 5.0)
car.body.pos, car.body.rot = pos, rot

# HUD fonts
tiny = pygame.font.SysFont(None, 24)
big = pygame.font.SysFont(None, 48)
hud_h = big.get_height() * 2

# Main loop
running = True
mode = 0
substeps = 4
wheel_acc = [0.0] * 4
camera_mode = 0
use_bbmodel = False
while running:
    t0 = time.time()
    dt = clock.tick(60) / 1000
    if dt > 0.05: dt = 0.05

    # events
    for e in pygame.event.get():
        if e.type == QUIT:
            running = False
        elif e.type == KEYDOWN:
            if e.key == K_F1:
                mode = 0
            elif e.key == K_F2:
                mode = 1
            elif e.key == K_F3:
                mode = 2
            elif e.key == K_c:
                camera_mode = (camera_mode + 1) % 3
            elif e.key == K_b:
                use_bbmodel = not use_bbmodel
            elif e.key == K_v:
                render_ctx.wetness = 0.0 if render_ctx.wetness > 0.0 else 1.0
            elif e.key == K_t:
                render_ctx.cycle_terrain_mode()

    # controls & car switching
    s_i, a_i, b_i, new = get_controls(pygame.key.get_pressed())
    if new is not None and new != idx:
        idx = max(0, min(len(cars) - 1, new))
        oldp = car.body.pos
        car = Car(terrain, cars[idx])
        car.body.pos = oldp
        wheel_acc = [0.0] * 4

    car.apply_inputs(s_i, a_i, b_i)
    for _ in range(substeps): car.update(dt / substeps)

    # camera views
    car_dir = car.body.rot.rotate(np.array([0, 0, 1]))
    car_up_vec = car.body.rot.rotate(np.array([0, 1, 0]))
    if camera_mode == 2:
        # Drivers view
        car_forward = car.body.rot.rotate(np.array([0, 0, 1]))
        car_up = car.body.rot.rotate(np.array([0, 1, 0]))
        car_right = car.body.rot.rotate(np.array([1, 0, 0]))
        cam_offset = car_up * 0.30 - car_forward * 0.18
        camera_pos = car.body.pos + cam_offset
        forward = -car_forward / np.linalg.norm(car_forward)
        right = car_right / np.linalg.norm(car_right)
        up_vec = car_up / np.linalg.norm(car_up)
    else:
        # Follow view
        cam_dist = 8 if camera_mode == 0 else 4
        cam_hgt = 2 if camera_mode == 0 else 1.2
        camera_pos = car.body.pos - car_dir * cam_dist + np.array([0, cam_hgt, 0])
        forward = -(car.body.pos - camera_pos)
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.array([0, 1, 0]))
        right /= np.linalg.norm(right)
        up_vec = np.cross(right, forward)
        up_vec /= np.linalg.norm(up_vec)
    render_ctx.set_camera(camera_pos)
    mvp = compute_mvp(WIDTH, HEIGHT,
                      camera_pos,
                      right,
                      forward,
                      up_vec)

    # render scene
    render_ctx.set_mode(mode)
    render_ctx.clear()
    t_vao = terrain_vao_lit if mode == 2 else terrain_vao
    r_vao = road_vao_lit if mode == 2 else road_vao
    render_ctx.render_terrain(t_vao, mvp)
    render_ctx.render_terrain(r_vao, mvp, render_ctx.road_noise, terrain_mode=0)
    if sign_post_vao: render_ctx.render_signs(sign_post_vao, mvp)
    for vao, tex in sign_billboards:
        render_ctx.render_billboard(vao, tex, mvp)
    car_lines = collect_car_vertices(car, car_up_vec, car_dir, dt, wheel_acc)
    if use_bbmodel:
        model_verts = collect_car_model_vertices(car, car_model_data)
        render_ctx.render_car_model(model_verts, mvp)
    else:
        render_ctx.render_car(car_lines, mvp)
    render_ctx.render_weather(mvp, dt)

    # HUD
    spd = np.linalg.norm(car.body.vel) * 2.23694
    fps_r = clock.get_fps()
    fps_p = fps_r * substeps
    steer_angle = next(w.steer_angle for w in car.wheels if w.is_front)
    bar = pygame.Surface((WIDTH, hud_h), pygame.SRCALPHA)
    render_hud(bar, tiny, big,
               spd, fps_r, fps_p, steer_angle,
               f"{cars[idx]['make']} {cars[idx]['model']} ({cars[idx]['year']})",
               rpm=car.engine_rpm, gear=car.current_gear,
               surface_info=surface_info,
               render_mode=mode,
               camera_mode=camera_mode)
    full = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    full.blit(bar, (0, 0))
    render_ctx.render_hud(full)

    pygame.display.flip()
    if (time.time() - t0) * 1000 > 33:
        print(f"Slow frame: {(time.time() - t0) * 1000:.1f}ms")

pygame.quit()

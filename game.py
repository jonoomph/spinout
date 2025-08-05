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


WEATHER_MODIFIERS = {"dry": 1.0, "wet": 0.7}

ROAD_TYPES = {
    "asphalt": {"color": [0.3, 0.3, 0.3, 1.0], "friction": 1.0},
    "concrete": {"color": [0.7, 0.7, 0.7, 1.0], "friction": 0.95},
    "gravel": {"color": [0.36, 0.25, 0.2, 1.0], "friction": 0.8},
}

TERRAIN_TYPES = {
    "grass": {"color": [34 / 255, 139 / 255, 34 / 255, 1.0], "friction": 0.7},
    "sand": {"color": [0.76, 0.70, 0.50, 1.0], "friction": 0.6},
    "snow": {"color": [1.0, 1.0, 1.0, 1.0], "friction": 0.5},
}


def show_loading(progress, message, screen, font):
    width, height = screen.get_size()
    screen.fill((0, 0, 0))
    bar_w = int(width * 0.6)
    bar_h = 40
    x = (width - bar_w) // 2
    y = (height - bar_h) // 2
    pygame.draw.rect(screen, (255, 255, 255), (x, y, bar_w, bar_h), 2)
    inner_w = int(bar_w * progress)
    pygame.draw.rect(screen, (255, 255, 255), (x, y, inner_w, bar_h))
    text = font.render(message, True, (255, 255, 255))
    text_rect = text.get_rect(center=(width // 2, y - 30))
    screen.blit(text, text_rect)
    pygame.display.flip()

pygame.init()
width, height = 1854, 1168
screen = pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)
pygame.display.set_caption("Simple Car Driving Sim")
clock = pygame.time.Clock()

loading_font = pygame.font.SysFont(None, 48)
rng = np.random.default_rng()

weather = rng.choice(["dry", "wet"], p=[0.7, 0.3])
road_type = rng.choice(["asphalt", "concrete", "gravel"], p=[0.7, 0.2, 0.1])
terrain_type = rng.choice(["grass", "sand", "snow"], p=[0.7, 0.15, 0.15])

weather_mod = WEATHER_MODIFIERS[weather]
road_info = ROAD_TYPES[road_type]
terrain_info = TERRAIN_TYPES[terrain_type]
surface_info = f"Terrain: {terrain_type.title()}, Road: {weather.title()} {road_type.title()}"

show_loading(0.2, "Generating terrain...", screen, loading_font)
terrain = Terrain(
    size=800,
    res=200,
    terrain_type=terrain_type,
    color=terrain_info["color"],
    friction=terrain_info["friction"] * weather_mod,
)

show_loading(0.5, "Laying roads...", screen, loading_font)
road_points, road_plan = generate_plan(
    terrain,
    rng=rng,
    road_type=road_type,
    weather=weather,
    terrain_type=terrain_type,
    road_color=road_info["color"],
    skirt_color=terrain.color,
    road_friction=road_info["friction"] * weather_mod,
)
apply_plan(terrain, road_points, road_plan, rng=rng)

show_loading(0.8, "Building meshes...", screen, loading_font)
render_ctx = RenderContext(width, height)
road_vertices = build_road_vertices(terrain, road_points, **road_plan)
road_vbo = render_ctx.ctx.buffer(road_vertices.tobytes())
road_vao = render_ctx.ctx.vertex_array(render_ctx.prog, road_vbo, 'in_vert', 'in_color')
road_vertices_lit = np.hstack([
    road_vertices.reshape(-1, 7)[:, :3],
    np.tile([0.0, 1.0, 0.0], (road_vertices.size // 7, 1)),
    road_vertices.reshape(-1, 7)[:, 3:7],
]).astype('f4')
road_vbo_lit = render_ctx.ctx.buffer(road_vertices_lit.tobytes())
road_vao_lit = render_ctx.ctx.vertex_array(
    render_ctx.prog_lit, road_vbo_lit, 'in_vert', 'in_normal', 'in_color'
)

# Build speed limit sign geometry and textures so they can be drawn filled even in wireframe mode
sign_posts, sign_quads = build_speed_sign_vertices(
    terrain,
    road_points,
    lane_width=road_plan["lane_width"],
    lanes=road_plan["lanes"],
    shoulder=road_plan["shoulder"],
    speed_limits=road_plan.get("speed_limits"),
)

sign_post_vao = None
if sign_posts.size > 0:
    sign_post_vbo = render_ctx.ctx.buffer(sign_posts.tobytes())
    sign_post_vao = render_ctx.ctx.vertex_array(render_ctx.prog, sign_post_vbo, 'in_vert', 'in_color')

sign_billboards = []
for quad in sign_quads:
    vbo = render_ctx.ctx.buffer(quad["verts"].tobytes())
    vao = render_ctx.ctx.vertex_array(render_ctx.prog_tex, vbo, 'in_vert', 'in_tex')
    img = generate_speed_limit_sign(quad["speed"])
    tex = render_ctx.ctx.texture(img.size, 4, img.tobytes())
    tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
    sign_billboards.append((vao, tex))
terrain_basic, terrain_lit = build_terrain_triangles(terrain)
terrain_vbo = render_ctx.ctx.buffer(terrain_basic.tobytes())
terrain_vao = render_ctx.ctx.vertex_array(render_ctx.prog, terrain_vbo, 'in_vert', 'in_color')
terrain_vbo_lit = render_ctx.ctx.buffer(terrain_lit.tobytes())
terrain_vao_lit = render_ctx.ctx.vertex_array(
    render_ctx.prog_lit, terrain_vbo_lit, 'in_vert', 'in_normal', 'in_color'
)

show_loading(1.0, "Starting engines...", screen, loading_font)
time.sleep(0.5)

# Load cars from JSON
with open("data/cars.json", "r") as f:
    cars_data = json.load(f)

# Initialize with first car
current_car_index = 0
car = Car(terrain, cars_data[current_car_index])

# Place car on road, rotate in correct direction
car_pos, car_rot = get_safe_start_position_and_rot(terrain, road_points, 5.0)
car.body.pos = car_pos
car.body.rot = car_rot

running = True
render_mode = 0
font = pygame.font.SysFont(None, 24)
substeps = 2
wheel_spin_accum = [0.0] * 4

while running:
    start_time = time.time()
    dt = clock.tick(60) / 1000.0
    if dt > 0.05:
        dt = 0.05

    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        elif event.type == KEYDOWN:
            if event.key == K_F1:
                render_mode = 0
            elif event.key == K_F2:
                render_mode = 1
            elif event.key == K_F3:
                render_mode = 2

    keys = pygame.key.get_pressed()
    steer_i, accel_i, brake_i, car_index = get_controls(keys)
    if car_index is not None and car_index != current_car_index:
        # Change car
        current_car_index = max(0, min(len(cars_data) - 1, car_index))
        old_pos = car.body.pos
        car = Car(terrain, cars_data[current_car_index])
        car.body.pos = old_pos
        wheel_spin_accum = [0.0] * 4

    car.apply_inputs(steer_i, accel_i, brake_i)

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
    render_ctx.set_mode(render_mode)
    render_ctx.clear()
    t_vao = terrain_vao_lit if render_mode == 2 else terrain_vao
    r_vao = road_vao_lit if render_mode == 2 else road_vao
    render_ctx.render_terrain(t_vao, mvp)
    render_ctx.render_terrain(r_vao, mvp)
    if sign_post_vao:
        render_ctx.render_signs(sign_post_vao, mvp)
    for vao, tex in sign_billboards:
        render_ctx.render_billboard(vao, tex, mvp)
    render_ctx.render_car(collect_car_vertices(car, car_up, car_dir, dt, wheel_spin_accum), mvp)

    speed_mph = np.linalg.norm(car.body.vel) * 2.23694
    render_fps = clock.get_fps()
    physics_fps = render_fps * substeps
    steer_angle = car.wheels[0].steer_angle if car.wheels[0].is_front else 0
    hud_surf = pygame.Surface((width, height), pygame.SRCALPHA)
    hud_surf.fill((0, 0, 0, 0))
    car_info = (
        f"Car: {cars_data[current_car_index]['make']} "
        f"{cars_data[current_car_index]['model']} "
        f"({cars_data[current_car_index]['year']})"
    )
    render_hud(
        hud_surf,
        font,
        speed_mph,
        render_fps,
        physics_fps,
        steer_angle,
        car_info,
        rpm=car.engine_rpm,
        gear=car.current_gear,
        surface_info=surface_info,
        render_mode=render_mode,
    )
    render_ctx.render_hud(hud_surf)

    pygame.display.flip()

    # Basic profiling
    frame_time = time.time() - start_time
    if frame_time > 0.033:
        print(f"Slow frame: {frame_time*1000:.1f}ms")

pygame.quit()

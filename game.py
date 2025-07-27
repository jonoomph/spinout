# game.py
import pygame
from pygame.locals import *
import math
import numpy as np
from physics import Quaternion, RigidBody, Wheel, Terrain, Car
from controls import get_controls
from hud import render_hud
import moderngl

pygame.init()
width, height = 1920, 1080
screen = pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)
pygame.display.set_caption("Simple Car Driving Sim")
clock = pygame.time.Clock()

ctx = moderngl.create_context()

# 3D shader
vertex_shader = '''
#version 330
in vec3 in_vert;
in vec4 in_color;
uniform mat4 mvp;
out vec4 color;
void main() {
    gl_Position = mvp * vec4(in_vert, 1.0);
    color = in_color;
}
'''
fragment_shader = '''
#version 330
in vec4 color;
out vec4 fragColor;
void main() {
    fragColor = color;
}
'''
prog = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)

# 2D shader for HUD
vertex_shader_2d = '''
#version 330
in vec2 in_pos;
in vec2 in_tex;
out vec2 v_tex;
uniform mat4 mvp;
void main() {
    gl_Position = mvp * vec4(in_pos, 0.0, 1.0);
    v_tex = in_tex;
}
'''
fragment_shader_2d = '''
#version 330
in vec2 v_tex;
uniform sampler2D tex;
out vec4 fragColor;
void main() {
    fragColor = texture(tex, v_tex);
}
'''
prog2d = ctx.program(vertex_shader=vertex_shader_2d, fragment_shader=fragment_shader_2d)

# Ortho for HUD
ortho = np.array([
    [2.0 / width, 0.0, 0.0, 0.0],
    [0.0, 2.0 / height, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [-1.0, -1.0, 0.0, 1.0]
], dtype='f4')

# HUD quad
hud_quad_data = np.array([
    0, 0, 0, 1,
    width, 0, 1, 1,
    0, height, 0, 0,
    width, 0, 1, 1,
    width, height, 1, 0,
    0, height, 0, 0,
], dtype='f4')
hud_vbo = ctx.buffer(hud_quad_data.tobytes())
vao2d = ctx.vertex_array(prog2d, hud_vbo, 'in_pos', 'in_tex')

# Enable blend for HUD
ctx.enable(moderngl.BLEND)
ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

terrain = Terrain(size=400)
car = Car(terrain)
start_x, start_z = terrain.size / 2, terrain.size / 2
car.body.pos = np.array([start_x, terrain.get_height(start_x, start_z) + 2, start_z])

fov = 60
fov_rad = math.radians(fov)
fov_scale = width / 2 / math.tan(fov_rad / 2)

# Precompute terrain vertices
terrain_vertices = []
terrain_color = [34/255, 139/255, 34/255, 1.0]
for i in range(terrain.res - 1):
    for j in range(terrain.res - 1):
        x1 = i * terrain.cell_size
        z1 = j * terrain.cell_size
        x2 = (i + 1) * terrain.cell_size
        z2 = (j + 1) * terrain.cell_size
        p1 = [x1, terrain.heights[i, j], z1]
        p2 = [x2, terrain.heights[i + 1, j], z1]
        p3 = [x2, terrain.heights[i + 1, j + 1], z2]
        p4 = [x1, terrain.heights[i, j + 1], z2]
        terrain_vertices.extend(p1 + terrain_color + p2 + terrain_color)
        terrain_vertices.extend(p2 + terrain_color + p3 + terrain_color)
        terrain_vertices.extend(p3 + terrain_color + p4 + terrain_color)
        terrain_vertices.extend(p4 + terrain_color + p1 + terrain_color)
terrain_vbo = ctx.buffer(np.array(terrain_vertices, dtype='f4').tobytes())
terrain_vao = ctx.vertex_array(prog, terrain_vbo, 'in_vert', 'in_color')

running = True
font = pygame.font.SysFont(None, 24)
substeps = 2  # 2x physics steps
wheel_spin_accum = [0.0] * 4  # Accumulated spin angle per wheel

def quat_to_mat(q):
    w, x, y, z = q.w, q.x, q.y, q.z
    mat = np.eye(4, dtype='f4')
    mat[0,0] = 1 - 2*(y**2 + z**2)
    mat[0,1] = 2*(x*y - z*w)
    mat[0,2] = 2*(x*z + y*w)
    mat[1,0] = 2*(x*y + z*w)
    mat[1,1] = 1 - 2*(x**2 + z**2)
    mat[1,2] = 2*(y*z - x*w)
    mat[2,0] = 2*(x*z - y*w)
    mat[2,1] = 2*(y*z + x*w)
    mat[2,2] = 1 - 2*(x**2 + y**2)
    return mat

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

    # Camera setup
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

    # View matrix: Extend view_rot to 4x4
    basis = np.column_stack((camera_right, camera_up, camera_forward))
    view_rot = np.eye(4, dtype='f4')
    view_rot[:3, :3] = basis.T  # 3x3 rotation part
    trans = np.eye(4, dtype='f4')
    trans[:3, 3] = -camera_pos  # Translation part
    view = view_rot @ trans

    # Projection matrix (perspective)
    aspect = width / height
    near, far = 0.1, 200.0
    proj = np.zeros((4,4), dtype='f4')
    proj[0,0] = 1 / (aspect * math.tan(fov_rad / 2))
    proj[1,1] = 1 / math.tan(fov_rad / 2)
    proj[2,2] = -(far + near) / (far - near)
    proj[2,3] = -2 * far * near / (far - near)
    proj[3,2] = -1

    # Clear screen
    ctx.clear(135/255, 206/255, 235/255)

    # Render terrain
    mvp = proj @ view @ np.eye(4, dtype='f4')
    prog['mvp'].write(mvp.T.tobytes())
    terrain_vao.render(moderngl.LINES)

    # Collect dynamic vertices
    dynamic_vertices = []

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
    edges = [
        (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6),
        (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)
    ]
    car_color = [1.0, 0.0, 0.0, 1.0]
    for a, b in edges:
        pa = world_corners[a]
        pb = world_corners[b]
        dynamic_vertices.extend(list(pa) + car_color)
        dynamic_vertices.extend(list(pb) + car_color)

    # Draw wheels and suspensions
    black_color = [0.0, 0.0, 0.0, 1.0]
    for idx, wheel in enumerate(car.wheels):
        hub_pos = car.body.pos + car.body.rot.rotate(wheel.rel_pos)
        ground_h = terrain.get_height(hub_pos[0], hub_pos[2])
        compression = ground_h + wheel.radius - hub_pos[1]
        suspension_length = max(0.1, wheel.suspension_rest - compression)
        compression_ratio = max(0, min(1, compression / wheel.suspension_rest))
        r = compression_ratio  # Red for compressed
        g = 1 - compression_ratio  # Green for uncompressed
        b = 0.0
        susp_color = [g, r, b, 1.0]

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
        num_points = 4
        points_lists = []
        spin_angle = wheel_spin_accum[idx]
        cos_spin = math.cos(spin_angle)
        sin_spin = math.sin(spin_angle)
        for offset in offsets:
            offset_pos = hub_pos + axle_dir * offset
            points = []
            for i in range(num_points):
                theta = 2 * math.pi * i / num_points
                local_point = v1 * math.cos(theta) + v2 * math.sin(theta)
                rotated_point = local_point * cos_spin - np.cross(axle_dir, local_point) * sin_spin + axle_dir * np.dot(axle_dir, local_point) * (1 - cos_spin)
                point = offset_pos + rotated_point * wheel.radius
                points.append(point)
            points_lists.append(points)
        for points in points_lists:
            for i in range(num_points):
                p1 = points[i]
                p2 = points[(i + 1) % num_points]
                dynamic_vertices.extend(list(p1) + black_color)
                dynamic_vertices.extend(list(p2) + black_color)
        connect_indices = [0, num_points // 2]
        for i in connect_indices:
            p1 = points_lists[0][i]
            p2 = points_lists[1][i]
            dynamic_vertices.extend(list(p1) + black_color)
            dynamic_vertices.extend(list(p2) + black_color)
        wheel_spin_accum[idx] = spin_angle + wheel.ang_vel * dt

        shock_start = hub_pos + car_up * wheel.radius
        shock_end = shock_start + car_up * suspension_length
        dynamic_vertices.extend(list(shock_start) + susp_color)
        dynamic_vertices.extend(list(shock_end) + susp_color)

    # Draw wind resistance lines
    vel_mag = np.linalg.norm(car.body.vel)
    if vel_mag > 5:
        drag_mag = car.drag_coeff * vel_mag**2
        line_length = min(drag_mag / 100, 5)
        rear_top_left = car.body.pos + car.body.rot.rotate(np.array([half_width, half_height, -half_length]))
        rear_top_right = car.body.pos + car.body.rot.rotate(np.array([-half_width, half_height, -half_length]))
        wind_color = [200/255, 200/255, 255/255, 1.0]
        for start_pos in [rear_top_left, rear_top_right]:
            end_pos = start_pos - car_dir * line_length
            dynamic_vertices.extend(list(start_pos) + wind_color)
            dynamic_vertices.extend(list(end_pos) + wind_color)

    # Render dynamic
    if dynamic_vertices:
        dynamic_vbo = ctx.buffer(np.array(dynamic_vertices, dtype='f4').tobytes())
        dynamic_vao = ctx.vertex_array(prog, dynamic_vbo, 'in_vert', 'in_color')
        dynamic_vao.render(moderngl.LINES)

    # HUD
    speed_mph = np.linalg.norm(car.body.vel) * 2.23694
    render_fps = clock.get_fps()
    physics_fps = render_fps * substeps
    steer_angle = car.wheels[0].steer_angle if car.wheels[0].is_front else 0
    hud_surf = pygame.Surface((width, height), pygame.SRCALPHA)
    hud_surf.fill((0, 0, 0, 0))
    render_hud(hud_surf, font, speed_mph, render_fps, physics_fps, steer_angle)
    hud_data = pygame.image.tostring(hud_surf, 'RGBA', False)
    hud_tex = ctx.texture((width, height), 4, hud_data)
    hud_tex.use(0)
    prog2d['mvp'].write(ortho.T.tobytes())
    prog2d['tex'] = 0
    vao2d.render(moderngl.TRIANGLES)

    pygame.display.flip()

pygame.quit()
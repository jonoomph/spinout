# Simple demo of Modern GL, dynamic day/night cycle, dynamic shadows, grass particles

import math
import time
import sys

import numpy as np
import pygame
import moderngl


# --------------------
# Matrix helpers
# --------------------
def mat4_identity():
    return np.eye(4, dtype="f4")


def perspective(fovy_deg, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fovy_deg) / 2.0)
    m = np.zeros((4, 4), dtype="f4")
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def look_at(eye, target, up):
    eye = np.array(eye, dtype="f4")
    target = np.array(target, dtype="f4")
    up = np.array(up, dtype="f4")

    f = target - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)

    m = np.eye(4, dtype="f4")
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


def ortho(left, right, bottom, top, near, far):
    m = np.eye(4, dtype="f4")
    m[0, 0] = 2.0 / (right - left)
    m[1, 1] = 2.0 / (top - bottom)
    m[2, 2] = -2.0 / (far - near)
    m[0, 3] = -(right + left) / (right - left)
    m[1, 3] = -(top + bottom) / (top - bottom)
    m[2, 3] = -(far + near) / (far - near)
    return m


def translate(x, y, z):
    m = np.eye(4, dtype="f4")
    m[0, 3] = x
    m[1, 3] = y
    m[2, 3] = z
    return m


def scale(sx, sy, sz):
    m = np.eye(4, dtype="f4")
    m[0, 0] = sx
    m[1, 1] = sy
    m[2, 2] = sz
    return m


def to_gl(m):
    """Row-major numpy -> column-major float bytes for OpenGL."""
    return m.T.astype("f4").tobytes()


# --------------------
# Geometry (positions + normals)
# --------------------
def make_ground():
    # Ground plane centered at origin, y = 0
    size = 4.0
    y = 0.0
    n = [0.0, 1.0, 0.0]

    verts = [
        # tri 1 (CCW from above)
        [-size, y, -size, *n],
        [-size, y,  size, *n],
        [ size, y,  size, *n],
        # tri 2
        [-size, y, -size, *n],
        [ size, y,  size, *n],
        [ size, y, -size, *n],
    ]
    return np.array(verts, dtype="f4")


def make_cube():
    p = 0.5
    verts = []

    # +X
    n = [1, 0, 0]
    verts += [
        [p, -p, -p, *n],
        [p,  p, -p, *n],
        [p,  p,  p, *n],
        [p, -p, -p, *n],
        [p,  p,  p, *n],
        [p, -p,  p, *n],
    ]
    # -X
    n = [-1, 0, 0]
    verts += [
        [-p, -p,  p, *n],
        [-p,  p,  p, *n],
        [-p,  p, -p, *n],
        [-p, -p,  p, *n],
        [-p,  p, -p, *n],
        [-p, -p, -p, *n],
    ]
    # +Y
    n = [0, 1, 0]
    verts += [
        [-p,  p, -p, *n],
        [-p,  p,  p, *n],
        [ p,  p,  p, *n],
        [-p,  p, -p, *n],
        [ p,  p,  p, *n],
        [ p,  p, -p, *n],
    ]
    # -Y
    n = [0, -1, 0]
    verts += [
        [-p, -p,  p, *n],
        [-p, -p, -p, *n],
        [ p, -p, -p, *n],
        [-p, -p,  p, *n],
        [ p, -p, -p, *n],
        [ p, -p,  p, *n],
    ]
    # +Z
    n = [0, 0, 1]
    verts += [
        [-p, -p,  p, *n],
        [ p, -p,  p, *n],
        [ p,  p,  p, *n],
        [-p, -p,  p, *n],
        [ p,  p,  p, *n],
        [-p,  p,  p, *n],
    ]
    # -Z
    n = [0, 0, -1]
    verts += [
        [ p, -p, -p, *n],
        [-p, -p, -p, *n],
        [-p,  p, -p, *n],
        [ p, -p, -p, *n],
        [-p,  p, -p, *n],
        [ p,  p, -p, *n],
    ]
    return np.array(verts, dtype="f4")


def make_uv_sphere(radius=0.6, sectors=24, stacks=16):
    verts = []
    for i in range(stacks):
        stack0 = math.pi * (i / stacks - 0.5)
        stack1 = math.pi * ((i + 1) / stacks - 0.5)

        y0 = radius * math.sin(stack0)
        r0 = radius * math.cos(stack0)

        y1 = radius * math.sin(stack1)
        r1 = radius * math.cos(stack1)

        for j in range(sectors):
            s0 = 2.0 * math.pi * (j / sectors)
            s1 = 2.0 * math.pi * ((j + 1) / sectors)

            x00, z00 = r0 * math.cos(s0), r0 * math.sin(s0)
            x01, z01 = r0 * math.cos(s1), r0 * math.sin(s1)
            x10, z10 = r1 * math.cos(s0), r1 * math.sin(s0)
            x11, z11 = r1 * math.cos(s1), r1 * math.sin(s1)

            # tri 1
            for p in (
                np.array([x00, y0, z00], dtype="f4"),
                np.array([x10, y1, z10], dtype="f4"),
                np.array([x11, y1, z11], dtype="f4"),
            ):
                n = p / np.linalg.norm(p)
                verts.append([p[0], p[1], p[2], n[0], n[1], n[2]])

            # tri 2
            for p in (
                np.array([x00, y0, z00], dtype="f4"),
                np.array([x11, y1, z11], dtype="f4"),
                np.array([x01, y0, z01], dtype="f4"),
            ):
                n = p / np.linalg.norm(p)
                verts.append([p[0], p[1], p[2], n[0], n[1], n[2]])

    return np.array(verts, dtype="f4")


def make_cone(radius=0.7, height=2.0, segments=32):
    """
    Closed cone:
      - base circle at y = 0
      - tip at y = height
      - smooth outward side normals
      - upward-facing base cap
    Winding is set so outside is FRONT with backface culling.
    """
    verts = []
    h = float(height)
    r = float(radius)
    base_y = 0.0
    tip = np.array([0.0, h, 0.0], dtype="f4")

    # --- side surface ---
    slant = math.sqrt(h * h + r * r)
    ny = r / slant          # y component of side normal
    nr = h / slant          # radial scale for xz components

    for i in range(segments):
        theta0 = 2.0 * math.pi * (i / segments)
        theta1 = 2.0 * math.pi * ((i + 1) / segments)

        c0, s0 = math.cos(theta0), math.sin(theta0)
        c1, s1 = math.cos(theta1), math.sin(theta1)

        p0 = np.array([r * c0, base_y, r * s0], dtype="f4")
        p1 = np.array([r * c1, base_y, r * s1], dtype="f4")
        p2 = tip

        # outward normal for this side tri
        n0 = np.array([nr * c0, ny, nr * s0], dtype="f4")
        n1 = np.array([nr * c1, ny, nr * s1], dtype="f4")
        n2 = n0 + n1
        n2 /= np.linalg.norm(n2)

        # orientation p0 -> p1 -> tip: CCW when viewed from outside,
        # and n computed with cross(p1-p0, p2-p0)
        # (we don't explicitly use that cross here since we use smoothed normals)
        for p, n in ((p0, n0), (p1, n1), (p2, n2)):
            verts.append([p[0], p[1], p[2], n[0], n[1], n[2]])

    # --- base cap (faces UP, visible from above) ---
    center = np.array([0.0, base_y, 0.0], dtype="f4")
    n_base = np.array([0.0, 1.0, 0.0], dtype="f4")

    for i in range(segments):
        theta0 = 2.0 * math.pi * (i / segments)
        theta1 = 2.0 * math.pi * ((i + 1) / segments)

        c0, s0 = math.cos(theta0), math.sin(theta0)
        c1, s1 = math.cos(theta1), math.sin(theta1)

        p0 = np.array([r * c0, base_y, r * s0], dtype="f4")
        p1 = np.array([r * c1, base_y, r * s1], dtype="f4")

        # center -> p0 -> p1 gives upward normal
        for p in (center, p0, p1):
            verts.append([p[0], p[1], p[2], n_base[0], n_base[1], n_base[2]])

    return np.array(verts, dtype="f4")


def make_grass_blades(count=20000, area=3.6, min_h=0.4, max_h=0.9):
    """
    Bakes a field of vertical triangle "blades" directly in world space.
    Each blade:
      - small triangle rising from y=0
      - random x,z within the ground area
      - random height and orientation
    """
    verts = []
    for _ in range(count):
        x = np.random.uniform(-area, area)
        z = np.random.uniform(-area, area)
        h = np.random.uniform(min_h, max_h)
        yaw = np.random.uniform(0.0, 2.0 * math.pi)
        width = np.random.uniform(0.03, 0.06)

        dx = math.cos(yaw)
        dz = math.sin(yaw)

        base_center = np.array([x, 0.0, z], dtype="f4")
        # small base segment perpendicular to direction
        side = np.array([-dz, 0.0, dx], dtype="f4") * (width * 0.5)

        p0 = base_center - side
        p1 = base_center + side
        tip = base_center + np.array([0.0, h, 0.0], dtype="f4")

        # normal for the blade triangle
        e1 = p1 - p0
        e2 = tip - p0
        n = np.cross(e1, e2)
        if np.linalg.norm(n) == 0:
            n = np.array([0.0, 1.0, 0.0], dtype="f4")
        else:
            n = n / np.linalg.norm(n)

        for p in (p0, p1, tip):
            verts.append([p[0], p[1], p[2], n[0], n[1], n[2]])

    return np.array(verts, dtype="f4")


# --------------------
# Shaders
# --------------------
DEPTH_VS = """
#version 330
uniform mat4 mvp_light;
in vec3 in_pos;

void main() {
    gl_Position = mvp_light * vec4(in_pos, 1.0);
}
"""

DEPTH_FS = """
#version 330
void main() { }
"""

RENDER_VS = """
#version 330
in vec3 in_pos;
in vec3 in_nrm;

uniform mat4 m_model;
uniform mat4 m_view;
uniform mat4 m_proj;
uniform mat4 light_mvp;

out vec3 v_nrm_ws;
out vec4 v_pos_light;

void main() {
    vec4 world = m_model * vec4(in_pos, 1.0);
    v_nrm_ws = mat3(transpose(inverse(m_model))) * in_nrm;
    v_pos_light = light_mvp * world;
    gl_Position = m_proj * m_view * world;
}
"""

RENDER_FS = """
#version 330
in vec3 v_nrm_ws;
in vec4 v_pos_light;

uniform sampler2DShadow shadowmap;
uniform vec3 light_dir;
uniform float ambient_k;
uniform float sun_intensity;
uniform vec3 base_color;
uniform int draw_sun;    // 1 = draw solid white, 0 = normal shading

out vec4 fragColor;

float pcf_shadow(vec4 pos_light) {
    vec3 ndc = pos_light.xyz / pos_light.w;
    vec3 uvw = ndc * 0.5 + 0.5;

    if (uvw.x < 0.0 || uvw.x > 1.0 || uvw.y < 0.0 || uvw.y > 1.0) {
        return 1.0;
    }

    float texel = 1.0 / textureSize(shadowmap, 0).x;
    float result = 0.0;

    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            vec2 offs = vec2(x, y) * texel;
            result += texture(shadowmap, vec3(uvw.xy + offs, uvw.z - 0.0015));
        }
    }
    return result / 9.0;
}

void main() {
    // Sun marker: solid white, no shading/shadows
    if (draw_sun == 1) {
        fragColor = vec4(1.0);
        return;
    }

    vec3 N = normalize(v_nrm_ws);
    vec3 L = normalize(-light_dir);

    float ndotl = max(dot(N, L), 0.0);
    float s = pcf_shadow(v_pos_light);

    float lighting = ambient_k + sun_intensity * ndotl * s;
    lighting = clamp(lighting, 0.0, 1.0);

    fragColor = vec4(base_color * lighting, 1.0);
}
"""


# --------------------
# Mesh wrapper
# --------------------
def create_mesh(ctx, prog_render, prog_depth, vertices):
    """
    vertices: ndarray (N, 6) -> [px, py, pz, nx, ny, nz]
    """
    vertices = np.asarray(vertices, dtype="f4")
    vbo = ctx.buffer(vertices.tobytes())

    vao_render = ctx.vertex_array(
        prog_render,
        [(vbo, "3f 3f", "in_pos", "in_nrm")],
    )
    # depth pass only needs positions (skip normals)
    vao_depth = ctx.vertex_array(
        prog_depth,
        [(vbo, "3f 3x4", "in_pos")],
    )

    vert_count = len(vertices)
    return vbo, vao_render, vao_depth, vert_count


# --------------------
# Main
# --------------------
def main():
    pygame.init()
    width, height = 1280, 720
    pygame.display.set_mode(
        (width, height),
        pygame.OPENGL | pygame.DOUBLEBUF,
    )
    pygame.display.set_caption(
        "ModernGL + pygame: Moving sun + colored ground shadows + grass (30s orbit)"
    )

    ctx = moderngl.create_context()
    ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

    # Programs
    prog_depth = ctx.program(vertex_shader=DEPTH_VS, fragment_shader=DEPTH_FS)
    prog_render = ctx.program(vertex_shader=RENDER_VS, fragment_shader=RENDER_FS)
    prog_render["shadowmap"].value = 0

    # Shadow map
    shadow_size = 2048
    shadow_tex = ctx.depth_texture((shadow_size, shadow_size))
    shadow_tex.repeat_x = False
    shadow_tex.repeat_y = False
    shadow_tex.compare_func = "<="
    shadow_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    shadow_fbo = ctx.framebuffer(depth_attachment=shadow_tex)

    # Geometry
    ground_v, ground_vao_r, ground_vao_d, ground_count = create_mesh(
        ctx, prog_render, prog_depth, make_ground()
    )
    cube_v, cube_vao_r, cube_vao_d, cube_count = create_mesh(
        ctx, prog_render, prog_depth, make_cube()
    )
    sphere_v, sphere_vao_r, sphere_vao_d, sphere_count = create_mesh(
        ctx, prog_render, prog_depth, make_uv_sphere()
    )
    cone_v, cone_vao_r, cone_vao_d, cone_count = create_mesh(
        ctx, prog_render, prog_depth, make_cone()
    )
    grass_v, grass_vao_r, grass_vao_d, grass_count = create_mesh(
        ctx, prog_render, prog_depth, make_grass_blades()
    )
    # Small sphere for sun marker (render only)
    sun_v, sun_vao_r, sun_vao_d, sun_count = create_mesh(
        ctx, prog_render, prog_depth, make_uv_sphere(radius=0.2, sectors=16, stacks=12)
    )

    # Camera
    eye = [6.0, 4.0, 6.0]
    target = [0.0, 0.7, 0.0]
    up = [0.0, 1.0, 0.0]
    view = look_at(eye, target, up)
    proj = perspective(50.0, width / float(height), 0.1, 50.0)

    # Instances
    ground_model = mat4_identity()
    cube_models = [
        translate(-1.5, 0.5, -0.5) @ scale(1.0, 1.0, 1.0),
        translate(1.8, 0.5, 1.0) @ scale(1.0, 1.0, 1.0),
    ]
    sphere_models = [
        translate(-0.2, 0.6, 2.0),
        translate(2.2, 0.6, -1.7),
    ]
    cone_model = translate(0.0, 0.0, -2.2) @ scale(1.3, 1.3, 1.3)
    grass_model = mat4_identity()

    # Colors (RGB)
    ground_color = (0.15, 0.45, 0.18)   # grass green
    grass_color = (0.10, 0.35, 0.12)    # darker green for blades
    cube_colors = [
        (0.8, 0.3, 0.3),                # reddish
        (0.3, 0.3, 0.9),                # bluish
    ]
    sphere_colors = [
        (0.9, 0.9, 0.4),                # yellow-ish
        (0.6, 0.4, 0.9),                # purple-ish
    ]
    cone_color = (0.9, 0.6, 0.4)        # warm orange-ish

    start_time = time.time()
    clock = pygame.time.Clock()
    running = True

    while running:
        dt = clock.tick(60) / 1000.0  # unused, but handy if you add controls later

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        # -------------
        # Sun orbit: full circle in 30s around scene center
        # -------------
        t = time.time() - start_time
        period = 30.0
        angle = (t / period) * 2.0 * math.pi

        sun_radius = 12.0
        sun_pos = np.array(
            [
                math.cos(angle) * sun_radius,   # east–west
                math.sin(angle) * sun_radius,   # height above/below
                0.0,                            # fixed in Z
            ],
            dtype="f4",
        )
        light_dir = (-sun_pos / np.linalg.norm(sun_pos)).astype("f4")

        light_pos = sun_pos
        light_target = np.array([0.0, 0.0, 0.0], dtype="f4")
        light_up = np.array([0.0, 1.0, 0.0], dtype="f4")

        light_view = look_at(light_pos, light_target, light_up)
        extent = 8.0
        light_proj = ortho(-extent, extent, -extent, extent, 1.0, 30.0)
        light_mvp = light_proj @ light_view

        # Night/day intensity: when sun is below horizon (y<0) → full darkness
        sun_height = sun_pos[1] / sun_radius
        sun_above = max(0.0, sun_height)
        day_factor = min(1.0, sun_above * 3.0)

        # Dimmer, more balanced lighting
        ambient_k = 0.02 + 0.15 * day_factor   # small ambient, more at day
        sun_intensity = 1.0 * day_factor       # less harsh direct light

        # -------------
        # Depth pass (shadow map)
        # -------------
        shadow_fbo.use()
        ctx.viewport = (0, 0, shadow_size, shadow_size)
        ctx.clear(depth=1.0)

        prog_depth["mvp_light"].write(to_gl(light_mvp @ ground_model))
        ground_vao_d.render(moderngl.TRIANGLES, vertices=ground_count)

        for model in cube_models:
            prog_depth["mvp_light"].write(to_gl(light_mvp @ model))
            cube_vao_d.render(moderngl.TRIANGLES, vertices=cube_count)

        for model in sphere_models:
            prog_depth["mvp_light"].write(to_gl(light_mvp @ model))
            sphere_vao_d.render(moderngl.TRIANGLES, vertices=sphere_count)

        prog_depth["mvp_light"].write(to_gl(light_mvp @ cone_model))
        cone_vao_d.render(moderngl.TRIANGLES, vertices=cone_count)

        # grass: render into shadow map too, double-sided (disable culling)
        ctx.disable(moderngl.CULL_FACE)
        prog_depth["mvp_light"].write(to_gl(light_mvp @ grass_model))
        grass_vao_d.render(moderngl.TRIANGLES, vertices=grass_count)
        ctx.enable(moderngl.CULL_FACE)

        # -------------
        # Scene pass
        # -------------
        ctx.screen.use()
        ctx.viewport = (0, 0, width, height)
        ctx.clear(0.10, 0.10, 0.12, 1.0)

        shadow_tex.use(location=0)

        prog_render["m_view"].write(to_gl(view))
        prog_render["m_proj"].write(to_gl(proj))
        prog_render["light_mvp"].write(to_gl(light_mvp))
        prog_render["light_dir"].value = tuple(light_dir.tolist())
        prog_render["ambient_k"].value = ambient_k
        prog_render["sun_intensity"].value = sun_intensity

        prog_render["draw_sun"].value = 0

        # ground
        prog_render["base_color"].value = ground_color
        prog_render["m_model"].write(to_gl(ground_model))
        ground_vao_r.render(moderngl.TRIANGLES, vertices=ground_count)

        # grass (double-sided)
        ctx.disable(moderngl.CULL_FACE)
        prog_render["base_color"].value = grass_color
        prog_render["m_model"].write(to_gl(grass_model))
        grass_vao_r.render(moderngl.TRIANGLES, vertices=grass_count)
        ctx.enable(moderngl.CULL_FACE)

        # cubes
        for model, col in zip(cube_models, cube_colors):
            prog_render["base_color"].value = col
            prog_render["m_model"].write(to_gl(model))
            cube_vao_r.render(moderngl.TRIANGLES, vertices=cube_count)

        # spheres
        for model, col in zip(sphere_models, sphere_colors):
            prog_render["base_color"].value = col
            prog_render["m_model"].write(to_gl(model))
            sphere_vao_r.render(moderngl.TRIANGLES, vertices=sphere_count)

        # cone
        prog_render["base_color"].value = cone_color
        prog_render["m_model"].write(to_gl(cone_model))
        cone_vao_r.render(moderngl.TRIANGLES, vertices=cone_count)

        # sun marker: small white wireframe sphere at sun_pos
        sun_model = translate(float(sun_pos[0]), float(sun_pos[1]), float(sun_pos[2])) @ \
                    scale(0.4, 0.4, 0.4)
        prog_render["m_model"].write(to_gl(sun_model))
        prog_render["draw_sun"].value = 1
        ctx.wireframe = True
        sun_vao_r.render(moderngl.TRIANGLES, vertices=sun_count)
        ctx.wireframe = False
        prog_render["draw_sun"].value = 0

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()

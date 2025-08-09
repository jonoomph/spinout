# render.py
import moderngl
import pygame
import numpy as np
import random
from .colors import (
    FOG_DEFAULT_COLOR,
    FOG_DUST_COLOR,
    SKY_BOTTOM_COLOR,
    SKY_HORIZON_COLOR,
    SKY_TOP_COLOR,
    SUN_LIGHT_COLOR,
)

class RenderContext:
    def __init__(self, width, height):
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND | moderngl.PROGRAM_POINT_SIZE)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.ctx.blend_equation = moderngl.FUNC_ADD
        self.width = width
        self.height = height
        from .shaders import create_shaders
        self.prog, self.prog2d, self.prog_lit, self.prog_tex = create_shaders(self.ctx)
        if 'point_size' in self.prog:
            self.prog['point_size'].value = 1.0
        self.ortho = np.eye(4, dtype='f4')  # Identity for NDC
        self.mode = 1  # 0=wireframe,1=textured
        self.light_dir = np.array([0.5, 1.0, 0.3], dtype='f4')
        self.light_color = np.array(SUN_LIGHT_COLOR, dtype='f4')
        self.camera_pos = np.zeros(3, dtype='f4')
        self.fog_density = 0.0
        self.fog_color = np.array(FOG_DEFAULT_COLOR, dtype='f4')
        self.wetness = 0.0
        self.road_noise = 0.0
        self.terrain_mode = 0
        hud_quad_data = np.array([
            0.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 1.0,
            1.0, 0.0, 1.0, 0.0,
            1.0, 1.0, 1.0, 1.0,
            0.0, 1.0, 0.0, 1.0,
        ], dtype='f4')
        self.hud_vbo = self.ctx.buffer(hud_quad_data.tobytes())
        self.hud_vao = self.ctx.vertex_array(self.prog2d, self.hud_vbo, 'in_pos', 'in_tex')
        self.hud_tex = None
        self.main_vbo = None
        self.shock_vbo = None
        self.main_vao = None
        self.shock_vao = None
        self.model_vbo = None
        self.model_vao = None
        self.model_edge_vbo = None
        self.model_edge_vao = None
        self.car_model_tex = None

        # sky + stars geometry buffers generated per session
        self.sky_vbo = None
        self.sky_vao = None
        self.stars_vbo = None
        self.stars_vao = None
        self._generate_sky()

    def _generate_sky(self):
        """
        Daylight & twilight only (no full night).
        Rich, realistic gradients + faint twilight stars.
        Exports:
          - self.base_fog_color (horizon RGB)
          - self.sky_brightness (0..1 approx)
          - self.sky_vao / self.stars_vao
        """
        rng = np.random.default_rng()

        # ---------- choose time bucket (no deep night) ----------
        # centers near sunrise, morning, noon, afternoon, evening, sunset
        buckets = np.array([0.22, 0.33, 0.50, 0.60, 0.68, 0.78], dtype="f4")
        stds = np.array([0.015, 0.035, 0.030, 0.035, 0.030, 0.015], dtype="f4")
        weights = np.array([1.0, 1.1, 1.4, 1.1, 1.0, 1.0], dtype="f4");
        weights /= weights.sum()
        i = int(rng.choice(len(buckets), p=weights))
        t = float(np.clip(rng.normal(buckets[i], stds[i]), 0.12, 0.88))  # [0,1), 0.25≈sunrise, 0.75≈sunset

        # atmosphere knobs (kept conservative so we don’t go gray/flat)
        overcast = float(np.clip(rng.normal(0.30, 0.18), 0.0, 0.8))  # 0 clear → 0.8 quite cloudy
        turbidity = float(np.clip(rng.normal(2.0, 0.5), 1.2, 3.5))  # higher = hazier/whiter horizon

        # twilight strength around 0.25 / 0.75 (dawn/sunset)
        tw = float(np.exp(-((t - 0.25) / 0.045) ** 2) + np.exp(-((t - 0.75) / 0.045) ** 2))
        tw *= (1.0 - overcast)
        tw = float(np.clip(tw, 0.0, 1.0))

        # approximate “day factor” from solar elevation (no night clamp)
        sun_elev = np.sin(t * 2.0 * np.pi)
        day_fac = float(np.clip(sun_elev, 0.25, 1.0))  # never dim like night

        # ---------- base clear colors (linear-ish RGB) ----------
        # Zenith blue strengthens with day, horizon is warm at twilight, cool otherwise.
        top_clear = np.array([0.10, 0.30 + 0.30 * day_fac, 0.70 + 0.25 * day_fac, 1.0], dtype="f4")
        horizon_cool = np.array([0.70, 0.86, 1.00, 1.0], dtype="f4")
        horizon_warm = np.array([1.00, 0.58, 0.36, 1.0], dtype="f4")
        bottom_clear = np.array([0.86, 0.93, 1.00, 1.0], dtype="f4")  # light ground-scatter

        # Blend warm/cool horizon by twilight
        horizon = horizon_cool * (1.0 - 0.85 * tw) + horizon_warm * (0.85 * tw)
        top = top_clear.copy()
        bottom = bottom_clear.copy()

        # ---------- overcast/turbidity ----------
        oc_top = np.array([0.60, 0.64, 0.69, 1.0], dtype="f4")
        oc_horizon = np.array([0.72, 0.75, 0.78, 1.0], dtype="f4")
        oc_bottom = np.array([0.76, 0.78, 0.80, 1.0], dtype="f4")

        top = top * (1.0 - overcast) + oc_top * overcast
        horizon = horizon * (1.0 - overcast) + oc_horizon * overcast
        bottom = bottom * (1.0 - overcast) + oc_bottom * overcast

        haze = float((turbidity - 1.0) / 3.0);
        haze = np.clip(haze, 0.0, 1.0)
        horizon = horizon * (1.0 - 0.18 * haze) + bottom * (0.18 * haze)  # haze pulls horizon toward bottom
        top = top * (1.0 - 0.06 * haze) + horizon * (0.06 * haze)

        # small saturation jitter to keep skies varied (but not neon)
        sat_jit = float(rng.uniform(-0.04, 0.05) * (1.0 - overcast))

        def tweak_sat(c):
            rgb = c[:3]
            lum = float(np.dot(rgb, [0.2126, 0.7152, 0.0722]))
            rgb = lum + (rgb - lum) * (1.0 + sat_jit)
            return np.array([*np.clip(rgb, 0.0, 1.0), 1.0], dtype="f4")

        top, horizon, bottom = tweak_sat(top), tweak_sat(horizon), tweak_sat(bottom)

        # luminance floors so zenith/horizon never go near-black from jitter
        def ensure_luma(c, min_l):
            rgb = c[:3].copy()
            lum = float(np.dot(rgb, [0.2126, 0.7152, 0.0722]))
            if lum < min_l:
                rgb = np.clip(rgb + (min_l - lum), 0.0, 1.0)
            return np.array([*rgb, 1.0], dtype="f4")

        horizon = ensure_luma(horizon, 0.60)
        top = ensure_luma(top, 0.48)

        # exports used by fog/lighting
        self.base_fog_color = horizon[:3].astype("f4")
        self.sky_brightness = float(np.dot(self.base_fog_color, [0.2126, 0.7152, 0.0722]))

        # ---------- banding-free vertical gradient (curved) ----------
        rows = 64  # smoother than 32, still cheap
        verts = []

        def lerp(a, b, k):
            return a * (1.0 - k) + b * k

        # curve shaping: y in [-1,1] → u in [0,1] with extra weight near horizon
        def curve(u):
            # cubic smoothstep + slight bias to emphasize color change near horizon
            u = u * u * (3.0 - 2.0 * u)
            return np.clip(0.85 * u + 0.15 * u * u, 0.0, 1.0)

        for i in range(rows):
            y0 = -1.0 + 2.0 * (i / rows)
            y1 = -1.0 + 2.0 * ((i + 1) / rows)
            u0 = curve((y0 + 1.0) * 0.5)  # 0 bottom, 0.5 horizon, 1 top
            u1 = curve((y1 + 1.0) * 0.5)

            # blend: bottom→horizon up to ~0.5, then horizon→top
            def gc(u):
                k = np.clip(u * 2.0, 0.0, 1.0)  # 0..1 for bottom→horizon
                j = np.clip((u - 0.5) * 2.0, 0.0, 1.0)  # 0..1 for horizon→top
                col = lerp(bottom, horizon, k)
                col = lerp(col, top, j)
                return col

            c0 = gc(u0);
            c1 = gc(u1)
            verts += [-1.0, y0, 0.0, *c0, 1.0, y0, 0.0, *c0, -1.0, y1, 0.0, *c1,
                      -1.0, y1, 0.0, *c1, 1.0, y0, 0.0, *c0, 1.0, y1, 0.0, *c1]

        data = np.asarray(verts, dtype="f4")
        if self.sky_vbo is not None: self.sky_vbo.release()
        self.sky_vbo = self.ctx.buffer(data.tobytes())
        self.sky_vao = self.ctx.vertex_array(self.prog, self.sky_vbo, "in_vert", "in_color")

        # ---------- twilight stars (very faint), never on horizon ----------
        star_factor = (1.0 - overcast) * tw
        star_count = int(180 * star_factor)  # 0..~180
        if star_count > 0:
            xs = rng.uniform(-1.0, 1.0, star_count)
            ys = rng.uniform(0.20, 1.0, star_count)  # keep off horizon band
            mags = rng.uniform(0.6, 1.0, star_count) ** 2.0
            alpha = float(np.clip(0.14 * star_factor, 0.0, 0.18))
            cols = np.column_stack([mags, mags, mags, np.full(star_count, alpha, dtype="f4")]).astype("f4")
            v = np.column_stack([xs, ys, np.zeros(star_count, dtype="f4"), cols]).astype("f4").ravel()
            if self.stars_vbo is not None: self.stars_vbo.release()
            self.stars_vbo = self.ctx.buffer(v.tobytes())
            self.stars_vao = self.ctx.vertex_array(self.prog, self.stars_vbo, "in_vert", "in_color")
        else:
            self.stars_vbo = None
            self.stars_vao = None

        # keep fog/lighting in sync with the horizon; weather may tweak later
        self.fog_color = self.base_fog_color.copy()
        self.light_color = np.array([0.95, 0.95, 0.92], dtype="f4") * (0.70 + 0.30 * (1.0 - overcast))

    def set_camera(self, pos):
        self.camera_pos = np.array(pos, dtype='f4')

    def setup_weather(self, weather, terrain_type, road_type):
        self.wetness = 1.0 if weather == 'wet' else 0.0

        # Sky brightness from _generate_sky() controls base falloff
        sky_brightness = getattr(self, "sky_brightness", 1.0)
        self.fog_density = (1.0 - sky_brightness) * 0.02

        # Start fog from the sky's horizon tone
        horizon = np.array(getattr(self, "base_fog_color", FOG_DEFAULT_COLOR), dtype="f4")

        # Weather/terrain adjustments
        if self.wetness > 0.0:
            self.fog_density += 0.015
        elif terrain_type in ("sand", "gravel", "dirt"):
            self.fog_density += 0.001
            # Dusty terrain nudges the tint sandy
            horizon = 0.5 * horizon + 0.5 * np.array(FOG_DUST_COLOR, dtype="f4")

        # Keep it sane
        self.fog_density = float(min(self.fog_density, 0.05))

        # Final fog tint: as density rises, drift from horizon toward a neutral/white fog
        neutral = np.array(FOG_DEFAULT_COLOR, dtype="f4")
        d = self.fog_density / 0.05  # 0..1
        # Bias: mostly horizon at low fog, more neutral at high fog
        self.fog_color = ((1.0 - 0.6 * d) * horizon + (0.6 * d) * neutral).astype("f4")

        terrain_map = {'grass': 1, 'dirt': 2, 'sand': 2, 'snow': 3}
        self.terrain_mode = terrain_map.get(terrain_type, 0)

        noise_map = {'asphalt': 2.5, 'concrete': 1.5, 'gravel': 3.5}
        self.road_noise = noise_map.get(road_type, 0.0)

    def render_weather(self, mvp, dt):
        pass

    def set_mode(self, mode):
        self.mode = mode
        self.ctx.wireframe = mode == 0

    def clear(self):
        self.ctx.viewport = (0, 0, self.width, self.height)
        self.ctx.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)
        self.render_skybox()

    def _apply_common_uniforms(self, prog):
        if 'cam_pos' in prog:
            prog['cam_pos'].value = tuple(self.camera_pos)
        if 'fog_density' in prog:
            prog['fog_density'].value = self.fog_density
        if 'fog_color' in prog:
            prog['fog_color'].value = tuple(self.fog_color)
        if 'wetness' in prog:
            prog['wetness'].value = self.wetness
        if 'light_color' in prog:
            prog['light_color'].value = tuple(self.light_color)
        if 'sky_brightness' in prog:
            prog['sky_brightness'].value = float(getattr(self, 'sky_brightness', 1.0))

    def render_terrain(self, terrain_vao, mvp, noise_scale=0.0, terrain_mode=None):
        self.ctx.line_width = 1.0
        prog = terrain_vao.program
        prog['mvp'].write(mvp.T.tobytes())
        self._apply_common_uniforms(prog)
        if 'noise_scale' in prog:
            prog['noise_scale'].value = noise_scale
        if 'terrain_mode' in prog:
            prog['terrain_mode'].value = self.terrain_mode if terrain_mode is None else terrain_mode
        if 'light_dir' in prog:
            prog['light_dir'].value = tuple(self.light_dir)
        terrain_vao.render(moderngl.TRIANGLES)

    def cycle_terrain_mode(self):
        self.terrain_mode = (self.terrain_mode + 1) % 4

    def render_signs(self, vao, mvp):
        """Render simple colored billboards such as sign posts."""
        was_wireframe = self.ctx.wireframe
        self.ctx.wireframe = False
        prog = vao.program
        prog['mvp'].write(mvp.T.tobytes())
        self._apply_common_uniforms(prog)
        vao.render(moderngl.TRIANGLES)
        self.ctx.wireframe = was_wireframe

    def render_billboard(self, vao, tex, mvp):
        """Render a textured billboard (e.g. the speed limit sign face)."""
        was_wireframe = self.ctx.wireframe
        self.ctx.wireframe = False
        prog = vao.program
        prog['mvp'].write(mvp.T.tobytes())
        self._apply_common_uniforms(prog)
        tex.use(0)
        prog['tex'] = 0
        vao.render(moderngl.TRIANGLES)
        self.ctx.wireframe = was_wireframe

    def render_car(self, vertices, mvp):
        main_vertices, shock_vertices = vertices
        self.prog['mvp'].write(mvp.T.tobytes())
        self._apply_common_uniforms(self.prog)
        if main_vertices:
            main_data = np.array(main_vertices, dtype='f4').tobytes()
            if self.main_vbo is None or len(main_data) > self.main_vbo.size:
                if self.main_vbo:
                    self.main_vbo.release()
                self.main_vbo = self.ctx.buffer(main_data)
                self.main_vao = self.ctx.vertex_array(self.prog, self.main_vbo, 'in_vert', 'in_color')
            else:
                self.main_vbo.write(main_data, offset=0)
            self.ctx.line_width = 1.0
            self.main_vao.render(moderngl.LINES)
        if shock_vertices:
            shock_data = np.array(shock_vertices, dtype='f4').tobytes()
            if self.shock_vbo is None or len(shock_data) > self.shock_vbo.size:
                if self.shock_vbo:
                    self.shock_vbo.release()
                self.shock_vbo = self.ctx.buffer(shock_data)
                self.shock_vao = self.ctx.vertex_array(self.prog, self.shock_vbo, 'in_vert', 'in_color')
            else:
                self.shock_vbo.write(shock_data, offset=0)
            self.ctx.line_width = 3.0
            self.shock_vao.render(moderngl.LINES)

    def render_car_model(self, vertices, mvp):
        """Draw either wireframe or textured model using explicit VAO layouts."""
        tri_vertices, edge_vertices = vertices

        if self.ctx.wireframe:
            if not edge_vertices:
                return
            data = np.asarray(edge_vertices, 'f4').tobytes()
            if self.model_edge_vbo is None or len(data) > self.model_edge_vbo.size:
                if self.model_edge_vbo:
                    self.model_edge_vbo.release()
                self.model_edge_vbo = self.ctx.buffer(data)
                # 3 floats pos + 4 floats color
                self.model_edge_vao = self.ctx.vertex_array(
                    self.prog,
                    [(self.model_edge_vbo, '3f 4f', 'in_vert', 'in_color')],
                )
            else:
                self.model_edge_vbo.write(data, 0)
            self.prog['mvp'].write(mvp.T.tobytes())
            self._apply_common_uniforms(self.prog)
            self.ctx.line_width = 1.0
            self.model_edge_vao.render(moderngl.LINES)
            return

        if not tri_vertices:
            return
        data = np.asarray(tri_vertices, 'f4').tobytes()
        if self.model_vbo is None or len(data) > self.model_vbo.size:
            if self.model_vbo:
                self.model_vbo.release()
            self.model_vbo = self.ctx.buffer(data)
            # 3 floats pos + 2 floats UV
            self.model_vao = self.ctx.vertex_array(
                self.prog_tex,
                [(self.model_vbo, '3f 2f', 'in_vert', 'in_tex')],
            )
        else:
            self.model_vbo.write(data, 0)

        self.prog_tex['mvp'].write(mvp.T.tobytes())
        self._apply_common_uniforms(self.prog_tex)
        if self.car_model_tex:
            self.car_model_tex.use(0)
            self.prog_tex['tex'] = 0
        self.model_vao.render(moderngl.TRIANGLES)

    def render_skybox(self):
        # Always render the skybox filled even if the main scene uses wireframe
        was_wireframe = self.ctx.wireframe
        self.ctx.wireframe = False
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.prog['mvp'].write(self.ortho.tobytes())

        # Apply same fog/tint uniforms so the sky is hazed like the scene
        self._apply_common_uniforms(self.prog)

        if 'cam_pos' in self.prog:
            # Push sky back so exponential fog actually attenuates it
            self.prog['cam_pos'].value = (0.0, 0.0, -500.0)  # was -100.0

        if 'wetness' in self.prog:
            self.prog['wetness'].value = 0.0
        if 'noise_scale' in self.prog:
            self.prog['noise_scale'].value = 0.0
        if 'terrain_mode' in self.prog:
            self.prog['terrain_mode'].value = 0

        self.sky_vao.render(moderngl.TRIANGLES)

        if self.stars_vao is not None:
            if 'point_size' in self.prog:
                self.prog['point_size'].value = 2.0
            self.stars_vao.render(moderngl.POINTS)
            if 'point_size' in self.prog:
                self.prog['point_size'].value = 1.0

        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.wireframe = was_wireframe

    def render_hud(self, hud_surf):
        self.ctx.viewport = (0, 0, self.width, self.height)
        hud_data = pygame.image.tostring(hud_surf, 'RGBA', True)
        if self.hud_tex is None:
            self.hud_tex = self.ctx.texture((self.width, self.height), 4, hud_data)
            self.hud_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        else:
            self.hud_tex.write(hud_data)
        was_wireframe = self.ctx.wireframe
        self.ctx.wireframe = False
        self.hud_tex.use(0)
        self.prog2d['tex'] = 0
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.hud_vao.render(moderngl.TRIANGLES)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.wireframe = was_wireframe
